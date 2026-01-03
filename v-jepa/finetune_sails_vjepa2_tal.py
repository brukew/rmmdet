#!/usr/bin/env python
"""Fine-tune V-JEPA2 for SAILS TAL (Temporal Action Localization) window classification.

Adapted from `finetune_sails_vjepa2_cv_crop.py` for TAL window-based training with
5 classes (4 RMM types + background) and class-weighted loss balancing.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional
    h5py = None
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional
    wandb = None

logger = logging.getLogger(__name__)

# ==============================================================================
# TAL-specific constants
# ==============================================================================
TAL_LABEL_MAP = {
    0: "hands_flapping",
    1: "jumping",
    2: "rocking",
    3: "spinning",
    4: "background",  # primary_label=-1 maps here
}

TAL_ID2LABEL = TAL_LABEL_MAP.copy()
TAL_LABEL2ID = {v: k for k, v in TAL_LABEL_MAP.items()}

DEFAULT_TAL_CSV_DIR = Path("/orcd/data/satra/001/users/brukew/actreg/dataprep/tal/splits_cv_4class")
DEFAULT_TAL_CLIPS_ROOT = Path("/orcd/scratch/bcs/001/brukew/sails/tal_windows_4class/canonical_clips")

# Cropping defaults (same as original)
DEFAULT_PARSED_CSV = Path("/orcd/data/satra/001/users/brukew/actreg/dataprep/rmm_sam3_parsed.csv")
DEFAULT_MASK_CACHE_BASE = Path("/orcd/scratch/bcs/001/sensein/sails/cache_for_tracking")
DEFAULT_MASK_MODEL = "facebook-sam3"
DEFAULT_MASK_PROMPT = "person"
DEFAULT_CROP_PADDING = 20


@dataclass
class CropConfig:
    """Configuration for SAM3-based child cropping."""
    enabled: bool = False
    sam3_parsed_csv: Path = DEFAULT_PARSED_CSV
    mask_cache_base: Path = DEFAULT_MASK_CACHE_BASE
    mask_model: str = DEFAULT_MASK_MODEL
    mask_prompt: str = DEFAULT_MASK_PROMPT
    padding: int = DEFAULT_CROP_PADDING
    rotation_override: Optional[int] = None
    fallback: str = "full"  # "full" or "skip"
    video_meta_json: Optional[Path] = None


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger for job output."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger.setLevel(numeric_level)


# ==============================================================================
# Cropping utilities (unchanged from original)
# ==============================================================================
def load_parsed_sam3_csv(csv_path: Path) -> Dict[str, Dict]:
    """Load SAM3 child ID intervals from parsed CSV."""
    rows: Dict[str, Dict] = {}
    if not csv_path.exists():
        logger.warning("Parsed SAM3 CSV not found at %s; cropping will be disabled.", csv_path)
        return rows
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["intervals"] = json.loads(row.get("child_sam3_ids", "[]"))
            except Exception:
                row["intervals"] = []
            fname = row.get("FileName") or row.get("filename")
            if fname:
                rows[fname] = row
    return rows


def child_ids_for_time(row: Dict, time_sec: float) -> List[int]:
    """Get child SAM3 object IDs active at a given timestamp."""
    ids: List[int] = []
    for iv in row.get("intervals", []):
        start = iv.get("start_sec", 0)
        end = iv.get("end_sec") if iv.get("end_sec") is not None else float("inf")
        try:
            start_f = float(start)
        except Exception:
            start_f = 0.0
        try:
            end_f = float(end)
        except Exception:
            end_f = float("inf")
        if start_f <= time_sec < end_f:
            try:
                ids.append(int(iv["id"]))
            except Exception:
                pass
    return ids


def get_video_rotation(video_path: Path) -> int:
    """Get rotation metadata from video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream_side_data=rotation",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.stdout.strip():
            return int(float(result.stdout.strip()))
    except Exception:
        pass
    return 0


def rotate_boxes(boxes: np.ndarray, rotation: int, width: int, height: int) -> np.ndarray:
    """Rotate bounding boxes to compensate for video rotation metadata."""
    if rotation == 0:
        return boxes
    rotated = boxes.copy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        if rotation in [-90, 270]:
            rotated[i] = [y1, width - x2, y2, width - x1]
        elif rotation in [90, -270]:
            rotated[i] = [height - y2, x1, height - y1, x2]
        elif rotation in [180, -180]:
            rotated[i] = [width - x2, height - y2, width - x1, height - y1]
    return rotated


def load_mask_cache(
    base: Path,
    video_basename: str,
    prompt: str = DEFAULT_MASK_PROMPT,
    model_name: str = DEFAULT_MASK_MODEL,
    cache_path: Optional[Path] = None,
) -> Optional[Dict]:
    """Load SAM3 mask cache HDF5 file for a video."""
    if h5py is None:
        return None

    def _slugify(text: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text.strip())
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug.strip("-") or "none"

    prompt_slug = _slugify(prompt)
    base_dir = base / "masks" / video_basename
    candidates = []
    if cache_path:
        cp = Path(cache_path)
        if cp.exists() and cp.is_file():
            candidates.append(cp)
    candidates.append(base_dir / f"{model_name}__prompt-{prompt_slug}.h5")
    candidates.extend(sorted(base_dir.glob("*.h5")))

    errors = []
    for cand in candidates:
        if not cand.exists() or not cand.is_file():
            continue
        try:
            with h5py.File(cand, "r") as f:
                frame_indices = []
                obj_ids = {}
                boxes = {}
                scores = {}
                for key in f.keys():
                    if not key.startswith("frame_"):
                        continue
                    try:
                        frame_idx = int(key.split("_", 1)[1])
                    except ValueError:
                        continue
                    frame_indices.append(frame_idx)
                    grp = f[key]
                    obj_ids[frame_idx] = grp["obj_ids"][:]
                    boxes[frame_idx] = grp["boxes"][:]
                    scores[frame_idx] = grp["scores"][:]
                if not frame_indices:
                    errors.append((cand, "no frame_* groups"))
                    continue
                attrs = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in f.attrs.items()}
                return {
                    "path": cand,
                    "frame_indices": np.array(sorted(frame_indices)),
                    "obj_ids": obj_ids,
                    "boxes": boxes,
                    "scores": scores,
                    "attrs": attrs,
                }
        except Exception as exc:
            errors.append((cand, str(exc)))
            continue

    if errors:
        first = errors[0]
        logger.warning("Failed to load cache(s) for %s (first: %s -> %s)", video_basename, first[0], first[1])
    return None


def load_video_meta_json(path: Optional[Path]) -> Dict[str, Dict]:
    """Load video metadata JSON for rotation and cache path info."""
    if not path:
        return {}
    if not path.exists():
        logger.warning("Video meta JSON not found at %s", path)
        return {}
    try:
        data = json.loads(path.read_text())
        records = data.get("records") if isinstance(data, dict) else data
        meta = {}
        for rec in records or []:
            fname = rec.get("FileName") or rec.get("filename")
            if fname:
                meta[fname] = rec
        logger.info("Loaded video meta for %d videos from %s", len(meta), path)
        return meta
    except Exception as exc:
        logger.warning("Could not parse video meta JSON %s: %s", path, exc)
        return {}


# ==============================================================================
# TAL-specific data loading
# ==============================================================================
def load_tal_split(
    csv_paths: Sequence[Path],
    clips_root: Path,
    include_background: bool = True,
    exclude_windows: Optional[set] = None,
) -> Tuple[List[Dict], List[Tuple[str, Path]]]:
    """
    Load TAL window records from CSV files.
    
    Args:
        csv_paths: List of TAL window CSV files to load.
        clips_root: Root directory containing window clips (flat structure).
        include_background: Whether to include background windows (primary_label=-1).
        exclude_windows: Optional set of window_ids to exclude.
    
    Returns:
        Tuple of (records list, missing clips list).
    """
    records: List[Dict] = []
    missing: List[Tuple[str, Path]] = []
    excluded_count = 0
    background_excluded = 0
    
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            window_id = row.get("window_id")
            primary_label = row.get("primary_label")
            
            if pd.isna(window_id) or pd.isna(primary_label):
                continue
            
            primary_label = int(primary_label)
            
            # Skip background if not included
            if primary_label == -1 and not include_background:
                background_excluded += 1
                continue
            
            # Skip excluded windows
            if exclude_windows and str(window_id) in exclude_windows:
                excluded_count += 1
                continue
            
            # Map primary_label to class index (including -1 -> 4 for background)
            if primary_label == -1:
                label_id = 4  # background class
            else:
                label_id = primary_label  # 0-3 for RMM classes
            
            # TAL clips are in flat directory: clips_root/{window_id}.mp4
            clip_path = clips_root / f"{window_id}.mp4"
            if not clip_path.exists():
                missing.append((str(window_id), clip_path))
                continue
            
            rec = {
                "window_id": window_id,
                "label_id": label_id,
                "label_name": TAL_ID2LABEL[label_id],
                "clip": clip_path,
                "video_key": row.get("video_key"),
                "filename": row.get("filename"),
                "child_id": row.get("child_id"),
                "start_sec": pd.to_numeric(row.get("start_sec"), errors="coerce"),
                "end_sec": pd.to_numeric(row.get("end_sec"), errors="coerce"),
                "is_background": int(row.get("is_background", 0)),
                "mask_severity": row.get("mask_severity"),
                "pose_severity": row.get("pose_severity"),
            }
            records.append(rec)
    
    if excluded_count > 0:
        logger.info("Excluded %d windows from split", excluded_count)
    if background_excluded > 0:
        logger.info("Excluded %d background windows (--no-background)", background_excluded)
    
    return records, missing


def build_video_label_counts(records: Sequence[Dict], key_field: str = "video_key") -> Dict[str, int]:
    """Count unique labels per video for mixed-video detection."""
    counts: Dict[str, set] = {}
    for rec in records:
        video_key = rec.get(key_field)
        if video_key is None:
            continue
        counts.setdefault(video_key, set()).add(rec["label_id"])
    return {k: len(v) for k, v in counts.items()}


def get_frames_per_clip(processor: VJEPA2VideoProcessor, override: int) -> int:
    """Determine frames per clip from processor or override."""
    if override and override > 0:
        return int(override)
    frames = (
        getattr(processor, "num_frames", None)
        or getattr(getattr(processor, "image_processor", processor), "num_frames", None)
        or getattr(getattr(processor, "feature_extractor", processor), "num_frames", None)
        or getattr(getattr(processor, "config", {}), "num_frames", None)
        or getattr(getattr(processor, "config", {}), "frames_per_clip", None)
        or 32
    )
    return int(frames)


# ==============================================================================
# Dataset
# ==============================================================================
class TALDataset(Dataset):
    """
    PyTorch Dataset for TAL window classification.
    
    Loads video clips and applies optional SAM3-based child cropping.
    Uses integer label IDs directly (0-4 for 5 classes).
    """
    
    def __init__(
        self,
        records: List[Dict],
        id2label: Dict[int, str],
        frames_per_clip: int,
        video_label_counts: Dict,
        crop_cfg: Optional[CropConfig] = None,
        sam3_rows: Optional[Dict[str, Dict]] = None,
        video_meta: Optional[Dict[str, Dict]] = None,
    ):
        self.records = records
        self.id2label = id2label
        self.frames_per_clip = frames_per_clip
        self.video_label_counts = video_label_counts
        self.crop_cfg = crop_cfg or CropConfig(enabled=False)
        self.sam3_rows = sam3_rows or {}
        self.video_meta = video_meta or {}
        self._mask_cache_store: Dict[str, Optional[Dict]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _sample_indices(self, total: int) -> np.ndarray:
        """Sample frame indices uniformly across clip duration."""
        if total <= 0:
            return np.zeros(self.frames_per_clip, dtype="int64")
        return np.round(np.linspace(0, total - 1, self.frames_per_clip)).astype("int64")

    def _apply_crop(self, rec: Dict, frames: np.ndarray, indices: np.ndarray, fps: float):
        """Apply SAM3-based child cropping to frames."""
        cfg = self.crop_cfg
        meta = {
            "crop_applied": False,
            "crop_box": None,
            "crop_matched_ids": [],
            "crop_cache_frame": None,
            "crop_rotation": None,
            "crop_reason": "",
            "crop_cache_path": None,
        }
        if not cfg.enabled:
            return frames, meta
        if h5py is None:
            meta["crop_reason"] = "h5py_missing"
            return frames, meta
        if fps is None or fps <= 0:
            meta["crop_reason"] = "fps_missing"
            return frames, meta

        fname = rec.get("filename") or Path(rec.get("clip")).name
        sam3_row = self.sam3_rows.get(fname)
        if not sam3_row:
            meta["crop_reason"] = "no_sam3_row"
            return frames, meta

        meta_rec = self.video_meta.get(fname, {})
        cache_path_override = None
        if meta_rec.get("mask_cache_path"):
            try:
                cache_path_override = Path(meta_rec["mask_cache_path"])
            except Exception:
                cache_path_override = None

        basename = f"{Path(fname).stem}_segmented"
        cache_key = (basename, str(cache_path_override) if cache_path_override else None, cfg.mask_prompt, cfg.mask_model)
        if cache_key not in self._mask_cache_store:
            self._mask_cache_store[cache_key] = load_mask_cache(
                cfg.mask_cache_base,
                basename,
                prompt=cfg.mask_prompt,
                model_name=cfg.mask_model,
                cache_path=cache_path_override,
            )
        cache = self._mask_cache_store.get(cache_key)
        if not cache:
            meta["crop_reason"] = "no_cache"
            return frames, meta

        attrs = cache.get("attrs") or {}
        base_w = attrs.get("width")
        base_h = attrs.get("height")
        if base_w is None or base_h is None:
            base_w, base_h = frames.shape[2], frames.shape[1]

        if cfg.rotation_override is not None:
            rotation = cfg.rotation_override
        else:
            if "rotation_meta" in meta_rec:
                rot_val = meta_rec.get("rotation_meta")
                try:
                    rotation = int(rot_val) if rot_val is not None else 0
                except Exception:
                    rotation = 0
            else:
                rotation = get_video_rotation(Path(rec["clip"]))
        rotation *= -1  # align with notebook convention

        frame_w, frame_h = frames.shape[2], frames.shape[1]
        frame_indices_cache = cache["frame_indices"]
        xs: List[float] = []
        ys: List[float] = []
        matched_ids: List[int] = []
        cache_frame_used = None

        start_sec = rec.get("start_sec")
        if start_sec is None or (isinstance(start_sec, float) and np.isnan(start_sec)):
            start_sec = 0.0

        for pos, frame_idx_clip in enumerate(indices):
            offset_sec = float(frame_idx_clip) / fps
            abs_time = float(start_sec) + offset_sec
            child_ids = child_ids_for_time(sam3_row, abs_time)
            if not child_ids:
                continue

            frame_idx_est = int(round(abs_time * fps))
            closest_pos = int(np.argmin(np.abs(frame_indices_cache - frame_idx_est)))
            cache_frame_idx = int(frame_indices_cache[closest_pos])
            cache_frame_used = cache_frame_idx

            boxes = cache["boxes"][cache_frame_idx]
            obj_ids = cache["obj_ids"][cache_frame_idx]
            if boxes is None or len(boxes) == 0:
                continue

            boxes_rotated = boxes.copy()
            expected_w, expected_h = base_w, base_h
            if rotation:
                boxes_rotated = rotate_boxes(boxes_rotated, rotation, base_w, base_h)
                if rotation in (-90, 90, -270, 270):
                    expected_w, expected_h = base_h, base_w
                elif rotation in (-180, 180):
                    expected_w, expected_h = base_w, base_h

            if expected_w and expected_h and (frame_w != expected_w or frame_h != expected_h):
                scale_x = frame_w / expected_w
                scale_y = frame_h / expected_h
                boxes_rotated[:, [0, 2]] *= scale_x
                boxes_rotated[:, [1, 3]] *= scale_y

            frame_matches = [i for i, obj in enumerate(obj_ids) if int(obj) in child_ids]
            if frame_matches:
                matched_ids.extend([int(obj_ids[i]) for i in frame_matches])
                for det_idx in frame_matches:
                    x1, y1, x2, y2 = boxes_rotated[det_idx]
                    xs.extend([x1, x2])
                    ys.extend([y1, y2])

        if not xs or not ys:
            meta["crop_reason"] = "no_matching_ids"
            return frames, meta

        x1 = max(0, int(np.floor(min(xs) - cfg.padding)))
        y1 = max(0, int(np.floor(min(ys) - cfg.padding)))
        x2 = min(int(frame_w), int(np.ceil(max(xs) + cfg.padding)))
        y2 = min(int(frame_h), int(np.ceil(max(ys) + cfg.padding)))
        if x2 <= x1 or y2 <= y1:
            meta["crop_reason"] = "invalid_box"
            return frames, meta

        frames_cropped = frames[:, y1:y2, x1:x2, :]
        meta.update({
            "crop_applied": True,
            "crop_box": (x1, y1, x2, y2),
            "crop_matched_ids": sorted(set(matched_ids)),
            "crop_cache_frame": cache_frame_used,
            "crop_rotation": rotation,
            "crop_reason": "ok",
            "crop_cache_path": str(cache.get("path")) if cache.get("path") else None,
        })
        return frames_cropped, meta

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        try:
            vr = VideoReader(str(rec["clip"]), ctx=cpu(0), fault_tol=1)
            indices = self._sample_indices(len(vr))
            frames = vr.get_batch(indices).asnumpy()
            fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 0.0
        except Exception as exc:
            logger.warning("Bad clip %s: %s", rec["clip"], exc)
            return None

        label_id = rec["label_id"]
        video_key = rec.get("video_key")
        meta = {
            "window_id": rec.get("window_id"),
            "video_key": video_key,
            "label_id": label_id,
            "label_name": rec["label_name"],
            "is_background": rec.get("is_background", 0),
            "mask_severity": rec.get("mask_severity"),
            "pose_severity": rec.get("pose_severity"),
            "mixed_video": video_key in self.video_label_counts and self.video_label_counts[video_key] > 1,
        }
        if self.crop_cfg.enabled:
            frames, crop_meta = self._apply_crop(rec, frames, indices, fps)
            if self.crop_cfg.fallback == "skip" and not crop_meta.get("crop_applied"):
                return None
            meta.update(crop_meta)
        return frames, label_id, meta


def collate_fn(samples, processor: VJEPA2VideoProcessor):
    """Collate function for DataLoader."""
    samples = [s for s in samples if s is not None]
    if not samples:
        return None, None, None
    frame_batches, labels, metas = zip(*samples)
    inputs = processor(list(frame_batches), return_tensors="pt")
    labels_tensor = torch.tensor(labels)
    return inputs, labels_tensor, list(metas)


# ==============================================================================
# Evaluation utilities
# ==============================================================================
def aggregate_preds(labels, preds, probs, metas, id2label: Dict[int, str]) -> pd.DataFrame:
    """Aggregate predictions into a DataFrame with metadata."""
    df = pd.DataFrame(metas)
    df["label_id"] = labels
    df["pred_top1"] = preds
    df["label_name"] = df["label_id"].map(id2label)
    df["pred_name"] = df["pred_top1"].map(id2label)
    if probs is not None:
        df["pred_conf"] = probs.max(axis=1)
        for i in range(probs.shape[1]):
            df[f"score_class{i}"] = probs[:, i]
    return df


def evaluate(loader: DataLoader, model: torch.nn.Module, device: torch.device, collect_probs: bool = False):
    """Evaluate model on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_metas: List[Dict] = []
    all_probs: List[torch.Tensor] = []
    with torch.no_grad():
        for inputs, labels, metas in loader:
            if inputs is None or labels is None:
                continue
            labels = labels.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_metas.extend(metas)
            if collect_probs:
                all_probs.append(torch.softmax(logits, dim=-1).cpu())
    acc = correct / max(total, 1)
    probs_tensor = torch.cat(all_probs, dim=0) if all_probs else None
    return acc, all_preds, all_labels, all_metas, probs_tensor


def evaluate_topk(loader: DataLoader, model: torch.nn.Module, device: torch.device, k: int = 2):
    """Evaluate with top-k accuracy."""
    model.eval()
    correct_top1 = 0
    correct_topk = 0
    total = 0

    all_preds_top1: List[int] = []
    all_preds_topk: List[torch.Tensor] = []
    all_labels: List[int] = []
    all_probs: List[torch.Tensor] = []
    all_metas: List[Dict] = []

    with torch.no_grad():
        for inputs, labels, metas in loader:
            if inputs is None or labels is None:
                continue

            labels = labels.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

            preds_top1 = logits.argmax(-1)
            correct_top1 += (preds_top1 == labels).sum().item()

            _, topk_indices = torch.topk(probs, k, dim=-1)
            correct_topk += torch.any(topk_indices == labels.unsqueeze(-1), dim=-1).sum().item()

            total += labels.size(0)

            all_preds_top1.extend(preds_top1.cpu().tolist())
            all_preds_topk.append(topk_indices.cpu())
            all_labels.extend(labels.cpu().tolist())
            all_probs.append(probs.cpu())
            all_metas.extend(metas)

    top1_acc = correct_top1 / max(total, 1)
    topk_acc = correct_topk / max(total, 1)

    all_preds_topk_tensor = torch.cat(all_preds_topk, dim=0) if all_preds_topk else torch.tensor([])
    all_probs_tensor = torch.cat(all_probs, dim=0) if all_probs else torch.tensor([])

    return {
        "top1_acc": top1_acc,
        f"top{k}_acc": topk_acc,
        "improvement": topk_acc - top1_acc,
        "preds_top1": all_preds_top1,
        "preds_topk": all_preds_topk_tensor,
        "labels": all_labels,
        "probs": all_probs_tensor,
        "metas": all_metas,
    }


def compute_classification_metrics(labels, preds, probs, id2label: Dict[int, str]):
    """Compute comprehensive classification metrics."""
    class_names = [id2label[i] for i in range(len(id2label))]
    label_ids = list(range(len(id2label)))
    report = classification_report(
        labels,
        preds,
        labels=label_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    prec_rec_f1 = precision_recall_fscore_support(
        labels, preds, labels=label_ids, average=None, zero_division=0
    )
    per_class = pd.DataFrame({
        "class": class_names,
        "precision": prec_rec_f1[0],
        "recall": prec_rec_f1[1],
        "f1": prec_rec_f1[2],
        "support": prec_rec_f1[3],
    })

    # PR curves for top classes by support
    y_true = np.array(labels)
    pr_curves = {}
    if probs is not None and len(class_names) > 0:
        supports = per_class.set_index("class")["support"]
        top_classes = supports.sort_values(ascending=False).head(min(5, len(class_names))).index.tolist()
        for cls in top_classes:
            cls_id = class_names.index(cls)
            y_bin = (y_true == cls_id).astype(int)
            scores = probs[:, cls_id]
            prec, rec, _ = precision_recall_curve(y_bin, scores)
            ap = average_precision_score(y_bin, scores)
            pr_curves[cls] = {"precision": prec, "recall": rec, "ap": ap}

    def _avg(key: str) -> Dict[str, float]:
        return report.get(key, {"f1-score": 0.0, "precision": 0.0, "recall": 0.0})

    macro = _avg("macro avg")
    micro = _avg("micro avg")
    weighted = _avg("weighted avg")

    summary = {
        "macro_f1": macro["f1-score"],
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "micro_f1": micro["f1-score"],
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "weighted_f1": weighted["f1-score"],
    }
    return summary, per_class, pr_curves


def aggregate_video_predictions(
    metas_df: pd.DataFrame, labels: np.ndarray, preds: np.ndarray, probs: Optional[np.ndarray], id2label: Dict[int, str]
):
    """Aggregate window-level predictions into video-level predictions."""
    if metas_df is None or metas_df.empty or "video_key" not in metas_df:
        logger.warning("No video_key metadata available; skipping video-level aggregation.")
        return None

    valid = metas_df["video_key"].notna()
    if not valid.any():
        logger.warning("video_key missing for all samples; skipping video-level aggregation.")
        return None

    probs_available = probs is not None and len(probs) == len(labels)
    rows = []
    for vid, grp in metas_df[valid].reset_index().groupby("video_key"):
        clip_indices = grp["index"].to_numpy()
        label_ids = labels[clip_indices]
        pred_ids = preds[clip_indices]

        true_label = Counter(label_ids).most_common(1)[0][0]
        if probs_available:
            mean_probs = probs[clip_indices].mean(axis=0)
            pred_label = int(np.argmax(mean_probs))
            pred_conf = float(mean_probs.max())
        else:
            pred_label = Counter(pred_ids).most_common(1)[0][0]
            pred_conf = None
            mean_probs = None

        rows.append({
            "video_key": vid,
            "true_id": int(true_label),
            "pred_id": int(pred_label),
            "true_name": id2label[int(true_label)],
            "pred_name": id2label[int(pred_label)],
            "n_windows": len(grp),
            "pred_conf": pred_conf,
            "mean_probs": mean_probs,
        })

    if not rows:
        return None

    video_df = pd.DataFrame(rows)
    probs_video = None
    if probs_available:
        probs_video = np.stack(video_df["mean_probs"].to_numpy(), axis=0)

    return {
        "df": video_df,
        "labels": video_df["true_id"].to_numpy(),
        "preds": video_df["pred_id"].to_numpy(),
        "probs": probs_video,
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_ids: List[int],
    label_names: List[str],
    out_path: Path,
    normalize: Optional[str] = "true",
) -> None:
    """Save confusion matrix visualization."""
    if y_true is None or y_pred is None or len(y_true) == 0:
        logger.warning("Empty inputs; skipping confusion matrix.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=label_ids,
        display_labels=label_names,
        normalize=normalize,
        values_format=".2f" if normalize else "d",
        ax=ax,
        colorbar=False,
    )
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", out_path)


def metrics_for_subset(mask, labels, preds, probs, id2label: Dict[int, str]):
    """Compute metrics for a subset of samples."""
    if mask.sum() == 0:
        return None
    l = labels[mask]
    p = preds[mask]
    pr = probs[mask] if probs is not None else None
    summary, _, _ = compute_classification_metrics(l, p, pr, id2label)
    summary["size"] = len(l)
    return summary


# ==============================================================================
# W&B logging
# ==============================================================================
class WandbAdapter:
    """Wrapper for Weights & Biases logging."""
    
    def __init__(self, mode: str, project: str, run_name: str, config: Dict):
        self.run = None
        self.enabled = False
        if mode == "disabled":
            logger.info("W&B logging disabled.")
            return
        if wandb is None:
            logger.warning("wandb not installed; disabling logging.")
            return
        self.run = wandb.init(project=project, name=run_name, config=config, mode=mode)
        self.enabled = True

    def log(self, data: Dict):
        if self.enabled:
            wandb.log(data)

    def log_table(self, name: str, dataframe: pd.DataFrame):
        if self.enabled and not dataframe.empty:
            wandb.log({name: wandb.Table(dataframe=dataframe)})

    def finish(self):
        if self.run is not None:
            self.run.finish()


# ==============================================================================
# Argument parsing
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune V-JEPA2 for SAILS TAL window classification (5 classes: 4 RMM + background)."
    )
    # Data paths
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_TAL_CSV_DIR,
        help="Directory containing TAL window CSVs (fold_*_train_windows.csv for CV, or train/val/test_windows.csv for single).",
    )
    parser.add_argument(
        "--clips-root",
        type=Path,
        default=DEFAULT_TAL_CLIPS_ROOT,
        help="Directory containing window clips ({window_id}.mp4).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/vjepa2_tal_cv"),
        help="Where to store per-fold checkpoints and summaries.",
    )
    
    # Split mode
    parser.add_argument(
        "--split-mode",
        choices=["cv", "single"],
        default="cv",
        help="Split mode: 'cv' for cross-validation folds, 'single' for train/val/test split.",
    )
    
    # Background class
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Exclude background windows (train on 4 RMM classes only).",
    )
    
    # Model
    parser.add_argument("--model-id", default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=32,
        help="Frames per clip; set <=0 to use the processor default.",
    )
    
    # Evaluation
    parser.add_argument("--topk", type=int, default=2, help="k for top-k evaluation.")
    
    # CV-specific
    parser.add_argument("--max-folds", type=int, default=None, help="Optional cap on folds to run (CV mode).")
    parser.add_argument(
        "--start-fold",
        type=int,
        default=0,
        help="Skip folds with index < start-fold (CV mode, useful for resuming).",
    )
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="If a fold output dir exists, load model/processor and only run evaluation/summary.",
    )
    
    # W&B
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="Weights & Biases logging mode.",
    )
    parser.add_argument("--wandb-project", default="vjepa-tal", help="W&B project name.")
    parser.add_argument(
        "--run-prefix",
        default="vjepa2-tal",
        help="Prefix for W&B run names.",
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity.",
    )
    
    # Cropping
    parser.add_argument("--enable-crop", action="store_true", help="Enable SAM3-based cropping inside the dataloader.")
    parser.add_argument(
        "--sam3-parsed-csv",
        type=Path,
        default=DEFAULT_PARSED_CSV,
        help="Path to rmm_sam3_parsed.csv (with child_sam3_ids).",
    )
    parser.add_argument(
        "--mask-cache-base",
        type=Path,
        default=DEFAULT_MASK_CACHE_BASE,
        help="Base dir containing mask caches (MaskCacheManager layout).",
    )
    parser.add_argument("--mask-model", default=DEFAULT_MASK_MODEL, help="Mask model name prefix in cache files.")
    parser.add_argument("--mask-prompt", default=DEFAULT_MASK_PROMPT, help="Mask prompt used when saving caches.")
    parser.add_argument("--crop-padding", type=int, default=DEFAULT_CROP_PADDING, help="Pixels of padding around boxes.")
    parser.add_argument(
        "--rotation-override",
        type=int,
        default=None,
        help="Force rotation correction in degrees (e.g., 90/-90). If unset, use video metadata.",
    )
    parser.add_argument(
        "--crop-fallback",
        choices=["full", "skip"],
        default="full",
        help="When no crop is found: full=keep full frame, skip=drop sample.",
    )
    parser.add_argument(
        "--video-meta-json",
        type=Path,
        default=Path("/orcd/data/satra/001/users/brukew/actreg/dataprep/video_meta.json"),
        help="JSON from save_video_metadata_cache for reuse (rotation/cache info).",
    )
    
    return parser.parse_args()


# ==============================================================================
# Main training loop
# ==============================================================================
def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if args.wandb_mode != "disabled" and wandb is None:
        logger.warning('wandb not installed; set --wandb-mode disabled or install wandb to log runs.')

    include_background = not args.no_background
    num_classes = 5 if include_background else 4
    
    # Build label mappings based on whether background is included
    if include_background:
        id2label = TAL_ID2LABEL.copy()
        label2id = TAL_LABEL2ID.copy()
    else:
        id2label = {i: TAL_ID2LABEL[i] for i in range(4)}
        label2id = {v: k for k, v in id2label.items()}
    
    logger.info("TAL mode: %d classes | background=%s", num_classes, include_background)
    logger.info("Labels: %s", id2label)

    # Cropping config
    crop_cfg = CropConfig(
        enabled=args.enable_crop,
        sam3_parsed_csv=args.sam3_parsed_csv,
        mask_cache_base=args.mask_cache_base,
        mask_model=args.mask_model,
        mask_prompt=args.mask_prompt,
        padding=args.crop_padding,
        rotation_override=args.rotation_override,
        fallback=args.crop_fallback,
        video_meta_json=args.video_meta_json,
    )
    sam3_rows = load_parsed_sam3_csv(args.sam3_parsed_csv) if crop_cfg.enabled else {}
    video_meta = load_video_meta_json(args.video_meta_json) if crop_cfg.enabled else {}
    if crop_cfg.enabled:
        logger.info(
            "Cropping enabled | parsed_csv=%s | cache_base=%s | model=%s | prompt=%s | padding=%d | fallback=%s",
            args.sam3_parsed_csv,
            args.mask_cache_base,
            args.mask_model,
            args.mask_prompt,
            args.crop_padding,
            args.crop_fallback,
        )

    # Discover CSVs based on split mode
    if args.split_mode == "cv":
        train_csvs = sorted(args.csv_dir.glob("fold_*_train_windows.csv"))
        val_csvs = sorted(args.csv_dir.glob("fold_*_val_windows.csv"))
        if not train_csvs or not val_csvs or len(train_csvs) != len(val_csvs):
            raise RuntimeError(f"Found {len(train_csvs)} train CSVs and {len(val_csvs)} val CSVs in {args.csv_dir}")
        logger.info("CV mode: Found %d folds in %s", len(train_csvs), args.csv_dir)
    else:
        # Single split mode
        train_csv = args.csv_dir / "train_windows.csv"
        val_csv = args.csv_dir / "val_windows.csv"
        test_csv = args.csv_dir / "test_windows.csv"
        if not train_csv.exists() or not val_csv.exists():
            raise RuntimeError(f"Missing train_windows.csv or val_windows.csv in {args.csv_dir}")
        train_csvs = [train_csv]
        val_csvs = [val_csv]
        logger.info("Single split mode: train=%s | val=%s", train_csv, val_csv)

    # Load all records to compute video label counts
    all_records, _ = load_tal_split(
        train_csvs + val_csvs,
        args.clips_root,
        include_background=include_background,
    )
    video_label_counts = build_video_label_counts(all_records, key_field="video_key")
    logger.info("Total records: %d | Unique videos: %d", len(all_records), len(video_label_counts))

    # Load processor
    base_processor = VJEPA2VideoProcessor.from_pretrained(args.model_id)
    base_frames_per_clip = get_frames_per_clip(base_processor, args.frames_per_clip)
    logger.info("Frames per clip: %d", base_frames_per_clip)

    args.output_root.mkdir(parents=True, exist_ok=True)

    fold_results = []
    fold_pairs = list(zip(train_csvs, val_csvs))
    if args.split_mode == "cv" and args.max_folds is not None:
        fold_pairs = fold_pairs[: args.max_folds]

    for fold_idx, (train_csv, val_csv) in enumerate(fold_pairs):
        if args.split_mode == "cv" and fold_idx < args.start_fold:
            logger.info("Skipping fold %d (start-fold=%d)", fold_idx, args.start_fold)
            continue
        
        logger.info("=" * 90)
        if args.split_mode == "cv":
            logger.info("Fold %d | train=%s | val=%s", fold_idx, train_csv.name, val_csv.name)
            fold_out = args.output_root / f"fold_{fold_idx}"
        else:
            logger.info("Single split | train=%s | val=%s", train_csv.name, val_csv.name)
            fold_out = args.output_root / "single"

        reuse_existing = args.reuse_checkpoints and fold_out.exists()
        fold_out.mkdir(parents=True, exist_ok=True)
        
        if reuse_existing:
            fold_processor = VJEPA2VideoProcessor.from_pretrained(fold_out)
            frames_per_clip_fold = get_frames_per_clip(fold_processor, args.frames_per_clip)
            logger.info(
                "Reusing checkpoint from %s (frames_per_clip=%d)",
                fold_out,
                frames_per_clip_fold,
            )
        else:
            fold_processor = base_processor
            frames_per_clip_fold = base_frames_per_clip

        # Load train/val records
        train_records, miss_train = load_tal_split(
            [train_csv], args.clips_root, include_background=include_background
        )
        val_records, miss_val = load_tal_split(
            [val_csv], args.clips_root, include_background=include_background
        )
        logger.info("Train: %d windows | Val: %d windows", len(train_records), len(val_records))
        logger.info("Missing clips -> train: %d | val: %d", len(miss_train), len(miss_val))
        if miss_train or miss_val:
            example_missing = (miss_train + miss_val)[:5]
            logger.warning("Example missing clips: %s", example_missing)

        # Log class distribution
        train_label_counts = Counter(rec["label_id"] for rec in train_records)
        logger.info("Train class distribution: %s", {id2label[k]: v for k, v in sorted(train_label_counts.items())})

        # Create datasets
        train_ds = TALDataset(
            train_records,
            id2label,
            frames_per_clip_fold,
            video_label_counts,
            crop_cfg=crop_cfg,
            sam3_rows=sam3_rows,
            video_meta=video_meta,
        )
        val_ds = TALDataset(
            val_records,
            id2label,
            frames_per_clip_fold,
            video_label_counts,
            crop_cfg=crop_cfg,
            sam3_rows=sam3_rows,
            video_meta=video_meta,
        )

        # Create dataloaders
        collate = partial(collate_fn, processor=fold_processor)
        train_loader = None
        if not reuse_existing:
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
                prefetch_factor=2 if args.num_workers > 0 else None,
            )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        # Compute class weights for loss balancing
        cls_counts = Counter(rec["label_id"] for rec in train_records)
        weights = []
        total_train = sum(cls_counts.values())
        for i in range(num_classes):
            count = max(cls_counts.get(i, 0), 1)  # avoid divide-by-zero
            weights.append(total_train / (num_classes * count))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        logger.info("Class weights: %s", {id2label[i]: f"{w:.3f}" for i, w in enumerate(weights)})

        # Initialize model
        if reuse_existing:
            model = VJEPA2ForVideoClassification.from_pretrained(fold_out).to(device)
        else:
            model = VJEPA2ForVideoClassification.from_pretrained(
                args.model_id,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True,
            ).to(device)
            # Freeze backbone
            for param in model.vjepa2.parameters():
                param.requires_grad = False

        if not reuse_existing:
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

        # W&B logging
        crop_tag = "-crop" if crop_cfg.enabled else ""
        bg_tag = "" if include_background else "-no_bg"
        if args.split_mode == "cv":
            run_name = f"{args.run_prefix}-fold{fold_idx}-{frames_per_clip_fold}fr{crop_tag}{bg_tag}"
        else:
            run_name = f"{args.run_prefix}-single-{frames_per_clip_fold}fr{crop_tag}{bg_tag}"
        
        wb_logger = WandbAdapter(
            mode=args.wandb_mode,
            project=args.wandb_project,
            run_name=run_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "frames_per_clip": frames_per_clip_fold,
                "fold": fold_idx if args.split_mode == "cv" else -1,
                "accumulation_steps": args.accumulation_steps,
                "num_classes": num_classes,
                "include_background": include_background,
                "split_mode": args.split_mode,
            },
        )

        # Training loop
        if reuse_existing:
            logger.info("Skipping training (reuse-checkpoints).")
        else:
            for epoch in range(1, args.num_epochs + 1):
                model.train()
                optimizer.zero_grad()
                running_loss = 0.0
                num_batches = 0

                for step, (inputs, labels, metas) in enumerate(train_loader, start=1):
                    if inputs is None or labels is None:
                        continue
                    labels = labels.to(device)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    loss = F.cross_entropy(outputs.logits, labels, weight=class_weights) / args.accumulation_steps
                    loss.backward()
                    running_loss += loss.item()
                    num_batches += 1

                    if step % args.accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    if step % args.log_interval == 0:
                        avg_loss = running_loss / max(num_batches, 1) * args.accumulation_steps
                        logger.info("  Epoch %d Step %d/%d | Loss: %.4f", epoch, step, len(train_loader), avg_loss)

                    wb_logger.log({"train/loss": loss.item(), "epoch": epoch, "step": step})

                if num_batches and num_batches % args.accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()

                val_acc, _, _, _, _ = evaluate(val_loader, model, device)
                wb_logger.log({"val/acc": val_acc, "epoch": epoch})
                logger.info("  Epoch %d complete | val_acc=%.3f", epoch, val_acc)

            # Save checkpoint
            model.save_pretrained(fold_out)
            fold_processor.save_pretrained(fold_out)
            logger.info("Saved model + processor to %s", fold_out)

        # Full evaluation with top-k and metrics
        topk_metrics = evaluate_topk(val_loader, model, device, k=args.topk)
        labels_arr = np.array(topk_metrics["labels"])
        preds_arr = np.array(topk_metrics["preds_top1"])
        probs_arr = (
            topk_metrics["probs"].cpu().numpy() if isinstance(topk_metrics["probs"], torch.Tensor) else None
        )
        metas_df = aggregate_preds(labels_arr, preds_arr, probs_arr, topk_metrics["metas"], id2label)
        
        # Save predictions
        window_preds_path = fold_out / "window_level_preds.csv"
        metas_df.to_csv(window_preds_path, index=False)
        logger.info("Saved window-level predictions to %s", window_preds_path)

        summary_metrics, per_class_df, pr_curves = compute_classification_metrics(
            labels_arr, preds_arr, probs_arr, id2label
        )

        # Save confusion matrix
        conf_path = fold_out / "confusion_matrix_window.png"
        save_confusion_matrix(
            labels_arr,
            preds_arr,
            list(id2label.keys()),
            list(id2label.values()),
            conf_path,
            normalize="true",
        )

        # Video-level aggregation
        video_metrics = aggregate_video_predictions(metas_df, labels_arr, preds_arr, probs_arr, id2label)
        video_summary = None
        video_per_class_df = None
        video_acc = None
        video_kappa = None
        if video_metrics:
            video_labels = video_metrics["labels"]
            video_preds = video_metrics["preds"]
            video_probs = video_metrics["probs"]
            video_summary, video_per_class_df, _ = compute_classification_metrics(
                video_labels, video_preds, video_probs, id2label
            )
            video_acc = float((video_labels == video_preds).mean()) if len(video_labels) else 0.0
            video_kappa = float(cohen_kappa_score(video_labels, video_preds)) if len(video_labels) else float("nan")

            video_conf_path = fold_out / "confusion_matrix_video.png"
            save_confusion_matrix(
                video_labels,
                video_preds,
                list(id2label.keys()),
                list(id2label.values()),
                video_conf_path,
                normalize="true",
            )

            video_pred_path = fold_out / "video_level_preds.csv"
            video_metrics["df"].to_csv(video_pred_path, index=False)
            logger.info("Saved video-level predictions to %s", video_pred_path)

        # Persist per-class tables
        per_class_path = fold_out / "per_class_window.csv"
        per_class_df.to_csv(per_class_path, index=False)
        if video_per_class_df is not None:
            video_per_class_path = fold_out / "per_class_video.csv"
            video_per_class_df.to_csv(video_per_class_path, index=False)

        # Subset metrics
        masks = {
            "is_background": metas_df["is_background"] == 1,
            "is_rmm": metas_df["is_background"] == 0,
            "mixed_video": metas_df["mixed_video"] == True,  # noqa: E712
        }
        if "mask_severity" in metas_df.columns:
            for sev in metas_df["mask_severity"].dropna().unique():
                masks[f"mask_{sev}"] = metas_df["mask_severity"] == sev
        if "pose_severity" in metas_df.columns:
            for sev in metas_df["pose_severity"].dropna().unique():
                masks[f"pose_{sev}"] = metas_df["pose_severity"] == sev

        subset_metrics: Dict[str, Dict] = {}
        for name, mask in masks.items():
            res = metrics_for_subset(mask.values, labels_arr, preds_arr, probs_arr, id2label)
            if res:
                subset_metrics[name] = res

        fold_results.append({
            "fold": fold_idx if args.split_mode == "cv" else -1,
            "train_csv": train_csv.name,
            "val_csv": val_csv.name,
            "top1_acc": topk_metrics["top1_acc"],
            "topk_acc": topk_metrics[f"top{args.topk}_acc"],
            "improvement": topk_metrics["improvement"],
            **summary_metrics,
            "per_class": per_class_df,
            "pr_curves": pr_curves,
            "subset_metrics": subset_metrics,
            "video_acc": video_acc,
            "video_macro_f1": video_summary["macro_f1"] if video_summary else None,
            "video_kappa": video_kappa,
            "video_per_class": video_per_class_df,
            "metas": metas_df,
        })

        # Log to W&B
        wb_logger.log({
            "eval/top1_acc": topk_metrics["top1_acc"],
            f"eval/top{args.topk}_acc": topk_metrics[f"top{args.topk}_acc"],
            "eval/topk_improvement": topk_metrics["improvement"],
            "eval/macro_f1": summary_metrics["macro_f1"],
            "eval/micro_f1": summary_metrics["micro_f1"],
            "eval/weighted_f1": summary_metrics["weighted_f1"],
        })
        wb_logger.log_table("eval/per_class", per_class_df)
        if video_summary and video_per_class_df is not None:
            wb_logger.log({
                "eval/video_acc": video_acc,
                "eval/video_macro_f1": video_summary["macro_f1"],
                "eval/video_kappa": video_kappa,
            })
            wb_logger.log_table("eval/per_class_video", video_per_class_df)
        if subset_metrics:
            subset_rows = []
            for name, metrics in subset_metrics.items():
                row = {"subset": name}
                row.update(metrics)
                subset_rows.append(row)
            wb_logger.log_table("eval/subsets", pd.DataFrame(subset_rows))
        wb_logger.finish()

        logger.info(
            "Results | top1=%.3f | top%d=%.3f | macro F1=%.3f",
            topk_metrics["top1_acc"],
            args.topk,
            topk_metrics[f"top{args.topk}_acc"],
            summary_metrics["macro_f1"],
        )
        if video_summary:
            logger.info(
                "Video-level | acc=%.3f | macro F1=%.3f | kappa=%.3f",
                video_acc,
                video_summary["macro_f1"],
                video_kappa,
            )

    # Aggregate cross-fold results (CV mode only)
    if args.split_mode == "cv" and len(fold_results) > 1:
        summary_rows = [
            {
                "fold": fr["fold"],
                "top1_acc": fr["top1_acc"],
                f"top{args.topk}_acc": fr["topk_acc"],
                "macro_f1": fr["macro_f1"],
                "micro_f1": fr["micro_f1"],
                "weighted_f1": fr["weighted_f1"],
                "video_acc": fr.get("video_acc"),
                "video_macro_f1": fr.get("video_macro_f1"),
                "video_kappa": fr.get("video_kappa"),
            }
            for fr in fold_results
        ]
        summary_df = pd.DataFrame(summary_rows)
        logger.info("Cross-fold summary:\n%s", summary_df)
        logger.info("Averages:\n%s", summary_df.mean(numeric_only=True))

        summary_path = args.output_root / "cv_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved cross-fold summary to %s", summary_path)

        json_path = args.output_root / "cv_summary.json"
        json_path.write_text(json.dumps(summary_rows, indent=2))
        logger.info("Saved cross-fold summary JSON to %s", json_path)


if __name__ == "__main__":
    main()

