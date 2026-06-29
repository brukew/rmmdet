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
import random
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
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional
    wandb = None

logger = logging.getLogger(__name__)

# Resolve filesystem locations from the repo's single source of truth (config.yaml).
# Walk up from this file until paths.py (sitting at the repo root) is found, so the
# import works regardless of the current working directory.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import PATHS  # noqa: E402

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

# Binary classification (RMM vs Background)
BINARY_LABEL_MAP = {
    0: "rmm",
    1: "background",
}
BINARY_ID2LABEL = BINARY_LABEL_MAP.copy()
BINARY_LABEL2ID = {v: k for k, v in BINARY_LABEL_MAP.items()}

DEFAULT_TAL_CSV_DIR = Path("/orcd/data/satra/001/users/brukew/actreg/dataprep/tal/splits_cv_4class")
DEFAULT_TAL_CLIPS_ROOT = PATHS.tal_clips_root

# Cropping defaults (same as original)
DEFAULT_PARSED_CSV = Path("/orcd/data/satra/001/users/brukew/actreg/dataprep/rmm_sam3_parsed.csv")
DEFAULT_MASK_CACHE_BASE = PATHS.cache_for_tracking
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
    binary_mode: bool = False,
) -> Tuple[List[Dict], List[Tuple[str, Path]]]:
    """
    Load TAL window records from CSV files.
    
    Args:
        csv_paths: List of TAL window CSV files to load.
        clips_root: Root directory containing window clips (flat structure).
        include_background: Whether to include background windows (primary_label=-1).
        exclude_windows: Optional set of window_ids to exclude.
        binary_mode: If True, use 2-class labels (0=RMM, 1=background) instead of 5-class.
    
    Returns:
        Tuple of (records list, missing clips list).
    """
    records: List[Dict] = []
    missing: List[Tuple[str, Path]] = []
    excluded_count = 0
    background_excluded = 0
    
    # Select label map based on mode
    if binary_mode:
        id2label = BINARY_ID2LABEL
    else:
        id2label = TAL_ID2LABEL
    
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            window_id = row.get("window_id")
            primary_label = row.get("primary_label")
            
            if pd.isna(window_id) or pd.isna(primary_label):
                continue
            
            primary_label = int(primary_label)
            
            # Skip background if not included (only relevant for non-binary mode)
            if primary_label == -1 and not include_background and not binary_mode:
                background_excluded += 1
                continue
            
            # Skip excluded windows
            if exclude_windows and str(window_id) in exclude_windows:
                excluded_count += 1
                continue
            
            # Map primary_label to class index
            if binary_mode:
                # Binary mode: 0=RMM (any of 0-3), 1=background (-1)
                if primary_label == -1:
                    label_id = 1  # background
                else:
                    label_id = 0  # RMM (any type)
            else:
                # 5-class mode: 0-3 for RMM types, 4 for background
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
                "label_name": id2label[label_id],
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


def subsample_background(
    records: List[Dict],
    bg_multiplier: float,
    bg_label_id: int = 4,
    seed: int = 42,
) -> List[Dict]:
    """
    (Deprecated: use balance_classes instead)
    Subsample background windows to reduce class imbalance.
    """
    rmm_records = [r for r in records if r["label_id"] != bg_label_id]
    bg_records = [r for r in records if r["label_id"] == bg_label_id]
    
    n_rmm = len(rmm_records)
    n_bg_target = int(n_rmm * bg_multiplier)
    
    if len(bg_records) <= n_bg_target:
        logger.info(
            "Background subsampling: keeping all %d (target was %d)", 
            len(bg_records), n_bg_target
        )
        return records
    
    rng = random.Random(seed)
    sampled_bg = rng.sample(bg_records, n_bg_target)
    
    logger.info(
        "Background subsampling: %d -> %d (%.1fx RMM count of %d)",
        len(bg_records), n_bg_target, bg_multiplier, n_rmm
    )
    
    return rmm_records + sampled_bg


def balance_classes(
    records: List[Dict],
    class_probs: List[float],
    num_classes: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """
    Balance classes by sampling with specified probabilities.
    
    Args:
        records: All loaded records.
        class_probs: List of probabilities for each class.
                     - prob > 1.0: Upsample (duplicate samples to achieve this multiplier)
                     - prob < 1.0: Downsample (keep this fraction of samples)
                     - prob = 1.0: Keep all samples
        num_classes: Number of classes (5 for TAL: 4 RMM + background).
        seed: Random seed for reproducibility.
    
    Returns:
        Records with balanced classes.
    
    Example:
        class_probs = [1.0, 1.0, 1.93, 9.12, 0.1]
        - Classes 0, 1: Keep all
        - Class 2: Upsample ~2x
        - Class 3: Upsample ~9x
        - Class 4 (background): Keep only 10%
    """
    if len(class_probs) != num_classes:
        raise ValueError(f"class_probs length ({len(class_probs)}) != num_classes ({num_classes})")
    
    rng = random.Random(seed)
    
    # Group records by class
    class_records: Dict[int, List[Dict]] = {i: [] for i in range(num_classes)}
    for rec in records:
        label_id = rec["label_id"]
        if 0 <= label_id < num_classes:
            class_records[label_id].append(rec)
    
    # Log original distribution
    orig_counts = {i: len(class_records[i]) for i in range(num_classes)}
    logger.info("Original class distribution: %s", orig_counts)
    
    balanced_records: List[Dict] = []
    
    for class_id in range(num_classes):
        prob = class_probs[class_id]
        recs = class_records[class_id]
        n_orig = len(recs)
        
        if n_orig == 0:
            continue
        
        if prob == 1.0:
            # Keep all
            balanced_records.extend(recs)
        elif prob < 1.0:
            # Downsample: keep fraction of samples
            n_keep = max(1, int(n_orig * prob))
            sampled = rng.sample(recs, min(n_keep, n_orig))
            balanced_records.extend(sampled)
            logger.info("  Class %d: downsampled %d -> %d (prob=%.2f)", class_id, n_orig, len(sampled), prob)
        else:
            # Upsample: duplicate samples to achieve multiplier
            n_target = int(n_orig * prob)
            # Start with all original samples
            upsampled = recs.copy()
            # Add duplicates until we reach target
            while len(upsampled) < n_target:
                remaining = n_target - len(upsampled)
                upsampled.extend(rng.choices(recs, k=min(remaining, n_orig)))
            balanced_records.extend(upsampled)
            logger.info("  Class %d: upsampled %d -> %d (prob=%.2f)", class_id, n_orig, len(upsampled), prob)
    
    # Log final distribution
    final_counts: Dict[int, int] = {}
    for rec in balanced_records:
        label_id = rec["label_id"]
        final_counts[label_id] = final_counts.get(label_id, 0) + 1
    logger.info("Balanced class distribution: %s", final_counts)
    logger.info("Total records: %d -> %d", len(records), len(balanced_records))
    
    # Shuffle to mix classes
    rng.shuffle(balanced_records)
    
    return balanced_records


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


def evaluate_with_loss(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """
    Evaluate model and compute both accuracy and loss.
    
    Args:
        loader: DataLoader for validation set.
        model: Model to evaluate.
        device: Device to run on.
        class_weights: Optional class weights for loss computation.
    
    Returns:
        Tuple of (accuracy, average_loss).
    """
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    num_batches = 0
    total_batches = len(loader)
    
    # #region agent log
    import json as _json; _log_path = "/orcd/data/satra/001/users/brukew/.cursor/debug.log"
    with open(_log_path, "a") as _f: _f.write(_json.dumps({"hypothesisId": "A", "location": "evaluate_with_loss:entry", "message": "Starting validation", "data": {"total_batches": total_batches}, "timestamp": int(__import__('time').time()*1000)}) + "\n")
    # #endregion
    
    logger.info("  Starting validation (%d batches)...", total_batches)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels, metas) in enumerate(loader):
            # #region agent log
            if batch_idx % 500 == 0:
                with open(_log_path, "a") as _f: _f.write(_json.dumps({"hypothesisId": "B", "location": "evaluate_with_loss:loop", "message": "Val batch progress", "data": {"batch_idx": batch_idx, "total": total_batches}, "timestamp": int(__import__('time').time()*1000)}) + "\n")
                logger.info("    Val progress: %d/%d batches", batch_idx, total_batches)
            # #endregion
            
            if inputs is None or labels is None:
                continue
            labels = labels.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            
            # Compute loss
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            total_loss += loss.item()
            num_batches += 1
            
            # Compute accuracy
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = correct / max(total, 1)
    avg_loss = total_loss / max(num_batches, 1)
    
    # #region agent log
    with open(_log_path, "a") as _f: _f.write(_json.dumps({"hypothesisId": "A", "location": "evaluate_with_loss:exit", "message": "Validation complete", "data": {"acc": acc, "avg_loss": avg_loss, "total_samples": total}, "timestamp": int(__import__('time').time()*1000)}) + "\n")
    # #endregion
    
    return acc, avg_loss


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


def compute_classification_metrics(labels, preds, probs, id2label: Dict[int, str], task: str = "tal"):
    """
    Compute comprehensive classification metrics.
    
    Args:
        labels: Ground truth labels.
        preds: Predicted labels.
        probs: Prediction probabilities (N x num_classes).
        id2label: Label ID to name mapping.
        task: Task type - 'tal' for 5-class TAL (includes RMM vs BG metrics),
              'rmm' for 4-class RMM classification.
    
    Returns:
        Tuple of (summary_dict, per_class_df, pr_curves_dict).
    """
    labels = np.array(labels)
    preds = np.array(preds)
    
    class_names = [id2label[i] for i in range(len(id2label))]
    label_ids = list(range(len(id2label)))
    num_classes = len(id2label)
    
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
    pr_curves = {}
    if probs is not None and len(class_names) > 0:
        supports = per_class.set_index("class")["support"]
        top_classes = supports.sort_values(ascending=False).head(min(5, len(class_names))).index.tolist()
        for cls in top_classes:
            cls_id = class_names.index(cls)
            y_bin = (labels == cls_id).astype(int)
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
        "cohens_kappa": cohen_kappa_score(labels, preds),
    }
    
    # Per-class precision and recall (skip background for TAL)
    for i, name in enumerate(class_names):
        if task == "tal" and "background" in name.lower():
            continue
        # Create sanitized key name
        key_name = name.replace(" ", "_").replace("-", "_")
        summary[f"prec_{key_name}"] = prec_rec_f1[0][i]
        summary[f"rec_{key_name}"] = prec_rec_f1[1][i]
        summary[f"f1_{key_name}"] = prec_rec_f1[2][i]
    
    # ========== RMM vs Background Binary Metrics ==========
    # For 5-class TAL: compute derived binary metrics from multi-class predictions
    # For 2-class binary mode: these are the primary metrics (computed directly from predictions)
    if task == "tal":
        if num_classes == 5:
            # 5-class TAL: Derive binary metrics from multi-class predictions
            BACKGROUND_CLASS = 4
            RMM_CLASS = None  # Not used in 5-class mode
            
            # Binary labels: 1 = RMM (any of classes 0-3), 0 = Background (class 4)
            y_true_binary = (labels != BACKGROUND_CLASS).astype(int)
            y_pred_binary = (preds != BACKGROUND_CLASS).astype(int)
            
            # Binary scores: sum of RMM class probabilities
            if probs is not None and probs.shape[1] > BACKGROUND_CLASS:
                rmm_scores = probs[:, :BACKGROUND_CLASS].sum(axis=1)  # Sum classes 0-3
            else:
                rmm_scores = None
        elif num_classes == 2:
            # Binary mode: class 0 = RMM, class 1 = background
            RMM_CLASS = 0
            BACKGROUND_CLASS = 1
            
            # Binary labels: 1 = RMM (class 0), 0 = Background (class 1)
            y_true_binary = (labels == RMM_CLASS).astype(int)
            y_pred_binary = (preds == RMM_CLASS).astype(int)
            
            # Binary scores: probability of RMM class
            if probs is not None and probs.shape[1] >= 2:
                rmm_scores = probs[:, RMM_CLASS]
            else:
                rmm_scores = None
        else:
            # 4-class RMM only - no binary metrics needed
            y_true_binary = None
            y_pred_binary = None
            rmm_scores = None
        
        if y_true_binary is not None:
            # Binary classification metrics
            summary["rmm_vs_bg_accuracy"] = accuracy_score(y_true_binary, y_pred_binary)
            summary["rmm_vs_bg_f1"] = f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
            summary["rmm_vs_bg_precision"] = precision_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
            summary["rmm_vs_bg_recall"] = recall_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
            
            # AUC-ROC for RMM detection
            if rmm_scores is not None:
                try:
                    if len(np.unique(y_true_binary)) > 1:
                        summary["rmm_vs_bg_auc"] = roc_auc_score(y_true_binary, rmm_scores)
                    else:
                        summary["rmm_vs_bg_auc"] = 0.0
                except Exception:
                    summary["rmm_vs_bg_auc"] = 0.0
            else:
                summary["rmm_vs_bg_auc"] = 0.0
            
            # Error rates
            # False alarm: Background predicted as RMM
            bg_mask = y_true_binary == 0
            if bg_mask.sum() > 0:
                summary["bg_false_alarm_rate"] = (y_pred_binary[bg_mask] == 1).mean()
            else:
                summary["bg_false_alarm_rate"] = 0.0
            
            # Miss rate: RMM predicted as Background
            rmm_mask = y_true_binary == 1
            if rmm_mask.sum() > 0:
                summary["rmm_miss_rate"] = (y_pred_binary[rmm_mask] == 0).mean()
            else:
                summary["rmm_miss_rate"] = 0.0
        
        # RMM-only metrics (excluding background) - only for 5-class TAL
        if num_classes == 5:
            rmm_indices = list(range(4))  # Classes 0-3 (RMM types)
            rmm_mask_multiclass = np.isin(labels, rmm_indices)
            if rmm_mask_multiclass.sum() > 0:
                y_true_rmm = labels[rmm_mask_multiclass]
                y_pred_rmm = preds[rmm_mask_multiclass]
                summary["rmm_only_macro_f1"] = f1_score(y_true_rmm, y_pred_rmm, average='macro', zero_division=0)
                summary["rmm_only_macro_precision"] = precision_score(y_true_rmm, y_pred_rmm, average='macro', zero_division=0)
                summary["rmm_only_macro_recall"] = recall_score(y_true_rmm, y_pred_rmm, average='macro', zero_division=0)
            else:
                summary["rmm_only_macro_f1"] = 0.0
                summary["rmm_only_macro_precision"] = 0.0
                summary["rmm_only_macro_recall"] = 0.0
    
    return summary, per_class, pr_curves


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
    
    # Classification mode
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Exclude background windows (train on 4 RMM classes only).",
    )
    parser.add_argument(
        "--binary-classification",
        action="store_true",
        help="Use 2-class binary mode (RMM vs Background) instead of 5-class. "
             "Combines all RMM classes (0-3) into class 0, background becomes class 1.",
    )
    parser.add_argument(
        "--bg-subsample",
        type=float,
        default=None,
        help="(Deprecated: use --class-prob instead) Subsample background windows to this multiple of total RMM count.",
    )
    parser.add_argument(
        "--class-prob",
        type=str,
        default=None,
        help="Class sampling probabilities as JSON list, e.g., '[1.0,1.0,1.93,9.12,0.1]'. "
             "prob > 1.0: upsample (duplicate), prob < 1.0: downsample, prob = 1.0: keep all. "
             "Overrides --bg-subsample if provided.",
    )
    
    # Early stopping
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop training if validation loss doesn't improve for N epochs (0 to disable).",
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
    binary_mode = args.binary_classification
    
    # Build label mappings based on classification mode
    if binary_mode:
        # Binary classification: RMM vs Background
        num_classes = 2
        id2label = BINARY_ID2LABEL.copy()
        label2id = BINARY_LABEL2ID.copy()
        # Binary mode always includes background (it's one of the two classes)
        include_background = True
        logger.info("Binary classification mode: 2 classes (RMM vs Background)")
    elif include_background:
        # 5-class: 4 RMM types + background
        num_classes = 5
        id2label = TAL_ID2LABEL.copy()
        label2id = TAL_LABEL2ID.copy()
    else:
        # 4-class: RMM types only (no background)
        num_classes = 4
        id2label = {i: TAL_ID2LABEL[i] for i in range(4)}
        label2id = {v: k for k, v in id2label.items()}
    
    logger.info("TAL mode: %d classes | background=%s | binary=%s", num_classes, include_background, binary_mode)
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
        binary_mode=binary_mode,
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
            [train_csv], args.clips_root, include_background=include_background,
            binary_mode=binary_mode
        )
        val_records, miss_val = load_tal_split(
            [val_csv], args.clips_root, include_background=include_background,
            binary_mode=binary_mode
        )
        
        # Class balancing (training only)
        if args.class_prob is not None:
            # Parse class_prob JSON string
            import json as json_module
            try:
                class_probs = json_module.loads(args.class_prob)
                if not isinstance(class_probs, list) or len(class_probs) != num_classes:
                    raise ValueError(f"class_prob must be a list of {num_classes} floats (got {len(class_probs)})")
                train_records = balance_classes(
                    train_records,
                    class_probs,
                    num_classes=num_classes,
                    seed=42 + fold_idx,
                )
            except json_module.JSONDecodeError as e:
                logger.error("Failed to parse --class-prob: %s", e)
                raise
        elif args.bg_subsample is not None and include_background and not binary_mode:
            # Legacy: subsample background only (5-class mode)
            train_records = subsample_background(
                train_records,
                args.bg_subsample,
                bg_label_id=4,
                seed=42 + fold_idx,
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

        # Training loop with early stopping
        if reuse_existing:
            logger.info("Skipping training (reuse-checkpoints).")
        else:
            # Early stopping tracking
            best_val_loss = float('inf')
            epochs_without_improvement = 0
            best_epoch = 0
            early_stopping_enabled = args.early_stopping_patience > 0
            
            if early_stopping_enabled:
                logger.info("Early stopping enabled: patience=%d epochs (monitoring val_loss)", args.early_stopping_patience)
            
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

                # Validation with loss for early stopping
                val_acc, val_loss = evaluate_with_loss(val_loader, model, device, class_weights)
                wb_logger.log({"val/acc": val_acc, "val/loss": val_loss, "epoch": epoch})
                logger.info("  Epoch %d complete | val_acc=%.3f | val_loss=%.4f", epoch, val_acc, val_loss)

                # Early stopping check
                if early_stopping_enabled:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        epochs_without_improvement = 0
                        # Save best checkpoint
                        model.save_pretrained(fold_out)
                        fold_processor.save_pretrained(fold_out)
                        logger.info("  New best model saved (val_loss=%.4f)", val_loss)
                    else:
                        epochs_without_improvement += 1
                        logger.info("  No improvement for %d epoch(s) (best=%.4f at epoch %d)",
                                   epochs_without_improvement, best_val_loss, best_epoch)
                        if epochs_without_improvement >= args.early_stopping_patience:
                            logger.info("Early stopping triggered after %d epochs", epoch)
                            break

            # Save final checkpoint (if early stopping not enabled or didn't trigger)
            if not early_stopping_enabled:
                model.save_pretrained(fold_out)
                fold_processor.save_pretrained(fold_out)
                logger.info("Saved model + processor to %s", fold_out)
            elif epochs_without_improvement < args.early_stopping_patience:
                # Training completed without early stopping - save final model
                model.save_pretrained(fold_out)
                fold_processor.save_pretrained(fold_out)
                logger.info("Saved final model + processor to %s", fold_out)

        # Full evaluation with top-k and metrics
        topk_metrics = evaluate_topk(val_loader, model, device, k=args.topk)
        labels_arr = np.array(topk_metrics["labels"])
        preds_arr = np.array(topk_metrics["preds_top1"])
        probs_arr = (
            topk_metrics["probs"].cpu().numpy() if isinstance(topk_metrics["probs"], torch.Tensor) else None
        )
        metas_df = aggregate_preds(labels_arr, preds_arr, probs_arr, topk_metrics["metas"], id2label)
        
        # Save predictions (clip-level)
        clip_preds_path = fold_out / "predictions_clip.csv"
        metas_df.to_csv(clip_preds_path, index=False)
        logger.info("Saved clip-level predictions to %s", clip_preds_path)
        
        # Determine task type
        task_type = "tal" if include_background else "rmm"

        summary_metrics, per_class_df, pr_curves = compute_classification_metrics(
            labels_arr, preds_arr, probs_arr, id2label, task=task_type
        )

        # Save confusion matrix
        conf_path = fold_out / "confusion_matrix_clip.png"
        save_confusion_matrix(
            labels_arr,
            preds_arr,
            list(id2label.keys()),
            list(id2label.values()),
            conf_path,
            normalize="true",
        )

        # Persist per-class table
        per_class_path = fold_out / "per_class_clip.csv"
        per_class_df.to_csv(per_class_path, index=False)

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

        # Build fold results with clip_ prefix and percentages (like pyskl)
        fold_result = {
            "fold": fold_idx if args.split_mode == "cv" else -1,
            "train_csv": train_csv.name,
            "val_csv": val_csv.name,
            "split": "val",
            "task": task_type,
            "clip_top1_acc": topk_metrics["top1_acc"] * 100,
            "clip_top2_acc": topk_metrics[f"top{args.topk}_acc"] * 100,
            "clip_macro_f1": summary_metrics["macro_f1"] * 100,
            "clip_weighted_f1": summary_metrics["weighted_f1"] * 100,
            "clip_macro_precision": summary_metrics["macro_precision"] * 100,
            "clip_macro_recall": summary_metrics["macro_recall"] * 100,
            "clip_cohens_kappa": summary_metrics["cohens_kappa"],
        }
        
        # Add TAL-specific RMM vs BG metrics
        if task_type == "tal":
            fold_result.update({
                "clip_rmm_vs_bg_accuracy": summary_metrics.get("rmm_vs_bg_accuracy", 0) * 100,
                "clip_rmm_vs_bg_f1": summary_metrics.get("rmm_vs_bg_f1", 0) * 100,
                "clip_rmm_vs_bg_precision": summary_metrics.get("rmm_vs_bg_precision", 0) * 100,
                "clip_rmm_vs_bg_recall": summary_metrics.get("rmm_vs_bg_recall", 0) * 100,
                "clip_rmm_vs_bg_auc": summary_metrics.get("rmm_vs_bg_auc", 0) * 100,
                "clip_bg_false_alarm_rate": summary_metrics.get("bg_false_alarm_rate", 0) * 100,
                "clip_rmm_miss_rate": summary_metrics.get("rmm_miss_rate", 0) * 100,
                "clip_rmm_only_macro_f1": summary_metrics.get("rmm_only_macro_f1", 0) * 100,
                "clip_rmm_only_macro_precision": summary_metrics.get("rmm_only_macro_precision", 0) * 100,
                "clip_rmm_only_macro_recall": summary_metrics.get("rmm_only_macro_recall", 0) * 100,
            })
        
        # Add per-class metrics
        for key, val in summary_metrics.items():
            if key.startswith(("prec_", "rec_", "f1_")):
                fold_result[f"clip_{key}"] = val * 100
        
        fold_result["per_class"] = per_class_df
        fold_result["pr_curves"] = pr_curves
        fold_result["subset_metrics"] = subset_metrics
        fold_result["metas"] = metas_df
        
        fold_results.append(fold_result)
        
        # Save metrics.json (pyskl-compatible format)
        metrics_json_path = fold_out / "metrics.json"
        metrics_to_save = {k: v for k, v in fold_result.items() 
                         if k not in ("per_class", "pr_curves", "subset_metrics", "metas")}
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        logger.info("Saved metrics to %s", metrics_json_path)

        # Log to W&B
        wb_log_data = {
            "eval/clip_top1_acc": fold_result["clip_top1_acc"],
            "eval/clip_top2_acc": fold_result["clip_top2_acc"],
            "eval/clip_macro_f1": fold_result["clip_macro_f1"],
            "eval/clip_macro_precision": fold_result["clip_macro_precision"],
            "eval/clip_macro_recall": fold_result["clip_macro_recall"],
            "eval/clip_weighted_f1": fold_result["clip_weighted_f1"],
        }
        if task_type == "tal":
            wb_log_data.update({
                "eval/clip_rmm_vs_bg_f1": fold_result["clip_rmm_vs_bg_f1"],
                "eval/clip_rmm_vs_bg_auc": fold_result["clip_rmm_vs_bg_auc"],
                "eval/clip_rmm_only_macro_f1": fold_result["clip_rmm_only_macro_f1"],
            })
        wb_logger.log(wb_log_data)
        wb_logger.log_table("eval/per_class", per_class_df)
        
        if subset_metrics:
            subset_rows = []
            for name, metrics in subset_metrics.items():
                row = {"subset": name}
                row.update(metrics)
                subset_rows.append(row)
            wb_logger.log_table("eval/subsets", pd.DataFrame(subset_rows))
        wb_logger.finish()

        # Log summary
        logger.info(
            "Results | top1=%.2f%% | top%d=%.2f%% | macro_f1=%.2f%%",
            fold_result["clip_top1_acc"],
            args.topk,
            fold_result["clip_top2_acc"],
            fold_result["clip_macro_f1"],
        )
        if task_type == "tal":
            logger.info(
                "RMM vs BG | f1=%.2f%% | auc=%.2f%% | rmm_only_f1=%.2f%%",
                fold_result["clip_rmm_vs_bg_f1"],
                fold_result["clip_rmm_vs_bg_auc"],
                fold_result["clip_rmm_only_macro_f1"],
            )

    # Aggregate cross-fold results (CV mode only)
    if args.split_mode == "cv" and len(fold_results) > 1:
        # Extract numeric metrics for summary
        metric_keys = [k for k in fold_results[0].keys() 
                      if k not in ("per_class", "pr_curves", "subset_metrics", "metas", 
                                   "train_csv", "val_csv", "split", "task")]
        
        summary_rows = []
        for fr in fold_results:
            row = {k: fr.get(k) for k in metric_keys if k in fr}
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        logger.info("Cross-fold summary:\n%s", summary_df)
        logger.info("Averages:\n%s", summary_df.mean(numeric_only=True))

        summary_path = args.output_root / "cv_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved cross-fold summary to %s", summary_path)

        # Create comprehensive JSON summary with mean/std
        json_summary = {
            "model": "V-JEPA2",
            "task": "tal" if include_background else "rmm",
            "per_fold": summary_rows,
        }
        
        # Compute mean/std for numeric metrics
        numeric_keys = [k for k in summary_rows[0].keys() if k != "fold" and isinstance(summary_rows[0].get(k), (int, float))]
        for key in numeric_keys:
            vals = [fr[key] for fr in summary_rows if key in fr and fr[key] is not None]
            if vals:
                json_summary[f"{key}_mean"] = float(np.mean(vals))
                json_summary[f"{key}_std"] = float(np.std(vals))
        
        json_path = args.output_root / "cv_summary.json"
        json_path.write_text(json.dumps(json_summary, indent=2))
        logger.info("Saved cross-fold summary JSON to %s", json_path)
        
        # Print key metrics
        logger.info("=" * 60)
        logger.info("CV Summary:")
        logger.info("  Clip Top-1: %.2f ± %.2f%%", 
                   json_summary.get("clip_top1_acc_mean", 0), 
                   json_summary.get("clip_top1_acc_std", 0))
        logger.info("  Macro-F1: %.2f ± %.2f%%", 
                   json_summary.get("clip_macro_f1_mean", 0), 
                   json_summary.get("clip_macro_f1_std", 0))
        if include_background:
            logger.info("  RMM vs BG F1: %.2f ± %.2f%%", 
                       json_summary.get("clip_rmm_vs_bg_f1_mean", 0), 
                       json_summary.get("clip_rmm_vs_bg_f1_std", 0))
            logger.info("  RMM-Only Macro-F1: %.2f ± %.2f%%", 
                       json_summary.get("clip_rmm_only_macro_f1_mean", 0), 
                       json_summary.get("clip_rmm_only_macro_f1_std", 0))


if __name__ == "__main__":
    main()

