#!/usr/bin/env python3
"""
Extract V-JEPA2 features from videos for OpenTAD/ActionFormer.

This script extracts features from video files using a fine-tuned V-JEPA2 model
and saves them as NPY files compatible with OpenTAD's feature loading.

Output format:
- One NPY file per video with shape (T, 1024) where T = number of 16-frame snippets
- Files named as {childid}_{filename_stem}.npy to match OpenTAD annotations

Usage:
    python extract_vjepa_features.py \
        --model-source runs/vjepa2_tal_cv_5class_balanced/fold_0 \
        --fold 0 \
        --output-dir /path/to/features
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# SAILS paths
BASE_DIR = Path("/orcd/data/satra/001/users/brukew")
DATAPREP_DIR = BASE_DIR / "actreg/dataprep"
VIDEOS_ROOT = Path("/orcd/scratch/bcs/001/sensein/sails/rmm/videos")
OUTPUT_ROOT = Path("/orcd/scratch/bcs/001/sensein/sails/rmm/features")

# Feature extraction settings
SNIPPET_SIZE = 16  # frames per snippet
SNIPPET_STRIDE = 16  # non-overlapping snippets
FEATURE_DIM = 1024


# =============================================================================
# Video index loading
# =============================================================================

def load_video_index(fold: int, dataprep_dir: Path = DATAPREP_DIR) -> pd.DataFrame:
    """
    Load the video-to-split mapping for a given fold.
    
    Uses the window splits CSVs which contain actual video_path with full absolute paths.
    Loads BOTH train and val videos since ActionFormer needs features for all videos.
    Returns DataFrame with columns: video_key, child_id, filename_stem, video_path, video_file, fps, duration_sec
    """
    splits_dir = dataprep_dir / "tal" / "splits_cv_4class"
    
    # Load both train and val windows CSVs
    train_csv = splits_dir / f"fold_{fold}_train_windows.csv"
    val_csv = splits_dir / f"fold_{fold}_val_windows.csv"
    
    dfs = []
    for csv_path in [train_csv, val_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        dfs.append(pd.read_csv(csv_path))
    
    # Concatenate train and val
    df = pd.concat(dfs, ignore_index=True)
    
    # Get unique videos (windows are per-snippet, we need unique videos)
    # Group by video_key and take first row for each (they share video_path, fps, etc.)
    video_df = df.groupby("video_key").first().reset_index()
    
    # Keep only needed columns and rename for clarity
    video_df = video_df[["video_key", "child_id", "filename", "video_path", "video_file", "fps", "duration_sec"]].copy()
    
    # Add filename_stem column
    video_df["filename_stem"] = video_df["filename"].apply(lambda x: Path(x).stem)
    
    logger.info(f"Loaded {len(video_df)} videos (train + val) for fold {fold}")
    
    return video_df


def normalize_video_key(video_file: str) -> str:
    """Normalize video_file path for consistent matching."""
    normalized = video_file.replace("\\", "/").lstrip("/")
    prefixes = [
        "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/",
        "Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/",
    ]
    for prefix in prefixes:
        if normalized.lower().startswith(prefix.lower()):
            normalized = normalized[len(prefix):]
    return normalized


# =============================================================================
# SAM3-based cropping (matching finetune_sails_vjepa2_tal.py implementation)
# =============================================================================

try:
    import h5py
except ImportError:
    h5py = None

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
    fallback: str = "full"  # "full" or "skip"


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


def load_mask_cache(
    base: Path,
    video_basename: str,
    prompt: str = DEFAULT_MASK_PROMPT,
    model_name: str = DEFAULT_MASK_MODEL,
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
    candidates = [base_dir / f"{model_name}__prompt-{prompt_slug}.h5"]
    candidates.extend(sorted(base_dir.glob("*.h5")))

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
        except Exception:
            continue
    return None


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


def get_video_rotation(video_path: Path) -> int:
    """Get rotation metadata from video file using ffprobe."""
    import subprocess
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream_side_data=rotation",
        "-of", "csv=p=0", str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.stdout.strip():
            return int(float(result.stdout.strip()))
    except Exception:
        pass
    return 0


class CropHelper:
    """Helper class for SAM3-based cropping during feature extraction."""
    
    def __init__(self, crop_cfg: CropConfig):
        self.cfg = crop_cfg
        self.sam3_rows = load_parsed_sam3_csv(crop_cfg.sam3_parsed_csv) if crop_cfg.enabled else {}
        self._mask_cache_store: Dict[str, Optional[Dict]] = {}
        logger.info("Loaded SAM3 parsed CSV with %d entries", len(self.sam3_rows))
    
    def apply_crop(
        self,
        frames: np.ndarray,
        video_file: str,
        frame_indices: np.ndarray,
        fps: float,
        video_path: Path,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply SAM3-based child cropping to frames.
        
        Args:
            frames: Video frames (T, H, W, C).
            video_file: Video file path (for SAM3 lookup).
            frame_indices: Frame indices used.
            fps: Video FPS.
            video_path: Full path to video file.
        
        Returns:
            Tuple of (cropped_frames, metadata_dict).
        """
        cfg = self.cfg
        meta = {
            "crop_applied": False,
            "crop_box": None,
            "crop_reason": "",
        }
        
        if not cfg.enabled:
            return frames, meta
        
        if h5py is None:
            meta["crop_reason"] = "h5py_missing"
            return frames, meta
        
        if fps is None or fps <= 0:
            meta["crop_reason"] = "fps_missing"
            return frames, meta
        
        # Get filename for SAM3 lookup
        fname = Path(video_file).name
        sam3_row = self.sam3_rows.get(fname)
        if not sam3_row:
            meta["crop_reason"] = "no_sam3_row"
            return frames, meta
        
        # Load mask cache
        basename = f"{Path(fname).stem}_segmented"
        if basename not in self._mask_cache_store:
            self._mask_cache_store[basename] = load_mask_cache(
                cfg.mask_cache_base, basename,
                prompt=cfg.mask_prompt, model_name=cfg.mask_model,
            )
        cache = self._mask_cache_store.get(basename)
        if not cache:
            meta["crop_reason"] = "no_cache"
            return frames, meta
        
        # Get cache dimensions
        attrs = cache.get("attrs") or {}
        base_w = attrs.get("width", frames.shape[2])
        base_h = attrs.get("height", frames.shape[1])
        
        # Get rotation
        rotation = -get_video_rotation(video_path)  # Negative to align with notebook convention
        
        frame_w, frame_h = frames.shape[2], frames.shape[1]
        frame_indices_cache = cache["frame_indices"]
        xs: List[float] = []
        ys: List[float] = []
        
        # Process each frame
        for pos, frame_idx in enumerate(frame_indices):
            time_sec = float(frame_idx) / fps
            child_ids = child_ids_for_time(sam3_row, time_sec)
            if not child_ids:
                continue
            
            # Find closest cache frame
            closest_pos = int(np.argmin(np.abs(frame_indices_cache - frame_idx)))
            cache_frame_idx = int(frame_indices_cache[closest_pos])
            
            boxes = cache["boxes"].get(cache_frame_idx)
            obj_ids = cache["obj_ids"].get(cache_frame_idx)
            if boxes is None or len(boxes) == 0:
                continue
            
            # Apply rotation correction
            boxes_rotated = boxes.copy()
            expected_w, expected_h = base_w, base_h
            if rotation:
                boxes_rotated = rotate_boxes(boxes_rotated, rotation, base_w, base_h)
                if rotation in (-90, 90, -270, 270):
                    expected_w, expected_h = base_h, base_w
            
            # Scale boxes to frame dimensions
            if expected_w and expected_h and (frame_w != expected_w or frame_h != expected_h):
                scale_x = frame_w / expected_w
                scale_y = frame_h / expected_h
                boxes_rotated[:, [0, 2]] *= scale_x
                boxes_rotated[:, [1, 3]] *= scale_y
            
            # Find matching child detections
            frame_matches = [i for i, obj in enumerate(obj_ids) if int(obj) in child_ids]
            for det_idx in frame_matches:
                x1, y1, x2, y2 = boxes_rotated[det_idx]
                xs.extend([x1, x2])
                ys.extend([y1, y2])
        
        if not xs or not ys:
            meta["crop_reason"] = "no_matching_ids"
            return frames, meta
        
        # Compute crop box with padding
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
            "crop_reason": "ok",
        })
        return frames_cropped, meta


# =============================================================================
# Feature extraction dataset
# =============================================================================

class VideoFeatureDataset(Dataset):
    """
    Dataset for extracting features from videos.
    
    Yields video snippets for feature extraction.
    """
    
    def __init__(
        self,
        video_df: pd.DataFrame,
        videos_root: Path,
        snippet_size: int = 16,
        snippet_stride: int = 16,
        processor=None,
        crop_helper: Optional[CropHelper] = None,
    ):
        """
        Args:
            video_df: DataFrame with video_file, child_id, filename_stem columns.
            videos_root: Root directory for video files.
            snippet_size: Number of frames per snippet.
            snippet_stride: Stride between snippets (frames).
            processor: HuggingFace video processor for normalization.
            crop_helper: Optional CropHelper instance for SAM3-based cropping.
        """
        self.video_df = video_df
        self.videos_root = videos_root
        self.snippet_size = snippet_size
        self.snippet_stride = snippet_stride
        self.processor = processor
        self.crop_helper = crop_helper
        
        # Build list of (video_idx, video_info) tuples
        self.records = []
        for idx, row in video_df.iterrows():
            record = {
                "video_file": row["video_file"],
                "child_id": row["child_id"],
                "filename_stem": row["filename_stem"],
                "feature_name": f"{row['child_id']}_{row['filename_stem']}",
            }
            # Add video_path if available (from window splits CSV)
            if "video_path" in row and pd.notna(row["video_path"]):
                record["video_path"] = row["video_path"]
            self.records.append(record)
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx: int):
        """
        Load video and extract all snippets.
        
        Returns:
            dict with 'feature_name', 'snippets' (N, T, C, H, W), 'n_snippets'
        """
        rec = self.records[idx]
        # Use video_path directly if available (from window splits CSV), otherwise construct from video_file
        if "video_path" in rec and rec["video_path"]:
            video_path = Path(rec["video_path"])
        else:
            video_path = self.videos_root / rec["video_file"]
        
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            return None
        
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0), fault_tol=1)
            total_frames = len(vr)
            fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
        except Exception as e:
            logger.warning(f"Failed to open video {video_path}: {e}")
            return None
        
        # Calculate snippet positions
        n_snippets = max(1, (total_frames - self.snippet_size) // self.snippet_stride + 1)
        
        snippets = []
        for i in range(n_snippets):
            start_frame = i * self.snippet_stride
            end_frame = min(start_frame + self.snippet_size, total_frames)
            
            # Handle short videos
            if end_frame - start_frame < self.snippet_size:
                # Pad by repeating last frame
                indices = list(range(start_frame, end_frame))
                while len(indices) < self.snippet_size:
                    indices.append(end_frame - 1)
                indices = np.array(indices)
            else:
                indices = np.arange(start_frame, end_frame)
            
            try:
                frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
                
                # Apply cropping if enabled
                if self.crop_helper is not None and self.crop_helper.cfg.enabled:
                    frames, crop_meta = self.crop_helper.apply_crop(
                        frames=frames,
                        video_file=rec["video_file"],
                        frame_indices=indices,
                        fps=fps,
                        video_path=video_path,
                    )
                
                # Convert to tensor format (T, C, H, W)
                frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
                
                # Apply processor for normalization and resizing
                if self.processor:
                    inputs = self.processor(frames_tensor, return_tensors="pt")
                    # Get the processed video (B, T, C, H, W) -> (T, C, H, W)
                    snippet_tensor = inputs["pixel_values_videos"].squeeze(0)
                else:
                    # Fallback: just normalize to [0, 1]
                    snippet_tensor = frames_tensor.float() / 255.0
                
                snippets.append(snippet_tensor)
                
            except Exception as e:
                logger.warning(f"Error extracting snippet {i} from {video_path}: {e}")
                continue
        
        if len(snippets) == 0:
            return None
        
        # Stack snippets (N, T, C, H, W)
        snippets_tensor = torch.stack(snippets)
        
        return {
            "feature_name": rec["feature_name"],
            "snippets": snippets_tensor,
            "n_snippets": len(snippets),
            "video_file": rec["video_file"],
        }


# =============================================================================
# Feature extraction
# =============================================================================

def extract_features(
    model,
    dataloader: DataLoader,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Path]:
    """
    Extract features from all videos and save as NPY files.
    
    Args:
        model: V-JEPA2 model for feature extraction.
        dataloader: DataLoader yielding video snippets.
        output_dir: Directory to save feature NPY files.
        device: Device to run inference on.
    
    Returns:
        Dict mapping feature_name -> output_path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_features = {}
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if batch is None:
                continue
            
            feature_name = batch["feature_name"]
            snippets = batch["snippets"]  # (N, T, C, H, W)
            
            # Skip if file already exists
            out_path = output_dir / f"{feature_name}.npy"
            if out_path.exists():
                saved_features[feature_name] = out_path
                continue
            
            # Move to device
            snippets = snippets.to(device)
            
            # Extract features snippet by snippet (to avoid OOM)
            features_list = []
            for i in range(snippets.shape[0]):
                snippet = snippets[i].unsqueeze(0)  # (1, T, C, H, W)
                
                # Forward through model
                outputs = model.get_vision_features(snippet)
                
                # Get feature vector by mean-pooling over patches
                # outputs.last_hidden_state shape: (B, num_patches, hidden_dim)
                if hasattr(outputs, "last_hidden_state"):
                    features = outputs.last_hidden_state.mean(dim=1)  # (B, hidden_dim)
                elif hasattr(outputs, "pooler_output"):
                    features = outputs.pooler_output
                else:
                    # Direct tensor output
                    features = outputs.mean(dim=1) if outputs.dim() == 3 else outputs
                
                features_list.append(features.cpu().numpy())
            
            # Stack features (N, hidden_dim)
            all_features = np.vstack(features_list).astype(np.float32)
            
            # Save as NPY
            np.save(out_path, all_features)
            saved_features[feature_name] = out_path
            
            logger.debug(f"Saved features for {feature_name}: shape {all_features.shape}")
    
    return saved_features


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract V-JEPA2 features for OpenTAD/ActionFormer."
    )
    
    parser.add_argument(
        "--model-source",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint directory.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="CV fold index (0, 1, or 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for features. Default: auto-generated from model source.",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=VIDEOS_ROOT,
        help="Root directory for video files.",
    )
    parser.add_argument(
        "--dataprep-dir",
        type=Path,
        default=DATAPREP_DIR,
        help="Directory for data preparation files.",
    )
    parser.add_argument(
        "--enable-crop",
        action="store_true",
        help="Enable SAM3-based child cropping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (number of videos processed at once).",
    )
    parser.add_argument(
        "--num-dataloader-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )
    
    # Distributed processing
    parser.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="Worker index for distributed processing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of workers for distributed processing.",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger.info("=" * 60)
    logger.info("V-JEPA2 Feature Extraction")
    logger.info("=" * 60)
    logger.info(f"Model source: {args.model_source}")
    logger.info(f"Fold: {args.fold}")
    
    # Resolve model path
    model_path = Path(args.model_source)
    if not model_path.is_absolute():
        # Try relative to script's parent directory
        script_dir = Path(__file__).parent.parent
        model_path = script_dir / args.model_source
    
    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    
    logger.info(f"Resolved model path: {model_path}")
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_name = model_path.parent.name if model_path.name.startswith("fold") else model_path.name
        output_dir = OUTPUT_ROOT / f"{model_name}_fold_{args.fold}"
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load video index
    video_df = load_video_index(args.fold, args.dataprep_dir)
    
    # Distributed processing: split videos across workers
    if args.num_workers > 1:
        n_videos = len(video_df)
        worker_videos = np.array_split(range(n_videos), args.num_workers)[args.worker_index]
        video_df = video_df.iloc[worker_videos]
        logger.info(
            f"Worker {args.worker_index}/{args.num_workers}: "
            f"processing {len(video_df)} videos"
        )
    
    # Load model
    logger.info("Loading V-JEPA2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    from transformers import AutoModel, AutoVideoProcessor
    
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
    model = model.to(device)
    model.eval()
    
    processor = AutoVideoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    
    # Create crop helper if enabled
    crop_cfg = CropConfig(enabled=args.enable_crop)
    crop_helper = CropHelper(crop_cfg) if args.enable_crop else None
    
    if args.enable_crop:
        logger.info(
            "Cropping enabled | sam3_csv=%s | cache_base=%s | padding=%d",
            crop_cfg.sam3_parsed_csv,
            crop_cfg.mask_cache_base,
            crop_cfg.padding,
        )
    
    dataset = VideoFeatureDataset(
        video_df=video_df,
        videos_root=args.videos_root,
        snippet_size=SNIPPET_SIZE,
        snippet_stride=SNIPPET_STRIDE,
        processor=processor,
        crop_helper=crop_helper,
    )
    
    # Custom collate that passes through None
    def collate_single(batch):
        return batch[0] if batch else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one video at a time
        shuffle=False,
        num_workers=args.num_dataloader_workers,
        collate_fn=collate_single,
    )
    
    # Extract features
    logger.info(f"Extracting features from {len(dataset)} videos...")
    saved = extract_features(model, dataloader, output_dir, device)
    
    logger.info(f"Done! Saved {len(saved)} feature files to {output_dir}")


if __name__ == "__main__":
    main()

