#!/usr/bin/env python3
"""
Create pyskl annotation pickle files from SAILS pose HDF5 caches and split CSVs.

This script converts pose data extracted by batch_sam_pose.py (wholebody 133 keypoints)
to the pyskl format (COCO 17 keypoints) for PoseC3D training.

Usage:
    # MODE 1: Single split (train/val/test)
    python create_sails_annotations.py \
        --mode single \
        --splits-dir /path/to/single_split_4class/ \
        --output-dir data/sails/single \
        --min-keypoint-conf 0.6

    # MODE 2: Cross-validation (generates all folds)
    python create_sails_annotations.py \
        --mode cv \
        --splits-dir /path/to/cv_splits_4class/ \
        --output-dir data/sails/cv \
        --min-keypoint-conf 0.6

    # MODE 3: TAL Windows mode (includes background as additional class)
    # Use --windows flag to process TAL window CSVs (window_id, primary_label)
    python create_sails_annotations.py \
        --mode cv \
        --splits-dir /path/to/tal/splits_cv_4class/ \
        --output-dir data/sails/tal_cv \
        --windows \
        --min-keypoint-conf 0.6

    # Legacy: Multiple CSVs with split names (still supported)
    python create_sails_annotations.py \
        --csv train.csv val.csv test.csv \
        --split-names train val test \
        --output sails_full.pkl
"""

import argparse
import csv
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    warnings.warn("h5py not installed. Pose cache loading will be disabled.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default class mapping (from SAILS 4-class setup)
DEFAULT_CLASS_MAP_4CLASS = {
    "hands flapping": 0,
    "jumping": 1,
    "rocking": 2,
    "spinning": 3,
}

# 5-class mapping (includes "one hand flap")
DEFAULT_CLASS_MAP_5CLASS = {
    "hands flapping": 0,
    "jumping": 1,
    "one hand flap": 2,
    "rocking": 3,
    "spinning": 4,
}

# Default to 4-class for backward compatibility
DEFAULT_CLASS_MAP = DEFAULT_CLASS_MAP_4CLASS

# TAL windows class mappings (includes background)
# Background gets the highest index in each configuration
TAL_CLASS_MAP_4CLASS = {
    "hands flapping": 0,
    "jumping": 1,
    "rocking": 2,
    "spinning": 3,
    "background": 4,  # Added for TAL windows
}

TAL_CLASS_MAP_5CLASS = {
    "hands flapping": 0,
    "jumping": 1,
    "one hand flap": 2,
    "rocking": 3,
    "spinning": 4,
    "background": 5,  # Added for TAL windows
}

# Reverse mapping from integer labels to class names (for TAL windows)
TAL_ID_TO_CLASS_4CLASS = {v: k for k, v in TAL_CLASS_MAP_4CLASS.items()}
TAL_ID_TO_CLASS_5CLASS = {v: k for k, v in TAL_CLASS_MAP_5CLASS.items()}

# COCO keypoint indices in wholebody (133 keypoints)
# The first 17 keypoints in wholebody ARE the COCO body keypoints
COCO_KEYPOINT_INDICES = list(range(17))

# Default paths for SAILS dataset
DEFAULT_POSE_CACHE_BASE = "/orcd/scratch/bcs/001/sensein/sails/cache_for_tracking"
DEFAULT_VIDEO_ROOT = "/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized"
DEFAULT_VIDEO_META_JSON = "/orcd/data/satra/001/users/brukew/actreg/dataprep/video_meta.json"

# Pose cache filename pattern (from batch_sam_pose.py)
# Format: {det_config}_{det_thresh}_{pose_config}_sam3guided.h5
POSE_CACHE_FILENAME = "dino-5scale_swin-l_8xb2-36e_coco_0.5_td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288_sam3guided.h5"


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SegmentInfo:
    """Information about a single video segment."""
    segment_id: str
    video_file: str
    filename: str
    start_sec: float
    end_sec: float
    label: int
    label_name: str
    child_id: str
    timepoint: str
    video_duration: Optional[float] = None
    img_shape: Optional[Tuple[int, int]] = None  # (height, width)


@dataclass
class PoseAnnotation:
    """Annotation in pyskl format."""
    frame_dir: str
    total_frames: int
    img_shape: Tuple[int, int]
    original_shape: Tuple[int, int]
    label: int
    keypoint: np.ndarray  # [M, T, V, C] - persons, frames, keypoints, coords
    keypoint_score: np.ndarray  # [M, T, V] - confidence scores


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def normalize_fieldnames(fieldnames: List[str]) -> List[str]:
    """Clean field names to handle BOM and variations."""
    cleaned = []
    for name in fieldnames:
        cleaned_name = name.replace("\ufeff", "").strip()
        lowered = cleaned_name.lower()
        
        if lowered == "segment_global_id":
            cleaned.append("segment_global_id")
        elif lowered == "window_id":
            # TAL window CSVs use window_id instead of segment_id
            cleaned.append("segment_id")
        elif lowered.endswith("segment_id"):
            cleaned.append("segment_id")
        elif lowered == "video_file":
            cleaned.append("video_file")
        elif lowered == "filename":
            cleaned.append("filename")
        elif lowered == "start_sec":
            cleaned.append("start_sec")
        elif lowered == "end_sec":
            cleaned.append("end_sec")
        elif lowered == "rmm_type":
            cleaned.append("rmm_type")
        elif lowered == "primary_label":
            # TAL window CSVs use primary_label (int) instead of rmm_type (str)
            cleaned.append("primary_label")
        elif lowered == "child_id":
            cleaned.append("child_id")
        elif lowered == "timepoint":
            cleaned.append("timepoint")
        elif lowered == "video_duration":
            cleaned.append("video_duration")
        elif lowered == "duration_sec":
            # TAL windows have duration_sec instead of video_duration
            cleaned.append("video_duration")
        else:
            cleaned.append(cleaned_name)
    return cleaned


def load_csv_segments(csv_path: Path, class_map: Dict[str, int]) -> List[SegmentInfo]:
    """Load segments from a CSV file."""
    segments = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = normalize_fieldnames(reader.fieldnames)
        
        for row in reader:
            segment_id = row.get("segment_id") or row.get("segment_global_id")
            if not segment_id:
                continue
            
            rmm_type = row.get("rmm_type", "").strip()
            if rmm_type not in class_map:
                print(f"  [warn] Unknown class '{rmm_type}' for segment {segment_id}, skipping")
                continue
            
            try:
                start_sec = float(row.get("start_sec", 0))
                end_sec = float(row.get("end_sec", 0))
            except ValueError:
                print(f"  [warn] Invalid start/end for segment {segment_id}, skipping")
                continue
            
            video_duration = None
            if row.get("video_duration"):
                try:
                    video_duration = float(row["video_duration"])
                except ValueError:
                    pass
            
            segments.append(SegmentInfo(
                segment_id=segment_id,
                video_file=row.get("video_file", ""),
                filename=row.get("filename", ""),
                start_sec=start_sec,
                end_sec=end_sec,
                label=class_map[rmm_type],
                label_name=rmm_type,
                child_id=row.get("child_id", ""),
                timepoint=row.get("timepoint", ""),
                video_duration=video_duration,
            ))
    
    return segments


def load_csv_windows(
    csv_path: Path,
    id_to_class: Dict[int, str],
    background_label: int,
) -> List[SegmentInfo]:
    """
    Load TAL window segments from a CSV file.
    
    TAL window CSVs have:
    - window_id (mapped to segment_id)
    - primary_label (integer: 0-3 for RMM classes, -1 for background)
    - start_sec, end_sec (half-open interval [start, end))
    
    Args:
        csv_path: Path to the window CSV file.
        id_to_class: Mapping from integer label to class name.
        background_label: Integer label to assign to background windows (primary_label=-1).
    
    Returns:
        List of SegmentInfo objects.
    """
    segments = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = normalize_fieldnames(reader.fieldnames)
        
        for row in reader:
            segment_id = row.get("segment_id") or row.get("segment_global_id")
            if not segment_id:
                continue
            
            # Parse primary_label as integer
            try:
                primary_label = int(row.get("primary_label", -1))
            except (ValueError, TypeError):
                print(f"  [warn] Invalid primary_label for {segment_id}, skipping")
                continue
            
            # Map -1 (background) to the background class
            if primary_label == -1:
                label = background_label
                label_name = "background"
            else:
                label = primary_label
                label_name = id_to_class.get(primary_label, f"class_{primary_label}")
            
            try:
                start_sec = float(row.get("start_sec", 0))
                end_sec = float(row.get("end_sec", 0))
            except ValueError:
                print(f"  [warn] Invalid start/end for {segment_id}, skipping")
                continue
            
            video_duration = None
            if row.get("video_duration"):
                try:
                    video_duration = float(row["video_duration"])
                except ValueError:
                    pass
            
            segments.append(SegmentInfo(
                segment_id=segment_id,
                video_file=row.get("video_file", ""),
                filename=row.get("filename", ""),
                start_sec=start_sec,
                end_sec=end_sec,
                label=label,
                label_name=label_name,
                child_id=row.get("child_id", ""),
                timepoint=row.get("timepoint", ""),
                video_duration=video_duration,
            ))
    
    return segments


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Get video metadata (fps, frame_count, width, height)."""
    info = {"fps": 30.0, "frame_count": 0, "width": 0, "height": 0}
    
    if HAS_CV2 and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            info["fps"] = cap.get(cv2.CAP_PROP_FPS) or 30.0
            info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
    
    return info


def resolve_video_path(video_root: Path, video_file: str) -> Path:
    """Convert CSV video path to standardized .mp4 path."""
    path_str = video_file.replace("\\", "/")
    
    # Remove common prefixes
    prefixes = [
        "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/",
        "/Volumes/",
    ]
    for prefix in prefixes:
        if path_str.startswith(prefix):
            path_str = path_str[len(prefix):]
            break
    
    relative_path = Path(path_str.lstrip("/"))
    
    # Try .mp4 extension first (standardized), then original
    candidates = [
        video_root / relative_path.with_suffix(".mp4"),
        video_root / relative_path,
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Return the first candidate even if not found (for error reporting)
    return candidates[0]


def load_pose_cache(cache_path: Path) -> Optional[Dict[int, List[Dict]]]:
    """Load pose results from HDF5 cache."""
    if not HAS_H5PY:
        return None
    if not cache_path.exists():
        return None
    
    poses = {}
    try:
        with h5py.File(cache_path, "r") as f:
            for frame_key in f.keys():
                if not frame_key.startswith("frame_"):
                    continue
                frame_idx = int(frame_key.split("_")[1])
                frame_group = f[frame_key]
                
                frame_poses = []
                for pose_key in sorted(frame_group.keys()):
                    pose_group = frame_group[pose_key]
                    frame_poses.append({
                        "keypoints": pose_group["keypoints"][:],
                        "bbox": pose_group["bbox"][:] if "bbox" in pose_group else None,
                    })
                poses[frame_idx] = frame_poses
        return poses
    except Exception as e:
        print(f"  [warn] Error loading pose cache {cache_path}: {e}")
        return None


def find_pose_cache(
    segment: SegmentInfo, 
    pose_cache_base: Path,
    video_meta_lookup: Optional[Dict[str, Dict]] = None,
) -> Optional[Path]:
    """
    Find the pose cache file for a segment.
    
    The pose cache structure from batch_sam_pose.py is:
        {pose_cache_base}/pose_sam3/{video_basename}/{POSE_CACHE_FILENAME}
    
    Where video_basename is the stem of the original video filename.
    """
    filename_stem = Path(segment.filename).stem
    
    # Primary path: matches batch_sam_pose.py output structure
    primary_path = pose_cache_base / "pose_sam3" / filename_stem / POSE_CACHE_FILENAME
    if primary_path.exists() and filename_stem!= "IMG_5399":
        return primary_path
    
    # Try video_meta lookup if provided (maps FileName to row info)
    if video_meta_lookup and segment.filename in video_meta_lookup:
        meta = video_meta_lookup[segment.filename]
        original_path = meta.get("original_path", "")
        altered_basename = meta.get("altered_basename", "")
        if altered_basename:
            video_basename = altered_basename
            alt_path = pose_cache_base / "pose_sam3" / video_basename / POSE_CACHE_FILENAME
            if alt_path.exists():
                return alt_path
        if original_path:
            video_basename = Path(original_path).stem
            alt_path = pose_cache_base / "pose_sam3" / video_basename / POSE_CACHE_FILENAME
            if alt_path.exists():
                return alt_path
    
    # Fallback: try without the pose_sam3 subdirectory
    fallback_candidates = [
        pose_cache_base / filename_stem / POSE_CACHE_FILENAME,
        pose_cache_base / f"{filename_stem}_pose.h5",
        pose_cache_base / f"{filename_stem}.h5",
    ]
    
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate
    
    return None


def load_video_meta(video_meta_path: Path) -> Dict[str, Dict]:
    """Load video metadata JSON and create lookup by FileName."""
    if not video_meta_path.exists():
        return {}
    
    try:
        with open(video_meta_path, "r") as f:
            data = json.load(f)
        
        lookup = {}
        for record in data.get("records", []):
            filename = record.get("FileName", "")
            if filename:
                lookup[filename] = record
        return lookup
    except Exception as e:
        print(f"Warning: Could not load video meta: {e}")
        return {}


def extract_coco_keypoints(
    wholebody_kp: np.ndarray,
    min_conf: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract COCO 17 keypoints from wholebody 133 keypoints.
    
    Args:
        wholebody_kp: Array of shape [V, 3] where V=133 (or [V, 2] + scores)
                      Format: [x, y, score] per keypoint
        min_conf: Minimum confidence threshold. Keypoints below this are zeroed out.
    
    Returns:
        keypoint_xy: [17, 2] - x, y coordinates
        keypoint_score: [17] - confidence scores
    """
    if wholebody_kp.shape[0] < 17:
        # Pad with zeros if not enough keypoints
        padded = np.zeros((17, wholebody_kp.shape[1]))
        padded[:wholebody_kp.shape[0]] = wholebody_kp
        wholebody_kp = padded
    
    coco_kp = wholebody_kp[COCO_KEYPOINT_INDICES]
    
    if coco_kp.shape[1] >= 3:
        keypoint_xy = coco_kp[:, :2].copy()
        keypoint_score = coco_kp[:, 2].copy()
    else:
        keypoint_xy = coco_kp[:, :2].copy()
        keypoint_score = np.ones(17)
    
    # Zero out low-confidence keypoints
    if min_conf > 0:
        low_conf_mask = keypoint_score < min_conf
        keypoint_xy[low_conf_mask] = 0.0
        keypoint_score[low_conf_mask] = 0.0
    
    return keypoint_xy, keypoint_score


def create_annotation(
    segment: SegmentInfo,
    pose_cache_base: Path,
    video_root: Path,
    video_meta_lookup: Optional[Dict[str, Dict]] = None,
    default_fps: float = 30.0,
    default_shape: Tuple[int, int] = (480, 640),  # (height, width)
    min_keypoint_conf: float = 0.0,
) -> Optional[Dict]:
    """
    Create a single annotation entry for pyskl format.
    
    Returns:
        Dictionary with pyskl annotation format, or None if failed.
    """
    # Find pose cache
    cache_path = find_pose_cache(segment, pose_cache_base, video_meta_lookup)
    if cache_path is None:
        print(f"  [skip] No pose cache found for {segment.segment_id} (filename: {segment.filename})")
        return None
    
    # Load pose data
    pose_data = load_pose_cache(cache_path)
    if pose_data is None or len(pose_data) == 0:
        print(f"  [skip] Empty pose cache for {segment.segment_id}")
        return None
    
    # Get video info for FPS and shape
    # Prefer video_meta_lookup (from video_meta.json) to avoid slow OpenCV probing
    fps = default_fps
    img_shape = default_shape
    
    video_meta = video_meta_lookup.get(segment.video_file) if video_meta_lookup else None
    if video_meta:
        # Use cached metadata (fast path)
        if video_meta.get("fps", 0) > 0:
            fps = video_meta["fps"]
        if video_meta.get("width", 0) > 0 and video_meta.get("height", 0) > 0:
            img_shape = (video_meta["height"], video_meta["width"])
    else:
        # Fall back to OpenCV probe (slow path, only if metadata missing)
    video_path = resolve_video_path(video_root, segment.video_file)
    video_info = get_video_info(video_path)
        if video_info["fps"] > 0:
            fps = video_info["fps"]
    if video_info["width"] > 0 and video_info["height"] > 0:
        img_shape = (video_info["height"], video_info["width"])
    
    # Calculate frame range for this segment
    start_frame = int(segment.start_sec * fps)
    end_frame = int(segment.end_sec * fps)
    
    # Ensure at least 1 frame
    if end_frame <= start_frame:
        end_frame = start_frame + int(fps)  # Default to 1 second
    
    total_frames = end_frame - start_frame
    
    # Extract keypoints for the segment's frame range
    # We assume 1 person (the target child)
    num_persons = 1
    num_keypoints = 17  # COCO
    
    keypoints = np.zeros((num_persons, total_frames, num_keypoints, 2), dtype=np.float32)
    keypoint_scores = np.zeros((num_persons, total_frames, num_keypoints), dtype=np.float32)
    
    frames_with_pose = 0
    for t, frame_idx in enumerate(range(start_frame, end_frame)):
        if frame_idx in pose_data and len(pose_data[frame_idx]) > 0:
            # Take the first person's pose (should be the target child)
            pose = pose_data[frame_idx][0]
            kp_xy, kp_score = extract_coco_keypoints(pose["keypoints"], min_conf=min_keypoint_conf)
            keypoints[0, t] = kp_xy
            keypoint_scores[0, t] = kp_score
            frames_with_pose += 1
    
    # Skip if too few frames have pose data
    if frames_with_pose < total_frames * 0.1:  # Less than 10% coverage
        print(f"  [skip] Too few pose frames ({frames_with_pose}/{total_frames}) for {segment.segment_id}")
        return None
    
    return {
        "frame_dir": segment.segment_id,
        "total_frames": total_frames,
        "img_shape": img_shape,
        "original_shape": img_shape,
        "label": segment.label,
        "keypoint": keypoints,
        "keypoint_score": keypoint_scores,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create pyskl annotation pickle from SAILS CSVs and pose caches."
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["single", "cv", "manual"],
        default="manual",
        help="Generation mode: 'single' for train/val/test split, "
             "'cv' for cross-validation folds, 'manual' for legacy CSV-based (default).",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Directory containing split CSVs. For 'single' mode: expects train.csv, val.csv, test.csv. "
             "For 'cv' mode: expects fold_X_train.csv, fold_X_val.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for pickle files. For 'single' mode: creates conf{X}.pkl. "
             "For 'cv' mode: creates conf{X}/fold{Y}.pkl. Used with --mode single/cv.",
    )
    
    # Legacy mode (still supported)
    parser.add_argument(
        "--csv",
        nargs="+",
        default=None,
        help="(Legacy) One or more CSV files containing segment annotations.",
    )
    parser.add_argument(
        "--split-names",
        nargs="+",
        default=None,
        help="(Legacy) Split names for each CSV (e.g., 'train val test'). "
             "If not provided, uses CSV filename stems.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="(Legacy) Output pickle file path.",
    )
    
    # Common options
    parser.add_argument(
        "--pose-cache-base",
        type=str,
        default=DEFAULT_POSE_CACHE_BASE,
        help="Base directory for pose caches. Caches are at: "
             "{base}/pose_sam3/{video_basename}/{filename}.h5",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default=DEFAULT_VIDEO_ROOT,
        help="Root directory containing standardized video files.",
    )
    parser.add_argument(
        "--video-meta",
        type=str,
        default=DEFAULT_VIDEO_META_JSON,
        help="Path to video_meta.json for looking up video info.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        choices=[4, 5],
        default=4,
        help="Number of classes: 4 (default) or 5 (includes 'one hand flap').",
    )
    parser.add_argument(
        "--class-map",
        type=str,
        default=None,
        help="JSON file with class name to index mapping. "
             "If not provided, uses default based on --num-classes.",
    )
    parser.add_argument(
        "--label-map-output",
        type=str,
        default=None,
        help="Output path for label map text file (for pyskl).",
    )
    parser.add_argument(
        "--default-fps",
        type=float,
        default=30.0,
        help="Default FPS if video info unavailable.",
    )
    parser.add_argument(
        "--min-keypoint-conf",
        type=float,
        default=0.0,
        help="Minimum keypoint confidence threshold (0.0-1.0). "
             "Keypoints below this threshold are zeroed out. Default: 0.0 (no filtering).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be processed, don't create output.",
    )
    parser.add_argument(
        "--windows",
        action="store_true",
        help="Enable TAL windows mode. CSVs are expected to have window_id and primary_label "
             "instead of segment_id and rmm_type. Background windows (primary_label=-1) are "
             "included as an additional class.",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "manual":
        if not args.csv or not args.output:
            parser.error("--csv and --output are required for manual mode")
    else:
        if not args.splits_dir or not args.output_dir:
            parser.error(f"--splits-dir and --output-dir are required for {args.mode} mode")
    
    return args


def process_csvs(
    csv_paths: List[Path],
    split_names: List[str],
    output_path: Path,
    pose_cache_base: Path,
    video_root: Path,
    video_meta_lookup: Dict[str, Dict],
    class_map: Dict[str, int],
    default_fps: float = 30.0,
    min_keypoint_conf: float = 0.0,
    dry_run: bool = False,
    label_map_output: Optional[str] = None,
    windows_mode: bool = False,
    id_to_class: Optional[Dict[int, str]] = None,
    background_label: Optional[int] = None,
) -> bool:
    """
    Process CSVs and create a single pickle file.
    
    Args:
        windows_mode: If True, use load_csv_windows() for TAL window CSVs.
        id_to_class: Mapping from integer label to class name (required for windows_mode).
        background_label: Label index for background class (required for windows_mode).
    
    Returns:
        True if successful, False otherwise.
    """
    print(f"CSVs: {[p.name for p in csv_paths]}")
    print(f"Split names: {split_names}")
    if windows_mode:
        print(f"Windows mode: ON (background label={background_label})")
    if min_keypoint_conf > 0:
        print(f"Min keypoint confidence: {min_keypoint_conf} (keypoints below this are zeroed)")
    print()
    
    all_annotations = []
    split_dict = {}
    
    for csv_path, split_name in zip(csv_paths, split_names):
        print(f"Processing {csv_path.name} (split: {split_name})...")
        
        if windows_mode:
            segments = load_csv_windows(csv_path, id_to_class, background_label)
        else:
        segments = load_csv_segments(csv_path, class_map)
        print(f"  Loaded {len(segments)} segments")
        
        if dry_run:
            found_count = 0
            missing_count = 0
            for seg in segments[:5]:
                print(f"    - {seg.segment_id}: {seg.label_name} ({seg.start_sec:.1f}-{seg.end_sec:.1f}s)")
                cache_path = find_pose_cache(seg, pose_cache_base, video_meta_lookup)
                if cache_path:
                    print(f"      ✓ Pose cache: {cache_path}")
                    found_count += 1
                else:
                    print(f"      ✗ No pose cache found")
                    missing_count += 1
            
            for seg in segments[5:]:
                cache_path = find_pose_cache(seg, pose_cache_base, video_meta_lookup)
                if cache_path:
                    found_count += 1
                else:
                    missing_count += 1
            
            if len(segments) > 5:
                print(f"    ... and {len(segments) - 5} more")
            
            print(f"  Pose cache coverage: {found_count}/{len(segments)} ({100*found_count/len(segments):.1f}%)")
            if missing_count > 0:
                print(f"  ⚠️  Missing {missing_count} pose caches")
            
            split_dict[split_name] = [seg.segment_id for seg in segments]
            continue
        
        split_segment_ids = []
        
        for seg in segments:
            annotation = create_annotation(
                seg,
                pose_cache_base,
                video_root,
                video_meta_lookup=video_meta_lookup,
                default_fps=default_fps,
                min_keypoint_conf=min_keypoint_conf,
            )
            if annotation is not None:
                all_annotations.append(annotation)
                split_segment_ids.append(seg.segment_id)
        
        split_dict[split_name] = split_segment_ids
        print(f"  Created {len(split_segment_ids)} annotations")
    
    if dry_run:
        print(f"\n[Dry run] Would create: {output_path}")
        for split_name, segment_ids in split_dict.items():
            print(f"  - {split_name}: {len(segment_ids)} segments")
        return True
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pyskl_data = {
        "split": split_dict,
        "annotations": all_annotations,
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(pyskl_data, f)
    
    print(f"\n✓ Saved {len(all_annotations)} annotations to {output_path}")
    print(f"  Splits: {[(k, len(v)) for k, v in split_dict.items()]}")
    
    if label_map_output:
        label_map_path = Path(label_map_output)
        label_map_path.parent.mkdir(parents=True, exist_ok=True)
        index_to_name = {v: k for k, v in class_map.items()}
        with open(label_map_path, "w") as f:
            for i in range(len(class_map)):
                f.write(f"{index_to_name.get(i, f'class_{i}')}\n")
        print(f"  Saved label map to {label_map_path}")
    
    return True


def main():
    args = parse_args()
    
    # Load class map based on mode
    if args.class_map:
        with open(args.class_map, "r") as f:
            class_map = json.load(f)
        id_to_class = {v: k for k, v in class_map.items()}
        background_label = max(class_map.values()) + 1 if args.windows else None
    elif args.windows:
        # TAL windows mode: use class maps that include background
        if args.num_classes == 5:
            class_map = TAL_CLASS_MAP_5CLASS
            id_to_class = TAL_ID_TO_CLASS_5CLASS
            background_label = 5
        else:
            class_map = TAL_CLASS_MAP_4CLASS
            id_to_class = TAL_ID_TO_CLASS_4CLASS
            background_label = 4
    elif args.num_classes == 5:
        class_map = DEFAULT_CLASS_MAP_5CLASS
        id_to_class = {v: k for k, v in class_map.items()}
        background_label = None
    else:
        class_map = DEFAULT_CLASS_MAP_4CLASS
        id_to_class = {v: k for k, v in class_map.items()}
        background_label = None
    
    print(f"Class mapping: {class_map}")
    if args.windows:
        print(f"Windows mode: ON (background class index = {background_label})")
    
    pose_cache_base = Path(args.pose_cache_base)
    video_root = Path(args.video_root)
    
    # Load video metadata
    video_meta_lookup = {}
    if args.video_meta:
        video_meta_path = Path(args.video_meta)
        video_meta_lookup = load_video_meta(video_meta_path)
        print(f"Loaded {len(video_meta_lookup)} video metadata entries")
    
    print(f"Pose cache base: {pose_cache_base}")
    print(f"  Expected structure: {pose_cache_base}/pose_sam3/{{video_basename}}/{POSE_CACHE_FILENAME}")
    print(f"Video root: {video_root}")
    print()
    
    # Determine suffix for output filenames
    # For windows mode, add +1 to num_classes to account for background
    effective_classes = args.num_classes + 1 if args.windows else args.num_classes
    windows_tag = "_windows" if args.windows else ""
    conf_suffix = f"{effective_classes}class{windows_tag}_conf{args.min_keypoint_conf:.1f}".replace(".", "")
    
    if args.mode == "single":
        # Single split mode: train.csv, val.csv, test.csv
        # For windows mode: train_windows.csv, val_windows.csv, test_windows.csv
        splits_dir = Path(args.splits_dir)
        output_dir = Path(args.output_dir)
        
        csv_paths = []
        split_names = []
        for split in ["train", "val", "test"]:
            if args.windows:
                csv_path = splits_dir / f"{split}_windows.csv"
            else:
            csv_path = splits_dir / f"{split}.csv"
            if csv_path.exists():
                csv_paths.append(csv_path)
                split_names.append(split)
        
        if not csv_paths:
            print(f"Error: No CSVs found in {splits_dir}")
            return
        
        output_path = output_dir / f"{conf_suffix}.pkl"
        
        print(f"=" * 60)
        print(f"MODE: Single Split" + (" (Windows)" if args.windows else ""))
        print(f"Output: {output_path}")
        print(f"=" * 60)
        
        process_csvs(
            csv_paths=csv_paths,
            split_names=split_names,
            output_path=output_path,
            pose_cache_base=pose_cache_base,
            video_root=video_root,
            video_meta_lookup=video_meta_lookup,
            class_map=class_map,
            default_fps=args.default_fps,
            min_keypoint_conf=args.min_keypoint_conf,
            dry_run=args.dry_run,
            label_map_output=args.label_map_output,
            windows_mode=args.windows,
            id_to_class=id_to_class,
            background_label=background_label,
        )
    
    elif args.mode == "cv":
        # Cross-validation mode: fold_X_train.csv, fold_X_val.csv
        # For windows mode: fold_X_train_windows.csv, fold_X_val_windows.csv
        splits_dir = Path(args.splits_dir)
        output_dir = Path(args.output_dir)
        
        # Discover folds
        if args.windows:
            fold_files = list(splits_dir.glob("fold_*_train_windows.csv"))
        else:
        fold_files = list(splits_dir.glob("fold_*_train.csv"))
        fold_nums = sorted(set(int(f.stem.split("_")[1]) for f in fold_files))
        
        if not fold_nums:
            print(f"Error: No fold CSVs found in {splits_dir}")
            return
        
        print(f"=" * 60)
        print(f"MODE: Cross-Validation ({len(fold_nums)} folds)" + (" (Windows)" if args.windows else ""))
        print(f"Output directory: {output_dir}/{conf_suffix}/")
        print(f"=" * 60)
        
        for fold_num in fold_nums:
            print(f"\n--- Fold {fold_num} ---")
            
            csv_paths = []
            split_names = []
            
            if args.windows:
                train_csv = splits_dir / f"fold_{fold_num}_train_windows.csv"
                val_csv = splits_dir / f"fold_{fold_num}_val_windows.csv"
            else:
            train_csv = splits_dir / f"fold_{fold_num}_train.csv"
            val_csv = splits_dir / f"fold_{fold_num}_val.csv"
            
            if train_csv.exists():
                csv_paths.append(train_csv)
                split_names.append("train")
            if val_csv.exists():
                csv_paths.append(val_csv)
                split_names.append("val")
            
            if not csv_paths:
                print(f"  Skipping fold {fold_num}: no CSVs found")
                continue
            
            output_path = output_dir / conf_suffix / f"fold{fold_num}.pkl"
            
            # Skip if output already exists
            if output_path.exists():
                print(f"  Skipping fold {fold_num}: output already exists at {output_path}")
                continue
            
            process_csvs(
                csv_paths=csv_paths,
                split_names=split_names,
                output_path=output_path,
                pose_cache_base=pose_cache_base,
                video_root=video_root,
                video_meta_lookup=video_meta_lookup,
                class_map=class_map,
                default_fps=args.default_fps,
                min_keypoint_conf=args.min_keypoint_conf,
                dry_run=args.dry_run,
                windows_mode=args.windows,
                id_to_class=id_to_class,
                background_label=background_label,
            )
        
        print(f"\n{'=' * 60}")
        print(f"CV generation complete! Pickles saved to: {output_dir}/{conf_suffix}/")
        print(f"{'=' * 60}")
    
    else:  # manual mode (legacy)
        csv_paths = [Path(p) for p in args.csv]
        if args.split_names:
            if len(args.split_names) != len(csv_paths):
                raise ValueError(
                    f"Number of split names ({len(args.split_names)}) must match "
                    f"number of CSVs ({len(csv_paths)})"
                )
            split_names = args.split_names
        else:
            split_names = [p.stem for p in csv_paths]
        
        output_path = Path(args.output)
        
        print(f"=" * 60)
        print(f"MODE: Manual (Legacy)" + (" (Windows)" if args.windows else ""))
        print(f"Output: {output_path}")
        print(f"=" * 60)
        
        process_csvs(
            csv_paths=csv_paths,
            split_names=split_names,
            output_path=output_path,
            pose_cache_base=pose_cache_base,
            video_root=video_root,
            video_meta_lookup=video_meta_lookup,
            class_map=class_map,
            default_fps=args.default_fps,
            min_keypoint_conf=args.min_keypoint_conf,
            dry_run=args.dry_run,
            label_map_output=args.label_map_output,
            windows_mode=args.windows,
            id_to_class=id_to_class,
            background_label=background_label,
        )


if __name__ == "__main__":
    main()

