#!/usr/bin/env python3
"""
Generate TAL window-level splits with multi-label annotations.

This script creates sliding window splits (2s windows, 1s stride) for temporal
action localization. Windows are labeled using tIoU thresholding against
ground-truth RMM segments.

Key features:
- Multi-label support (windows can have multiple positive classes)
- Consistent half-open time semantics [start, end)
- tIoU-based window labeling with configurable threshold
- Preserves LCTO-safe split assignments from video-level mapping

Usage:
    python make_tal_window_splits.py --task 4class --mode cv \
        --window-len-sec 2.0 --stride-sec 1.0 --tiou-thresh 0.3

Output:
    - tal/splits_{mode}_{task}/*.csv (window-level split CSVs)
    - tal/label_maps/label_map_{task}.json (class name -> index)
    - tal/metadata_{task}_{mode}.json (generation parameters)
"""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Resolve paths relative to the repo root so this runs from any clone location.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())


# Class label definitions
LABEL_MAP_4CLASS = {
    "hands flapping": 0,
    "jumping": 1,
    "rocking": 2,
    "spinning": 3,
}

LABEL_MAP_5CLASS = {
    "hands flapping": 0,
    "jumping": 1,
    "one hand flap": 2,
    "rocking": 3,
    "spinning": 4,
}

# Priority order for tie-breaking in primary_label (higher index = lower priority)
PRIORITY_ORDER_4CLASS = ["hands flapping", "jumping", "rocking", "spinning"]
PRIORITY_ORDER_5CLASS = ["hands flapping", "jumping", "one hand flap", "rocking", "spinning"]


@dataclass
class GTSegment:
    """A ground-truth segment annotation."""
    segment_id: str
    video_key: str
    label: str
    label_idx: int
    start_sec: float  # Original annotation start
    end_sec: float    # Original annotation end (inclusive)
    start_exclusive: float  # Half-open start
    end_exclusive: float    # Half-open end (converted: original_end + 1.0)


@dataclass
class CoverageInfo:
    """Coverage information for mask or pose data."""
    cause: str  # "full", "sparse", "exceeds_cache", "no_data"
    severity: str  # "full", "mild", "moderate", "severe"
    frames_missing: int


@dataclass
class Window:
    """A sliding window with multi-label annotations."""
    window_id: str
    video_key: str
    video_file: str
    filename: str
    child_id: str
    video_path: str
    fps: float
    duration_sec: float
    start_sec: float  # Window start (half-open)
    end_sec: float    # Window end (half-open)
    labels: List[int] = field(default_factory=list)  # Multi-label class indices
    label_tious: Dict[int, float] = field(default_factory=dict)  # tIoU per class
    primary_label: int = -1  # Single label for compatibility (-1 = background)
    is_background: bool = True
    # Coverage info
    mask_coverage: Optional[CoverageInfo] = None
    pose_coverage: Optional[CoverageInfo] = None


def normalize_video_file(video_file: str) -> str:
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


def load_cache_info(cache_info_path: Path, cache_type: str = "mask") -> Dict[str, Dict]:
    """
    Load mask or pose cache info indexed by normalized source_file.
    
    Args:
        cache_info_path: Path to video_mask_info.json or video_pose_info.json
        cache_type: "mask" or "pose"
    
    Returns:
        dict[normalized_source_file] -> cache info dict with frame_indices as set
    """
    if not cache_info_path.exists():
        print(f"  Warning: Cache info file not found: {cache_info_path}")
        return {}
    
    with cache_info_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    lookup = {}
    for record in data.get("records", []):
        source_file = record.get("source_file", "")
        if not source_file:
            continue
        
        key = normalize_video_file(source_file)
        ci = record.get("cache_info", {})
        
        if cache_type == "pose":
            has_data = ci.get("has_data", False)
            frame_indices = set(ci.get("frame_indices", [])) if has_data else set()
        else:
            frame_indices = set(ci.get("frame_indices", []))
            has_data = len(frame_indices) > 0
        
        max_frame = max(frame_indices) if frame_indices else -1
        
        lookup[key] = {
            "has_data": has_data,
            "num_frames": ci.get("num_frames", 0),
            "frame_indices": frame_indices,
            "max_frame": max_frame,
        }
    
    return lookup


def compute_coverage(
    start_frame: int,
    end_frame: int,
    cache_info: Optional[Dict],
) -> CoverageInfo:
    """
    Compute coverage info for a window against a cache.
    
    Args:
        start_frame: Window start frame (inclusive)
        end_frame: Window end frame (inclusive)
        cache_info: Cache info dict with frame_indices, max_frame, has_data
    
    Returns:
        CoverageInfo with cause, severity, frames_missing
    """
    window_frames = set(range(start_frame, end_frame + 1))
    total_frames = len(window_frames)
    
    # No cache data
    if cache_info is None or not cache_info.get("has_data", False):
        return CoverageInfo(
            cause="no_data",
            severity="severe",
            frames_missing=total_frames,
        )
    
    frame_indices = cache_info.get("frame_indices", set())
    max_frame = cache_info.get("max_frame", -1)
    
    # Compute coverage
    covered = window_frames & frame_indices
    frames_missing = total_frames - len(covered)
    
    # Determine cause
    if frames_missing == 0:
        return CoverageInfo(cause="full", severity="full", frames_missing=0)
    
    if end_frame > max_frame:
        cause = "exceeds_cache"
    else:
        cause = "sparse"
    
    # Determine severity based on frames_missing
    if frames_missing <= 5:
        severity = "mild"
    elif frames_missing <= 20:
        severity = "moderate"
    else:
        severity = "severe"
    
    return CoverageInfo(cause=cause, severity=severity, frames_missing=frames_missing)


def compute_tiou(w_start: float, w_end: float, g_start: float, g_end: float) -> float:
    """
    Compute temporal Intersection over Union between window and GT segment.
    
    Both intervals are assumed to be half-open: [start, end)
    
    Args:
        w_start, w_end: Window interval
        g_start, g_end: GT segment interval (already converted to half-open)
    
    Returns:
        tIoU in [0, 1]
    """
    # Intersection
    inter_start = max(w_start, g_start)
    inter_end = min(w_end, g_end)
    intersection = max(0.0, inter_end - inter_start)
    
    # Union
    w_len = w_end - w_start
    g_len = g_end - g_start
    union = w_len + g_len - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def load_video_meta(video_meta_path: Path) -> Dict[str, Dict]:
    """
    Load video metadata indexed by SourceFile (normalized).
    
    Returns dict[normalized_source_file] -> video info dict
    """
    with video_meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    lookup = {}
    for record in data.get("records", []):
        source_file = record.get("SourceFile", "")
        if source_file:
            key = normalize_video_file(source_file)
            lookup[key] = record
    
    return lookup


def load_video_assignments(assignment_path: Path, mode: str) -> Dict[str, Dict[str, str]]:
    """
    Load video assignment CSV.
    
    Returns dict[video_key] -> {split info}
    """
    assignments = {}
    with assignment_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_key = row.get("video_key", "")
            if video_key:
                assignments[video_key] = dict(row)
    return assignments


def load_gt_segments(
    splits_root: Path,
    task: str,
    mode: str,
    label_map: Dict[str, int],
) -> Dict[str, List[GTSegment]]:
    """
    Load all GT segments from split CSVs.
    
    Returns dict[video_key] -> list of GTSegment
    """
    # Determine directory
    if task == "4class":
        split_dir = "cv_splits_4class" if mode == "cv" else "single_split_4class"
    else:
        split_dir = "cv_splits" if mode == "cv" else "single_split"
    
    split_path = splits_root / split_dir
    
    # Find all CSVs
    if mode == "cv":
        csv_files = sorted(split_path.glob("fold_*_*.csv"))
    else:
        csv_files = [split_path / f"{s}.csv" for s in ("train", "val", "test")]
        csv_files = [f for f in csv_files if f.exists()]
    
    segments_by_video: Dict[str, List[GTSegment]] = defaultdict(list)
    seen_segment_ids: Set[str] = set()
    
    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                segment_id = row.get("segment_id", "")
                
                # Skip duplicates (same segment appears in multiple fold CSVs)
                if segment_id in seen_segment_ids:
                    continue
                seen_segment_ids.add(segment_id)
                
                video_file = row.get("video_file", "")
                video_key = normalize_video_file(video_file)
                label_name = row.get("rmm_type", "")
                
                if label_name not in label_map:
                    continue  # Skip labels not in this task's label set
                
                try:
                    start_sec = float(row.get("start_sec", 0))
                    end_sec = float(row.get("end_sec", 0))
                except ValueError:
                    continue
                
                # Convert to half-open interval
                # Original annotation "2-4 means 2-5", so end_exclusive = end_sec + 1.0
                start_exclusive = start_sec
                end_exclusive = end_sec + 1.0
                
                gt = GTSegment(
                    segment_id=segment_id,
                    video_key=video_key,
                    label=label_name,
                    label_idx=label_map[label_name],
                    start_sec=start_sec,
                    end_sec=end_sec,
                    start_exclusive=start_exclusive,
                    end_exclusive=end_exclusive,
                )
                segments_by_video[video_key].append(gt)
    
    return dict(segments_by_video)


def generate_windows_for_video(
    video_key: str,
    video_info: Dict[str, Any],
    assignment: Dict[str, str],
    gt_segments: List[GTSegment],
    window_len: float,
    stride: float,
    tiou_thresh: float,
    label_map: Dict[str, int],
    priority_order: List[str],
    mask_cache: Optional[Dict] = None,
    pose_cache: Optional[Dict] = None,
) -> List[Window]:
    """
    Generate sliding windows for a single video with multi-label annotations.
    
    Args:
        video_key: Normalized video file path
        video_info: Video metadata from video_meta.json
        assignment: Split assignment info
        gt_segments: List of GT segments for this video
        window_len: Window length in seconds
        stride: Window stride in seconds
        tiou_thresh: tIoU threshold for positive labels
        label_map: Class name to index mapping
        priority_order: Class priority for tie-breaking
        mask_cache: Mask cache info for this video (optional)
        pose_cache: Pose cache info for this video (optional)
    
    Returns:
        List of Window objects
    """
    windows = []
    
    # Get video properties
    fps = video_info.get("fps", 30.0)
    duration = video_info.get("duration", 0.0)
    video_path = video_info.get("original_path", "")
    filename = video_info.get("FileName", "")
    video_file = video_info.get("SourceFile", "")
    child_id = assignment.get("child_id", "")
    
    if duration < window_len:
        # Video too short for even one window
        return windows
    
    # Generate windows
    num_classes = len(label_map)
    w_start = 0.0
    window_idx = 0
    
    while w_start + window_len <= duration + 0.001:  # Small epsilon for float comparison
        w_end = w_start + window_len
        
        # Ensure we don't exceed video duration
        if w_end > duration:
            w_end = duration
            if w_end - w_start < window_len * 0.5:
                # Window too short, skip
                break
        
        # Create window ID
        start_ms = int(w_start * 1000)
        end_ms = int(w_end * 1000)
        window_id = f"{child_id}_{Path(filename).stem}__t{start_ms}_{end_ms}"
        
        # Compute frame range for coverage
        start_frame = int(w_start * fps)
        end_frame = int(w_end * fps)
        
        # Compute coverage for mask and pose
        mask_coverage = compute_coverage(start_frame, end_frame, mask_cache)
        pose_coverage = compute_coverage(start_frame, end_frame, pose_cache)
        
        # Compute labels via tIoU
        class_max_tiou: Dict[int, float] = {i: 0.0 for i in range(num_classes)}
        
        for gt in gt_segments:
            tiou = compute_tiou(w_start, w_end, gt.start_exclusive, gt.end_exclusive)
            if tiou > class_max_tiou[gt.label_idx]:
                class_max_tiou[gt.label_idx] = tiou
        
        # Determine positive labels
        labels = []
        for class_idx, max_tiou in class_max_tiou.items():
            if max_tiou >= tiou_thresh:
                labels.append(class_idx)
        
        # Determine primary label (single label for compatibility)
        primary_label = -1
        if labels:
            # Find class with maximum tIoU
            best_tiou = -1.0
            best_class = -1
            for class_idx in labels:
                if class_max_tiou[class_idx] > best_tiou:
                    best_tiou = class_max_tiou[class_idx]
                    best_class = class_idx
                elif class_max_tiou[class_idx] == best_tiou and best_class >= 0:
                    # Tie-break by priority order
                    idx_to_name = {v: k for k, v in label_map.items()}
                    current_name = idx_to_name.get(best_class, "")
                    new_name = idx_to_name.get(class_idx, "")
                    current_priority = priority_order.index(current_name) if current_name in priority_order else 999
                    new_priority = priority_order.index(new_name) if new_name in priority_order else 999
                    if new_priority < current_priority:
                        best_class = class_idx
            primary_label = best_class
        
        window = Window(
            window_id=window_id,
            video_key=video_key,
            video_file=video_file,
            filename=filename,
            child_id=child_id,
            video_path=video_path,
            fps=fps,
            duration_sec=duration,
            start_sec=w_start,
            end_sec=w_end,
            labels=sorted(labels),
            label_tious=class_max_tiou,
            primary_label=primary_label,
            is_background=(len(labels) == 0),
            mask_coverage=mask_coverage,
            pose_coverage=pose_coverage,
        )
        windows.append(window)
        
        w_start += stride
        window_idx += 1
    
    return windows


def write_window_csv(
    windows: List[Window],
    output_path: Path,
    label_map: Dict[str, int],
):
    """Write windows to CSV file."""
    fieldnames = [
        "window_id",
        "video_key",
        "video_file",
        "filename",
        "child_id",
        "video_path",
        "fps",
        "duration_sec",
        "start_sec",
        "end_sec",
        "labels",
        "primary_label",
        "is_background",
    ]
    
    # Add per-class tIoU columns
    idx_to_name = {v: k for k, v in label_map.items()}
    for i in range(len(label_map)):
        class_name = idx_to_name[i].replace(" ", "_")
        fieldnames.append(f"tiou_{class_name}")
    
    # Add mask/pose coverage columns
    fieldnames.extend([
        "mask_cause",
        "mask_severity",
        "mask_frames_missing",
        "pose_cause",
        "pose_severity",
        "pose_frames_missing",
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for window in windows:
            row = {
                "window_id": window.window_id,
                "video_key": window.video_key,
                "video_file": window.video_file,
                "filename": window.filename,
                "child_id": window.child_id,
                "video_path": window.video_path,
                "fps": f"{window.fps:.6f}",
                "duration_sec": f"{window.duration_sec:.3f}",
                "start_sec": f"{window.start_sec:.3f}",
                "end_sec": f"{window.end_sec:.3f}",
                "labels": json.dumps(window.labels),
                "primary_label": window.primary_label,
                "is_background": 1 if window.is_background else 0,
            }
            
            for i in range(len(label_map)):
                class_name = idx_to_name[i].replace(" ", "_")
                row[f"tiou_{class_name}"] = f"{window.label_tious.get(i, 0.0):.4f}"
            
            # Add coverage columns
            if window.mask_coverage:
                row["mask_cause"] = window.mask_coverage.cause
                row["mask_severity"] = window.mask_coverage.severity
                row["mask_frames_missing"] = window.mask_coverage.frames_missing
            else:
                row["mask_cause"] = "no_data"
                row["mask_severity"] = "severe"
                row["mask_frames_missing"] = -1
            
            if window.pose_coverage:
                row["pose_cause"] = window.pose_coverage.cause
                row["pose_severity"] = window.pose_coverage.severity
                row["pose_frames_missing"] = window.pose_coverage.frames_missing
            else:
                row["pose_cause"] = "no_data"
                row["pose_severity"] = "severe"
                row["pose_frames_missing"] = -1
            
            writer.writerow(row)


def compute_statistics(
    windows: List[Window],
    label_map: Dict[str, int],
) -> Dict[str, Any]:
    """Compute summary statistics for windows."""
    total = len(windows)
    background = sum(1 for w in windows if w.is_background)
    
    class_counts = {name: 0 for name in label_map.keys()}
    idx_to_name = {v: k for k, v in label_map.items()}
    
    for window in windows:
        for label_idx in window.labels:
            class_name = idx_to_name.get(label_idx, f"class_{label_idx}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Multi-label stats
    multi_label_count = sum(1 for w in windows if len(w.labels) > 1)
    
    return {
        "total_windows": total,
        "background_windows": background,
        "background_pct": round(100.0 * background / total, 2) if total > 0 else 0,
        "positive_windows": total - background,
        "multi_label_windows": multi_label_count,
        "class_counts": class_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate TAL window-level splits with multi-label annotations."
    )
    parser.add_argument(
        "--video-meta",
        type=Path,
        default=_REPO_ROOT / "dataprep/video_meta.json",
        help="Path to video_meta.json",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=_REPO_ROOT / "dataprep/splits",
        help="Root directory containing existing segment split CSVs.",
    )
    parser.add_argument(
        "--assignment-dir",
        type=Path,
        default=_REPO_ROOT / "dataprep/tal/video_assignment",
        help="Directory containing video_to_split CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "dataprep/tal",
        help="Output directory for TAL splits.",
    )
    parser.add_argument(
        "--task",
        choices=["4class", "5class"],
        required=True,
        help="Task variant (4class or 5class).",
    )
    parser.add_argument(
        "--mode",
        choices=["cv", "single"],
        required=True,
        help="Split mode (cv for cross-validation, single for train/val/test).",
    )
    parser.add_argument(
        "--window-len-sec",
        type=float,
        default=2.0,
        help="Window length in seconds.",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=1.0,
        help="Window stride in seconds.",
    )
    parser.add_argument(
        "--tiou-thresh",
        type=float,
        default=0.3,
        help="tIoU threshold for positive labels.",
    )
    parser.add_argument(
        "--mask-info",
        type=Path,
        default=_REPO_ROOT / "dataprep/video_mask_info.json",
        help="Path to video_mask_info.json for coverage info.",
    )
    parser.add_argument(
        "--pose-info",
        type=Path,
        default=_REPO_ROOT / "dataprep/video_pose_info.json",
        help="Path to video_pose_info.json for coverage info.",
    )
    args = parser.parse_args()
    
    # Select label map and priority order
    if args.task == "4class":
        label_map = LABEL_MAP_4CLASS
        priority_order = PRIORITY_ORDER_4CLASS
    else:
        label_map = LABEL_MAP_5CLASS
        priority_order = PRIORITY_ORDER_5CLASS
    
    print(f"Generating TAL window splits")
    print(f"  Task: {args.task}, Mode: {args.mode}")
    print(f"  Window: {args.window_len_sec}s, Stride: {args.stride_sec}s")
    print(f"  tIoU threshold: {args.tiou_thresh}")
    print(f"  Classes: {list(label_map.keys())}")
    
    # Load video metadata
    print(f"\nLoading video metadata from {args.video_meta}")
    video_meta = load_video_meta(args.video_meta)
    print(f"  Loaded {len(video_meta)} videos")
    
    # Load mask and pose cache info
    print(f"\nLoading mask cache info from {args.mask_info}")
    mask_cache_lookup = load_cache_info(args.mask_info, cache_type="mask")
    print(f"  Loaded {len(mask_cache_lookup)} videos with mask cache")
    
    print(f"\nLoading pose cache info from {args.pose_info}")
    pose_cache_lookup = load_cache_info(args.pose_info, cache_type="pose")
    print(f"  Loaded {len(pose_cache_lookup)} videos with pose cache")
    
    # Load video assignments
    assignment_file = args.assignment_dir / f"video_to_split_{args.task}_{args.mode}.csv"
    print(f"\nLoading video assignments from {assignment_file}")
    assignments = load_video_assignments(assignment_file, args.mode)
    print(f"  Loaded {len(assignments)} video assignments")
    
    # Load GT segments
    print(f"\nLoading GT segments from {args.splits_root}")
    gt_segments = load_gt_segments(args.splits_root, args.task, args.mode, label_map)
    total_segments = sum(len(segs) for segs in gt_segments.values())
    print(f"  Loaded {total_segments} segments across {len(gt_segments)} videos")
    
    # Generate windows per split
    output_base = args.output_dir / f"splits_{args.mode}_{args.task}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Determine splits to generate
    if args.mode == "cv":
        splits_to_generate = []
        for fold in range(3):
            splits_to_generate.append((f"fold_{fold}_train", f"fold_{fold}_split", "train"))
            splits_to_generate.append((f"fold_{fold}_val", f"fold_{fold}_split", "val"))
    else:
        splits_to_generate = [
            ("train", "split", "train"),
            ("val", "split", "val"),
            ("test", "split", "test"),
        ]
    
    all_stats = {}
    
    for split_name, split_col, split_type in splits_to_generate:
        print(f"\nGenerating {split_name} windows...")
        
        # Filter videos for this split
        split_videos = []
        for video_key, assignment in assignments.items():
            assigned_split = assignment.get(split_col, "")
            if assigned_split == split_type:
                split_videos.append((video_key, assignment))
        
        print(f"  Videos in split: {len(split_videos)}")
        
        # Generate windows for each video
        all_windows = []
        videos_with_meta = 0
        videos_missing_meta = 0
        
        for video_key, assignment in split_videos:
            if video_key not in video_meta:
                videos_missing_meta += 1
                continue
            
            videos_with_meta += 1
            video_info = video_meta[video_key]
            video_gt = gt_segments.get(video_key, [])
            
            # Get cache info for this video
            mask_cache = mask_cache_lookup.get(video_key)
            pose_cache = pose_cache_lookup.get(video_key)
            
            windows = generate_windows_for_video(
                video_key=video_key,
                video_info=video_info,
                assignment=assignment,
                gt_segments=video_gt,
                window_len=args.window_len_sec,
                stride=args.stride_sec,
                tiou_thresh=args.tiou_thresh,
                label_map=label_map,
                priority_order=priority_order,
                mask_cache=mask_cache,
                pose_cache=pose_cache,
            )
            all_windows.extend(windows)
        
        if videos_missing_meta > 0:
            print(f"  Warning: {videos_missing_meta} videos missing from video_meta.json")
        
        # Write CSV
        output_path = output_base / f"{split_name}_windows.csv"
        write_window_csv(all_windows, output_path, label_map)
        print(f"  Wrote {len(all_windows)} windows to {output_path}")
        
        # Compute stats
        stats = compute_statistics(all_windows, label_map)
        stats["videos_with_meta"] = videos_with_meta
        stats["videos_missing_meta"] = videos_missing_meta
        all_stats[split_name] = stats
        
        print(f"  Background: {stats['background_pct']:.1f}% ({stats['background_windows']}/{stats['total_windows']})")
        print(f"  Multi-label windows: {stats['multi_label_windows']}")
        for class_name, count in stats["class_counts"].items():
            print(f"    {class_name}: {count}")
    
    # Write label map
    label_map_path = args.output_dir / "label_maps" / f"label_map_{args.task}.json"
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"\nWrote label map to {label_map_path}")
    
    # Write metadata
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "mode": args.mode,
        "window_len_sec": args.window_len_sec,
        "stride_sec": args.stride_sec,
        "tiou_thresh": args.tiou_thresh,
        "label_map": label_map,
        "priority_order": priority_order,
        "time_semantics": "half-open [start, end) intervals; GT end converted via end_exclusive = end_sec + 1.0",
        "split_statistics": all_stats,
    }
    metadata_path = args.output_dir / f"metadata_{args.task}_{args.mode}.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {metadata_path}")
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

