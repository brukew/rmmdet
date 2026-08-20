#!/usr/bin/env python3
"""
Validate TAL window splits for correctness.

This script performs sanity checks on the generated TAL window splits:
1. Leakage check: Verify no video appears in multiple incompatible splits
2. Coverage check: Verify windows cover [0, duration) for each video
3. Label sanity: Sample windows and verify labels match GT segments
4. Summary report: Print counts and statistics

Usage:
    python validate_tal_splits.py --task 4class --mode cv
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Resolve paths relative to the repo root so this runs from any clone location.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())


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


def compute_tiou(w_start: float, w_end: float, g_start: float, g_end: float) -> float:
    """Compute temporal IoU between window and GT segment (half-open intervals)."""
    inter_start = max(w_start, g_start)
    inter_end = min(w_end, g_end)
    intersection = max(0.0, inter_end - inter_start)
    w_len = w_end - w_start
    g_len = g_end - g_start
    union = w_len + g_len - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def load_window_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load window CSV and parse fields."""
    windows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            window = dict(row)
            # Parse numeric fields
            window["start_sec"] = float(row["start_sec"])
            window["end_sec"] = float(row["end_sec"])
            window["fps"] = float(row["fps"])
            window["duration_sec"] = float(row["duration_sec"])
            window["primary_label"] = int(row["primary_label"])
            window["is_background"] = int(row["is_background"]) == 1
            # Parse labels JSON
            window["labels"] = json.loads(row["labels"])
            windows.append(window)
    return windows


def load_gt_segments(
    splits_root: Path,
    task: str,
    label_map: Dict[str, int],
) -> Dict[str, List[Dict]]:
    """Load GT segments from all split CSVs."""
    # Determine directory
    if task == "4class":
        split_dirs = ["cv_splits_4class", "single_split_4class"]
    else:
        split_dirs = ["cv_splits", "single_split"]
    
    segments_by_video: Dict[str, List[Dict]] = defaultdict(list)
    seen_ids: Set[str] = set()
    
    for split_dir in split_dirs:
        split_path = splits_root / split_dir
        if not split_path.exists():
            continue
        
        for csv_path in split_path.glob("*.csv"):
            if csv_path.suffix != ".csv":
                continue
            
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    segment_id = row.get("segment_id", "")
                    if segment_id in seen_ids:
                        continue
                    seen_ids.add(segment_id)
                    
                    video_file = row.get("video_file", "")
                    video_key = normalize_video_file(video_file)
                    label_name = row.get("rmm_type", "")
                    
                    if label_name not in label_map:
                        continue
                    
                    try:
                        start_sec = float(row.get("start_sec", 0))
                        end_sec = float(row.get("end_sec", 0))
                    except ValueError:
                        continue
                    
                    segments_by_video[video_key].append({
                        "segment_id": segment_id,
                        "label": label_name,
                        "label_idx": label_map[label_name],
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "start_exclusive": start_sec,
                        "end_exclusive": end_sec + 1.0,  # Half-open conversion
                    })
    
    return dict(segments_by_video)


def check_leakage(
    tal_dir: Path,
    task: str,
    mode: str,
) -> Tuple[bool, List[str]]:
    """
    Check for data leakage - same video in incompatible splits.
    
    Returns (has_leakage, issues).
    """
    splits_dir = tal_dir / f"splits_{mode}_{task}"
    if not splits_dir.exists():
        return False, [f"Splits directory not found: {splits_dir}"]
    
    # Collect video_key -> set of split names
    video_splits: Dict[str, Set[str]] = defaultdict(set)
    
    for csv_path in splits_dir.glob("*_windows.csv"):
        split_name = csv_path.stem.replace("_windows", "")
        windows = load_window_csv(csv_path)
        
        for window in windows:
            video_key = window.get("video_key", "")
            if video_key:
                video_splits[video_key].add(split_name)
    
    issues = []
    has_leakage = False
    
    for video_key, splits in video_splits.items():
        if mode == "cv":
            # Group by fold
            fold_assignments = defaultdict(set)
            for split in splits:
                if "_train" in split or "_val" in split:
                    fold = split.rsplit("_", 1)[0]
                    split_type = split.rsplit("_", 1)[-1]
                    fold_assignments[fold].add(split_type)
            
            for fold, types in fold_assignments.items():
                if "train" in types and "val" in types:
                    issues.append(f"LEAKAGE: {video_key} in both {fold}_train and {fold}_val")
                    has_leakage = True
        else:
            # Single mode
            split_types = {s for s in splits if s in ("train", "val", "test")}
            if len(split_types) > 1:
                issues.append(f"LEAKAGE: {video_key} in multiple splits: {split_types}")
                has_leakage = True
    
    return has_leakage, issues


def check_coverage(
    tal_dir: Path,
    task: str,
    mode: str,
    window_len: float,
    stride: float,
) -> Tuple[bool, List[str]]:
    """
    Check that windows cover each video's timeline appropriately.
    
    Note: For CV mode, the same video can appear in multiple splits (e.g., train for fold 0,
    val for fold 1). We check coverage per-CSV-file, not across all files.
    
    Returns (all_ok, issues).
    """
    splits_dir = tal_dir / f"splits_{mode}_{task}"
    if not splits_dir.exists():
        return False, [f"Splits directory not found: {splits_dir}"]
    
    issues = []
    all_ok = True
    
    # Check each CSV file independently
    for csv_path in splits_dir.glob("*_windows.csv"):
        split_name = csv_path.stem
        windows = load_window_csv(csv_path)
        
        # Group windows by video within this split file
        windows_by_video: Dict[str, List[Dict]] = defaultdict(list)
        for window in windows:
            video_key = window.get("video_key", "")
            windows_by_video[video_key].append(window)
        
        for video_key, vid_windows in windows_by_video.items():
            if not vid_windows:
                continue
            
            duration = vid_windows[0]["duration_sec"]
            
            # Sort by start time
            vid_windows = sorted(vid_windows, key=lambda w: w["start_sec"])
            
            # Check first window starts at 0
            if vid_windows[0]["start_sec"] > 0.01:
                issues.append(f"[{split_name}] Coverage gap at start: {video_key} first window starts at {vid_windows[0]['start_sec']:.2f}s")
                all_ok = False
            
            # Check last window ends near duration
            last_end = vid_windows[-1]["end_sec"]
            expected_last_end = duration
            # Allow some tolerance for rounding
            if last_end < expected_last_end - window_len - 0.01:
                issues.append(f"[{split_name}] Coverage gap at end: {video_key} last window ends at {last_end:.2f}s, duration is {duration:.2f}s")
                all_ok = False
            
            # Check for gaps between windows
            for i in range(1, len(vid_windows)):
                prev_start = vid_windows[i-1]["start_sec"]
                curr_start = vid_windows[i]["start_sec"]
                expected_start = prev_start + stride
                
                if abs(curr_start - expected_start) > 0.01:
                    issues.append(f"[{split_name}] Unexpected stride: {video_key} window {i} starts at {curr_start:.2f}s, expected {expected_start:.2f}s")
                    all_ok = False
                    break  # Only report first issue per video
    
    return all_ok, issues


def verify_window_labels(
    window: Dict,
    gt_segments: List[Dict],
    tiou_thresh: float,
    label_map: Dict[str, int],
) -> Tuple[bool, str]:
    """
    Verify that a window's labels match expected labels from GT segments.
    
    Returns (matches, explanation).
    """
    w_start = window["start_sec"]
    w_end = window["end_sec"]
    expected_labels = window["labels"]
    
    # Compute expected labels from GT
    num_classes = len(label_map)
    class_max_tiou = {i: 0.0 for i in range(num_classes)}
    
    for gt in gt_segments:
        tiou = compute_tiou(w_start, w_end, gt["start_exclusive"], gt["end_exclusive"])
        if tiou > class_max_tiou[gt["label_idx"]]:
            class_max_tiou[gt["label_idx"]] = tiou
    
    computed_labels = []
    for class_idx, max_tiou in class_max_tiou.items():
        if max_tiou >= tiou_thresh:
            computed_labels.append(class_idx)
    computed_labels = sorted(computed_labels)
    
    if computed_labels == expected_labels:
        return True, "Labels match"
    else:
        idx_to_name = {v: k for k, v in label_map.items()}
        expected_names = [idx_to_name.get(i, f"class_{i}") for i in expected_labels]
        computed_names = [idx_to_name.get(i, f"class_{i}") for i in computed_labels]
        return False, f"Mismatch: expected {expected_names}, computed {computed_names}"


def sample_and_verify_labels(
    tal_dir: Path,
    splits_root: Path,
    task: str,
    mode: str,
    tiou_thresh: float,
    num_samples: int = 20,
) -> Tuple[int, int, List[str]]:
    """
    Sample windows and verify their labels match GT.
    
    Returns (num_correct, num_total, issues).
    """
    # Load label map
    if task == "4class":
        label_map = {"hands flapping": 0, "jumping": 1, "rocking": 2, "spinning": 3}
    else:
        label_map = {"hands flapping": 0, "jumping": 1, "one hand flap": 2, "rocking": 3, "spinning": 4}
    
    # Load GT segments
    gt_segments = load_gt_segments(splits_root, task, label_map)
    
    # Collect all windows
    splits_dir = tal_dir / f"splits_{mode}_{task}"
    all_windows = []
    
    for csv_path in splits_dir.glob("*_windows.csv"):
        windows = load_window_csv(csv_path)
        all_windows.extend(windows)
    
    if not all_windows:
        return 0, 0, ["No windows found"]
    
    # Sample windows (prefer some positive ones)
    positive_windows = [w for w in all_windows if not w["is_background"]]
    background_windows = [w for w in all_windows if w["is_background"]]
    
    # Sample half positive, half background (or all if not enough)
    n_pos = min(num_samples // 2, len(positive_windows))
    n_bg = min(num_samples - n_pos, len(background_windows))
    
    random.seed(42)  # Reproducible sampling
    sampled = random.sample(positive_windows, n_pos) + random.sample(background_windows, n_bg)
    
    correct = 0
    issues = []
    
    for window in sampled:
        video_key = window["video_key"]
        video_gt = gt_segments.get(video_key, [])
        
        matches, explanation = verify_window_labels(window, video_gt, tiou_thresh, label_map)
        
        if matches:
            correct += 1
        else:
            issues.append(f"Window {window['window_id']}: {explanation}")
    
    return correct, len(sampled), issues


def print_summary(
    tal_dir: Path,
    task: str,
    mode: str,
):
    """Print summary statistics from metadata."""
    metadata_path = tal_dir / f"metadata_{task}_{mode}.json"
    
    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}")
        return
    
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"TAL Split Summary: {task} / {mode}")
    print(f"{'='*60}")
    print(f"Generated at: {metadata.get('generated_at', 'unknown')}")
    print(f"Window: {metadata.get('window_len_sec', '?')}s, Stride: {metadata.get('stride_sec', '?')}s")
    print(f"tIoU threshold: {metadata.get('tiou_thresh', '?')}")
    print(f"Time semantics: {metadata.get('time_semantics', 'unknown')}")
    print()
    
    stats = metadata.get("split_statistics", {})
    for split_name, split_stats in stats.items():
        print(f"  {split_name}:")
        print(f"    Total windows: {split_stats.get('total_windows', '?')}")
        print(f"    Background: {split_stats.get('background_pct', '?')}%")
        print(f"    Multi-label: {split_stats.get('multi_label_windows', '?')}")
        print(f"    Class counts: {split_stats.get('class_counts', {})}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate TAL window splits."
    )
    parser.add_argument(
        "--tal-dir",
        type=Path,
        default=_REPO_ROOT / "dataprep/tal",
        help="TAL output directory.",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=_REPO_ROOT / "dataprep/splits",
        help="Original segment splits directory.",
    )
    parser.add_argument(
        "--task",
        choices=["4class", "5class"],
        required=True,
        help="Task variant.",
    )
    parser.add_argument(
        "--mode",
        choices=["cv", "single"],
        required=True,
        help="Split mode.",
    )
    parser.add_argument(
        "--tiou-thresh",
        type=float,
        default=0.3,
        help="tIoU threshold (must match generation).",
    )
    parser.add_argument(
        "--window-len",
        type=float,
        default=2.0,
        help="Window length (must match generation).",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=1.0,
        help="Window stride (must match generation).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of windows to sample for label verification.",
    )
    args = parser.parse_args()
    
    print(f"Validating TAL splits: {args.task} / {args.mode}")
    
    # Print summary
    print_summary(args.tal_dir, args.task, args.mode)
    
    # Check leakage
    print("\n--- Leakage Check ---")
    has_leakage, leakage_issues = check_leakage(args.tal_dir, args.task, args.mode)
    if has_leakage:
        print("FAILED: Data leakage detected!")
        for issue in leakage_issues[:5]:
            print(f"  {issue}")
    else:
        print("PASSED: No data leakage")
    
    # Check coverage
    print("\n--- Coverage Check ---")
    coverage_ok, coverage_issues = check_coverage(
        args.tal_dir, args.task, args.mode, args.window_len, args.stride
    )
    if coverage_ok:
        print("PASSED: Window coverage looks good")
    else:
        print("ISSUES: Coverage problems detected")
        for issue in coverage_issues[:5]:
            print(f"  {issue}")
        if len(coverage_issues) > 5:
            print(f"  ... and {len(coverage_issues) - 5} more issues")
    
    # Sample and verify labels
    print("\n--- Label Verification ---")
    correct, total, label_issues = sample_and_verify_labels(
        args.tal_dir, args.splits_root, args.task, args.mode,
        args.tiou_thresh, args.num_samples
    )
    print(f"Sampled {total} windows: {correct}/{total} labels verified correct")
    if label_issues:
        print("Issues:")
        for issue in label_issues[:5]:
            print(f"  {issue}")
    else:
        print("PASSED: All sampled labels correct")
    
    # Final verdict
    print("\n" + "="*60)
    if not has_leakage and coverage_ok and correct == total:
        print("✓ All validation checks PASSED")
    else:
        print("✗ Some validation checks FAILED")
    print("="*60)


if __name__ == "__main__":
    main()

