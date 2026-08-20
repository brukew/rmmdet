#!/usr/bin/env python3
"""
Generate video-to-split assignments for TAL (Temporal Action Localization).

This script reads existing segment-level split CSVs from actreg/dataprep/splits/
and derives video-level split assignments while preserving LCTO-safe groupings.

The key insight: in the existing splits, each segment belongs to a video,
and all segments from the same video should be in the same split (train/val/test).
This script extracts that implicit video-level assignment.

Usage:
    python make_tal_video_assignments.py --splits-root actreg/dataprep/splits/ \
        --task 4class --mode cv --output-dir actreg/dataprep/tal/video_assignment/
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Resolve paths relative to the repo root so this runs from any clone location.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())


def normalize_video_file(video_file: str) -> str:
    """
    Normalize video_file path for consistent matching.
    
    Handles variations like backslashes, leading slashes, etc.
    Returns lowercase for case-insensitive matching.
    """
    # Normalize path separators
    normalized = video_file.replace("\\", "/")
    # Remove leading slashes
    normalized = normalized.lstrip("/")
    # Remove common prefixes that might differ
    prefixes_to_remove = [
        "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/",
        "Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/",
    ]
    for prefix in prefixes_to_remove:
        if normalized.lower().startswith(prefix.lower()):
            normalized = normalized[len(prefix):]
    return normalized


def load_split_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Load a split CSV file and return list of row dicts."""
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_video_info_from_row(row: Dict[str, str]) -> Dict[str, str]:
    """
    Extract video-level information from a segment row.
    
    Returns dict with normalized video_file as key identifier plus metadata.
    """
    video_file = row.get("video_file", "")
    filename = row.get("filename", "")
    child_id = row.get("child_id", "")
    lcto_group = row.get("lcto_group", "")
    timepoint = row.get("timepoint", "")
    video_duration = row.get("video_duration", "")
    
    return {
        "video_file": video_file,
        "video_file_normalized": normalize_video_file(video_file),
        "filename": filename,
        "child_id": child_id,
        "lcto_group": lcto_group,
        "timepoint": timepoint,
        "video_duration": video_duration,
    }


def derive_video_assignments(
    splits_root: Path,
    task: str,  # "4class" or "5class"
    mode: str,  # "cv" or "single"
) -> Tuple[Dict[str, Dict], Dict[str, Set[str]], List[str]]:
    """
    Derive video-to-split assignments from existing segment splits.
    
    Returns:
        video_info: dict[video_file_normalized] -> video metadata
        video_splits: dict[video_file_normalized] -> set of split names
        issues: list of warning messages
    """
    # Determine which directory to read
    if task == "4class":
        split_dir = "cv_splits_4class" if mode == "cv" else "single_split_4class"
    else:
        split_dir = "cv_splits" if mode == "cv" else "single_split"
    
    split_path = splits_root / split_dir
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_path}")
    
    # Find all CSV files
    if mode == "cv":
        csv_files = sorted(split_path.glob("fold_*_*.csv"))
    else:
        csv_files = sorted(split_path.glob("*.csv"))
        # Filter to train/val/test only
        csv_files = [f for f in csv_files if f.stem in ("train", "val", "test")]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {split_path}")
    
    video_info: Dict[str, Dict] = {}
    video_splits: Dict[str, Set[str]] = defaultdict(set)
    issues: List[str] = []
    
    for csv_path in csv_files:
        # Determine split name from filename
        if mode == "cv":
            # e.g., fold_0_train.csv -> fold_0_train
            split_name = csv_path.stem
        else:
            # e.g., train.csv -> train
            split_name = csv_path.stem
        
        rows = load_split_csv(csv_path)
        print(f"  Loading {csv_path.name}: {len(rows)} segments")
        
        for row in rows:
            info = extract_video_info_from_row(row)
            video_key = info["video_file_normalized"]
            
            if not video_key:
                issues.append(f"Empty video_file in {csv_path.name}: {row.get('segment_id', '?')}")
                continue
            
            # Store video info (first occurrence wins for metadata)
            if video_key not in video_info:
                video_info[video_key] = info
            
            # Track which splits this video appears in
            video_splits[video_key].add(split_name)
    
    return video_info, dict(video_splits), issues


def check_leakage(
    video_splits: Dict[str, Set[str]],
    mode: str,
) -> Tuple[bool, List[str]]:
    """
    Check for data leakage - same video appearing in incompatible splits.
    
    For CV mode: a video should only be in train OR val for each fold.
    For single mode: a video should only be in one of train/val/test.
    
    Returns (has_leakage, issue_messages).
    """
    issues = []
    has_leakage = False
    
    for video_key, splits in video_splits.items():
        if mode == "cv":
            # Group splits by fold
            fold_assignments = defaultdict(set)
            for split in splits:
                # e.g., fold_0_train -> fold_0, train
                parts = split.rsplit("_", 1)
                if len(parts) == 2:
                    fold_assignments[parts[0]].add(parts[1])
            
            # Check each fold
            for fold, split_types in fold_assignments.items():
                if "train" in split_types and "val" in split_types:
                    issues.append(f"LEAKAGE: {video_key} appears in both {fold}_train and {fold}_val")
                    has_leakage = True
        else:
            # Single mode: should only be in one split
            split_types = set()
            for split in splits:
                if split in ("train", "val", "test"):
                    split_types.add(split)
            
            if len(split_types) > 1:
                issues.append(f"LEAKAGE: {video_key} appears in multiple splits: {split_types}")
                has_leakage = True
    
    return has_leakage, issues


def write_video_assignment_csv(
    video_info: Dict[str, Dict],
    video_splits: Dict[str, Set[str]],
    output_path: Path,
    mode: str,
):
    """Write video assignment CSV with one row per video."""
    fieldnames = [
        "video_key",
        "video_file",
        "filename",
        "child_id",
        "lcto_group",
        "timepoint",
        "video_duration",
    ]
    
    if mode == "cv":
        # Add columns for each fold
        for fold in range(3):
            fieldnames.append(f"fold_{fold}_split")
    else:
        fieldnames.append("split")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for video_key in sorted(video_info.keys()):
            info = video_info[video_key]
            splits = video_splits.get(video_key, set())
            
            row = {
                "video_key": video_key,
                "video_file": info["video_file"],
                "filename": info["filename"],
                "child_id": info["child_id"],
                "lcto_group": info["lcto_group"],
                "timepoint": info["timepoint"],
                "video_duration": info["video_duration"],
            }
            
            if mode == "cv":
                # Determine split for each fold
                for fold in range(3):
                    fold_splits = [s for s in splits if s.startswith(f"fold_{fold}_")]
                    if fold_splits:
                        # Extract train/val from fold_X_train/fold_X_val
                        split_type = fold_splits[0].rsplit("_", 1)[-1]
                        row[f"fold_{fold}_split"] = split_type
                    else:
                        row[f"fold_{fold}_split"] = ""
            else:
                # Single mode: just the split name
                split_list = [s for s in splits if s in ("train", "val", "test")]
                row["split"] = split_list[0] if split_list else ""
            
            writer.writerow(row)
    
    print(f"Wrote {len(video_info)} videos to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate video-to-split assignments for TAL from existing segment splits."
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=_REPO_ROOT / "dataprep/splits",
        help="Root directory containing existing segment split CSVs.",
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
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "dataprep/tal/video_assignment",
        help="Output directory for video assignment CSV.",
    )
    parser.add_argument(
        "--fail-on-leakage",
        action="store_true",
        help="Exit with error if data leakage is detected.",
    )
    args = parser.parse_args()
    
    print(f"Deriving video assignments from {args.splits_root}")
    print(f"Task: {args.task}, Mode: {args.mode}")
    
    # Derive assignments
    video_info, video_splits, load_issues = derive_video_assignments(
        args.splits_root, args.task, args.mode
    )
    
    if load_issues:
        print(f"\nWarnings during loading:")
        for issue in load_issues[:10]:
            print(f"  - {issue}")
        if len(load_issues) > 10:
            print(f"  ... and {len(load_issues) - 10} more")
    
    # Check for leakage
    has_leakage, leakage_issues = check_leakage(video_splits, args.mode)
    
    if leakage_issues:
        print(f"\nLeakage check results:")
        for issue in leakage_issues:
            print(f"  - {issue}")
    
    if has_leakage and args.fail_on_leakage:
        raise SystemExit("Data leakage detected. Aborting.")
    
    if not has_leakage:
        print("\n✓ No data leakage detected")
    
    # Write output
    output_filename = f"video_to_split_{args.task}_{args.mode}.csv"
    output_path = args.output_dir / output_filename
    
    write_video_assignment_csv(video_info, video_splits, output_path, args.mode)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total unique videos: {len(video_info)}")
    
    if args.mode == "cv":
        for fold in range(3):
            train_count = sum(1 for v, s in video_splits.items() 
                           if f"fold_{fold}_train" in s)
            val_count = sum(1 for v, s in video_splits.items() 
                          if f"fold_{fold}_val" in s)
            print(f"  Fold {fold}: {train_count} train, {val_count} val videos")
    else:
        train_count = sum(1 for v, s in video_splits.items() if "train" in s)
        val_count = sum(1 for v, s in video_splits.items() if "val" in s)
        test_count = sum(1 for v, s in video_splits.items() if "test" in s)
        print(f"  Train: {train_count}, Val: {val_count}, Test: {test_count} videos")


if __name__ == "__main__":
    main()


