#!/usr/bin/env python3
"""
Select 3 videos for TAL timeline visualization (Figure 5.3).

Criteria:
1. One video with exactly 1 RMM segment
2. One video with multiple segments of the same RMM class
3. One video with multiple different RMM classes

The selected videos should have ActionFormer predictions available,
prioritizing videos with higher-confidence predictions.

Usage:
    python select_videos_for_timeline.py
    python select_videos_for_timeline.py --output selected_videos.json
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Paths
SEGMENTS_FILE = ACTREG_ROOT / "dataprep" / "rmm_segments.csv"
ACTIONFORMER_RESULTS = ACTREG_ROOT / "tal" / "eval_results" / "actionformer_vjepa"
OUTPUT_DIR = ACTREG_ROOT / "insights" / "paper_plots"

# RMM types to consider (4-class, merging one_hand_flap into hands_flapping)
RMM_TYPES_4CLASS = {"hands flapping", "jumping", "rocking", "spinning"}
ONE_HAND_FLAP_MERGE = {"one hand flap": "hands flapping"}


def normalize_video_key(video_file: str) -> str:
    """Normalize video file path to match ActionFormer output format."""
    video_file = video_file.replace("\\", "/")
    parts = video_file.split("/")
    
    child_id = None
    for part in parts:
        if "AMES_" in part:
            child_id = part.split("AMES_")[-1]
            break
    
    if child_id is None:
        child_id = parts[0] if parts else "UNKNOWN"
    
    filename = parts[-1]
    video_stem = Path(filename).stem
    
    return f"{child_id}_{video_stem}"


def load_segments() -> pd.DataFrame:
    """Load RMM segments and filter to 4-class."""
    df = pd.read_csv(SEGMENTS_FILE)
    
    df["rmm_type_4class"] = df["rmm_type"].map(
        lambda x: ONE_HAND_FLAP_MERGE.get(x, x)
    )
    
    df = df[df["rmm_type_4class"].isin(RMM_TYPES_4CLASS)].copy()
    df["video_key_normalized"] = df["video_file"].apply(normalize_video_key)
    
    return df


def get_actionformer_videos(fold: int = 0) -> Dict[str, float]:
    """
    Get dict of video keys -> max prediction score for videos with ActionFormer predictions.
    Prioritizes videos with higher-confidence predictions.
    """
    pred_file = ACTIONFORMER_RESULTS / f"fold{fold}" / "predictions.json"
    
    if not pred_file.exists():
        print(f"Warning: ActionFormer predictions not found: {pred_file}")
        return {}
    
    with open(pred_file) as f:
        preds = json.load(f)
    
    # Handle nested structure (predictions under 'results' key)
    if "results" in preds:
        preds = preds["results"]
    
    # Return dict with max score for each video
    video_scores = {}
    for video_key, pred_list in preds.items():
        if pred_list:
            max_score = max(p["score"] for p in pred_list)
            video_scores[video_key] = max_score
        else:
            video_scores[video_key] = 0.0
    
    return video_scores


def analyze_videos(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze each video's segment composition."""
    video_stats = {}
    
    for video_key, group in df.groupby("video_key_normalized"):
        class_counts = group["rmm_type_4class"].value_counts().to_dict()
        segments = [
            (row["rmm_type_4class"], row["start_sec"], row["end_sec"])
            for _, row in group.iterrows()
        ]
        
        video_duration = None
        if "video_duration" in group.columns:
            video_duration = group["video_duration"].iloc[0]
        
        video_stats[video_key] = {
            "n_segments": len(group),
            "n_classes": len(class_counts),
            "class_counts": class_counts,
            "segments": segments,
            "video_file": group["video_file"].iloc[0],
            "video_duration": video_duration
        }
    
    return video_stats


def select_videos(video_stats: Dict[str, Dict], video_scores: Dict[str, float]) -> Dict[str, str]:
    """
    Select 3 videos meeting the criteria, prioritizing high-confidence predictions.
    
    Args:
        video_stats: Dict of video_key -> segment statistics
        video_scores: Dict of video_key -> max prediction score
    """
    selected = {}
    
    # Filter to videos with predictions and add score info
    available_stats = {}
    for k, v in video_stats.items():
        if k in video_scores:
            v_copy = v.copy()
            v_copy["max_pred_score"] = video_scores[k]
            available_stats[k] = v_copy
    
    if not available_stats:
        print("Warning: No videos available in ActionFormer predictions")
        for k, v in video_stats.items():
            v_copy = v.copy()
            v_copy["max_pred_score"] = 0.0
            available_stats[k] = v_copy
    
    # 1. Single segment video - prioritize by max prediction score
    single_segment_candidates = [
        (k, v) for k, v in available_stats.items()
        if v["n_segments"] == 1
    ]
    if single_segment_candidates:
        # Sort by max prediction score (descending)
        single_segment_candidates.sort(key=lambda x: -x[1].get("max_pred_score", 0))
        selected["single_segment"] = single_segment_candidates[0][0]
        print(f"Single segment: {selected['single_segment']}")
        print(f"  Max pred score: {available_stats[selected['single_segment']].get('max_pred_score', 0):.3f}")
        print(f"  Details: {available_stats[selected['single_segment']]}")
    
    # 2. Multiple segments of same class - prioritize by max prediction score
    multi_same_candidates = [
        (k, v) for k, v in available_stats.items()
        if v["n_classes"] == 1 and v["n_segments"] >= 2
    ]
    if multi_same_candidates:
        # Sort by max prediction score (descending)
        multi_same_candidates.sort(key=lambda x: -x[1].get("max_pred_score", 0))
        selected["multi_same_class"] = multi_same_candidates[0][0]
        print(f"Multi same class: {selected['multi_same_class']}")
        print(f"  Max pred score: {available_stats[selected['multi_same_class']].get('max_pred_score', 0):.3f}")
        print(f"  Details: {available_stats[selected['multi_same_class']]}")
    
    # 3. Multiple different classes - prioritize by max prediction score
    multi_class_candidates = [
        (k, v) for k, v in available_stats.items()
        if v["n_classes"] >= 2
    ]
    if multi_class_candidates:
        # Sort by max prediction score (descending)
        multi_class_candidates.sort(key=lambda x: -x[1].get("max_pred_score", 0))
        selected["multi_class"] = multi_class_candidates[0][0]
        print(f"Multi class: {selected['multi_class']}")
        print(f"  Max pred score: {available_stats[selected['multi_class']].get('max_pred_score', 0):.3f}")
        print(f"  Details: {available_stats[selected['multi_class']]}")
    
    return selected


def print_summary(video_stats: Dict[str, Dict], video_scores: Dict[str, float]):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("Video Analysis Summary")
    print("="*60)
    
    total_videos = len(video_stats)
    available = len(set(video_stats.keys()) & set(video_scores.keys()))
    
    print(f"\nTotal videos with RMM segments: {total_videos}")
    print(f"Videos with ActionFormer predictions: {available}")
    
    # Count videos with high-confidence predictions
    high_conf = sum(1 for k, s in video_scores.items() if k in video_stats and s >= 0.3)
    print(f"Videos with predictions score >= 0.3: {high_conf}")
    
    single = sum(1 for v in video_stats.values() if v["n_segments"] == 1)
    multi_same = sum(1 for v in video_stats.values() if v["n_classes"] == 1 and v["n_segments"] >= 2)
    multi_class = sum(1 for v in video_stats.values() if v["n_classes"] >= 2)
    
    print(f"\nBy segment pattern:")
    print(f"  - Single segment: {single}")
    print(f"  - Multiple same class: {multi_same}")
    print(f"  - Multiple classes: {multi_class}")


def save_selection(selected: Dict[str, str], video_stats: Dict[str, Dict], output_path: Path):
    """Save selected videos with full details."""
    output_data = {
        "selected_videos": {},
        "criteria": {
            "single_segment": "Exactly 1 RMM segment",
            "multi_same_class": "Multiple segments of same RMM class",
            "multi_class": "Multiple different RMM classes"
        }
    }
    
    for category, video_key in selected.items():
        if video_key:
            stats = video_stats[video_key]
            output_data["selected_videos"][category] = {
                "video_key": video_key,
                "video_file": stats["video_file"],
                "n_segments": stats["n_segments"],
                "n_classes": stats["n_classes"],
                "class_counts": stats["class_counts"],
                "segments": [
                    {"class": s[0], "start_sec": s[1], "end_sec": s[2]}
                    for s in stats["segments"]
                ]
            }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved selection to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Select 3 videos for TAL timeline visualization"
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fold", type=int, default=0)
    
    args = parser.parse_args()
    
    print("Loading RMM segments...")
    df = load_segments()
    print(f"Loaded {len(df)} segments from {df['video_key_normalized'].nunique()} videos")
    
    video_stats = analyze_videos(df)
    
    video_scores = get_actionformer_videos(args.fold)
    print(f"ActionFormer predictions available for {len(video_scores)} videos")
    
    print_summary(video_stats, video_scores)
    
    print("\n" + "="*60)
    print("Selected Videos")
    print("="*60)
    selected = select_videos(video_stats, video_scores)
    
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "selected_videos.json"
    save_selection(selected, video_stats, output_path)


if __name__ == "__main__":
    main()
