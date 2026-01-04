#!/usr/bin/env python3
"""
Convert window-level class scores to predicted segments for TAL evaluation.

This module takes per-window softmax/logit scores and produces temporal segments
by thresholding, smoothing, and merging adjacent positive windows.

Example usage:
    python window_to_segments.py \
        --window-preds window_level_preds.csv \
        --out-csv pred_segments.csv \
        --smooth-k 3 \
        --thr 0.5 \
        --merge-gap-sec 1.0

The output schema is:
    video_key, class_id, start_sec, end_sec, score, n_windows
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PostprocessParams:
    """Parameters for window-to-segment postprocessing."""
    
    smooth_k: int = 3
    """Number of windows for moving average smoothing (1 = no smoothing)."""
    
    threshold: float = 0.5
    """Global threshold for binary activation."""
    
    threshold_per_class: Optional[Dict[int, float]] = None
    """Per-class thresholds (overrides global if provided)."""
    
    merge_gap_sec: float = 1.0
    """Maximum gap (seconds) between segments to merge."""
    
    min_duration_sec: float = 0.0
    """Minimum segment duration; shorter segments are discarded."""
    
    score_reducer: str = "max"
    """How to aggregate window scores into segment score: 'max' or 'mean'."""
    
    class_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    """Which class IDs to process (default: RMM classes 0-3, excluding background)."""

    def get_threshold(self, class_id: int) -> float:
        """Get threshold for a specific class."""
        if self.threshold_per_class and class_id in self.threshold_per_class:
            return self.threshold_per_class[class_id]
        return self.threshold


@dataclass
class PredictedSegment:
    """A predicted temporal segment."""
    video_key: str
    class_id: int
    start_sec: float
    end_sec: float
    score: float
    n_windows: int


def smooth_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Apply moving average smoothing to a 1D score array.
    
    Args:
        scores: 1D array of scores (sorted by time).
        k: Window size for moving average.
    
    Returns:
        Smoothed scores (same length).
    """
    if k <= 1 or len(scores) == 0:
        return scores
    
    # Use uniform filter (moving average)
    kernel = np.ones(k) / k
    # Pad to handle edges
    padded = np.pad(scores, (k // 2, k - 1 - k // 2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(scores)]


def find_contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find contiguous runs of True values in a boolean array.
    
    Args:
        mask: 1D boolean array.
    
    Returns:
        List of (start_idx, end_idx) tuples (end is exclusive).
    """
    if len(mask) == 0:
        return []
    
    runs = []
    in_run = False
    start_idx = 0
    
    for i, val in enumerate(mask):
        if val and not in_run:
            # Start of a run
            start_idx = i
            in_run = True
        elif not val and in_run:
            # End of a run
            runs.append((start_idx, i))
            in_run = False
    
    # Handle run that extends to the end
    if in_run:
        runs.append((start_idx, len(mask)))
    
    return runs


def merge_close_segments(
    segments: List[Tuple[float, float, float, int]],
    gap_sec: float,
    score_reducer: str = "max",
) -> List[Tuple[float, float, float, int]]:
    """
    Merge segments that are within gap_sec of each other.
    
    Args:
        segments: List of (start_sec, end_sec, score, n_windows) tuples, sorted by start.
        gap_sec: Maximum gap to merge.
        score_reducer: How to combine scores ('max' or 'mean').
    
    Returns:
        Merged segments.
    """
    if len(segments) <= 1:
        return segments
    
    merged = []
    current_start, current_end, current_scores, current_n = segments[0][:2], segments[0][1], [segments[0][2]], segments[0][3]
    current_start = segments[0][0]
    current_end = segments[0][1]
    
    for start, end, score, n in segments[1:]:
        if start - current_end <= gap_sec:
            # Merge with current segment
            current_end = max(current_end, end)
            current_scores.append(score)
            current_n += n
        else:
            # Finalize current segment
            if score_reducer == "max":
                final_score = max(current_scores)
            else:
                final_score = sum(current_scores) / len(current_scores)
            merged.append((current_start, current_end, final_score, current_n))
            # Start new segment
            current_start = start
            current_end = end
            current_scores = [score]
            current_n = n
    
    # Finalize last segment
    if score_reducer == "max":
        final_score = max(current_scores)
    else:
        final_score = sum(current_scores) / len(current_scores)
    merged.append((current_start, current_end, final_score, current_n))
    
    return merged


def windows_to_segments_for_class(
    windows_df: pd.DataFrame,
    class_id: int,
    params: PostprocessParams,
) -> List[PredictedSegment]:
    """
    Convert windows for a single video/class into segments.
    
    Args:
        windows_df: DataFrame with columns start_sec, end_sec, score_class{class_id}.
                    Must be sorted by start_sec.
        class_id: Which class to process.
        params: Postprocessing parameters.
    
    Returns:
        List of PredictedSegment objects.
    """
    if windows_df.empty:
        return []
    
    score_col = f"score_class{class_id}"
    if score_col not in windows_df.columns:
        return []
    
    video_key = windows_df["video_key"].iloc[0]
    scores = windows_df[score_col].values.astype(float)
    starts = windows_df["start_sec"].values.astype(float)
    ends = windows_df["end_sec"].values.astype(float)
    
    # 1. Smooth scores
    smoothed = smooth_scores(scores, params.smooth_k)
    
    # 2. Threshold
    threshold = params.get_threshold(class_id)
    active_mask = smoothed >= threshold
    
    # 3. Find contiguous runs
    runs = find_contiguous_runs(active_mask)
    
    if not runs:
        return []
    
    # 4. Convert runs to segments
    raw_segments = []
    for run_start, run_end in runs:
        seg_start = starts[run_start]
        seg_end = ends[run_end - 1]  # run_end is exclusive index
        run_scores = smoothed[run_start:run_end]
        
        if params.score_reducer == "max":
            seg_score = float(np.max(run_scores))
        else:
            seg_score = float(np.mean(run_scores))
        
        n_windows = run_end - run_start
        raw_segments.append((seg_start, seg_end, seg_score, n_windows))
    
    # 5. Merge close segments
    merged = merge_close_segments(raw_segments, params.merge_gap_sec, params.score_reducer)
    
    # 6. Filter by minimum duration
    segments = []
    for seg_start, seg_end, seg_score, n_windows in merged:
        duration = seg_end - seg_start
        if duration >= params.min_duration_sec:
            segments.append(PredictedSegment(
                video_key=video_key,
                class_id=class_id,
                start_sec=seg_start,
                end_sec=seg_end,
                score=seg_score,
                n_windows=n_windows,
            ))
    
    return segments


def window_scores_to_segments(
    df: pd.DataFrame,
    params: Optional[PostprocessParams] = None,
) -> pd.DataFrame:
    """
    Convert window-level predictions to predicted segments.
    
    Args:
        df: DataFrame with columns:
            - video_key, start_sec, end_sec
            - score_class0, score_class1, score_class2, score_class3 (at minimum)
        params: Postprocessing parameters (uses defaults if None).
    
    Returns:
        DataFrame with columns:
            video_key, class_id, start_sec, end_sec, score, n_windows
    """
    if params is None:
        params = PostprocessParams()
    
    all_segments: List[PredictedSegment] = []
    
    # Group by video
    for video_key, video_df in df.groupby("video_key"):
        # Sort by start time
        video_df = video_df.sort_values("start_sec").reset_index(drop=True)
        
        # Process each class
        for class_id in params.class_ids:
            segments = windows_to_segments_for_class(video_df, class_id, params)
            all_segments.extend(segments)
    
    # Convert to DataFrame
    if not all_segments:
        return pd.DataFrame(columns=["video_key", "class_id", "start_sec", "end_sec", "score", "n_windows"])
    
    records = [
        {
            "video_key": seg.video_key,
            "class_id": seg.class_id,
            "start_sec": seg.start_sec,
            "end_sec": seg.end_sec,
            "score": seg.score,
            "n_windows": seg.n_windows,
        }
        for seg in all_segments
    ]
    
    result = pd.DataFrame(records)
    # Sort by video, class, start time
    result = result.sort_values(["video_key", "class_id", "start_sec"]).reset_index(drop=True)
    
    return result


def parse_threshold_per_class(s: str) -> Dict[int, float]:
    """
    Parse a per-class threshold string like '0:0.4,1:0.5,2:0.6,3:0.3'.
    
    Returns:
        Dict mapping class_id -> threshold.
    """
    result = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        class_id_str, thr_str = pair.split(":")
        result[int(class_id_str)] = float(thr_str)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert window-level predictions to temporal segments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--window-preds",
        type=Path,
        required=True,
        help="Path to window-level predictions CSV.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output path for predicted segments CSV.",
    )
    parser.add_argument(
        "--smooth-k",
        type=int,
        default=3,
        help="Moving average window size for score smoothing (default: 3).",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.5,
        help="Global threshold for binary activation (default: 0.5).",
    )
    parser.add_argument(
        "--thr-per-class",
        type=str,
        default=None,
        help="Per-class thresholds, e.g., '0:0.4,1:0.5,2:0.6,3:0.3'.",
    )
    parser.add_argument(
        "--merge-gap-sec",
        type=float,
        default=1.0,
        help="Maximum gap (seconds) between segments to merge (default: 1.0).",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=0.0,
        help="Minimum segment duration; shorter are discarded (default: 0.0).",
    )
    parser.add_argument(
        "--score-reducer",
        choices=["max", "mean"],
        default="max",
        help="How to aggregate window scores into segment score (default: max).",
    )
    parser.add_argument(
        "--class-ids",
        type=str,
        default="0,1,2,3",
        help="Comma-separated class IDs to process (default: 0,1,2,3).",
    )
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading window predictions from: {args.window_preds}")
    df = pd.read_csv(args.window_preds)
    print(f"  Loaded {len(df)} windows from {df['video_key'].nunique()} videos")
    
    # Parse parameters
    threshold_per_class = None
    if args.thr_per_class:
        threshold_per_class = parse_threshold_per_class(args.thr_per_class)
        print(f"  Per-class thresholds: {threshold_per_class}")
    
    class_ids = [int(x.strip()) for x in args.class_ids.split(",")]
    
    params = PostprocessParams(
        smooth_k=args.smooth_k,
        threshold=args.thr,
        threshold_per_class=threshold_per_class,
        merge_gap_sec=args.merge_gap_sec,
        min_duration_sec=args.min_duration_sec,
        score_reducer=args.score_reducer,
        class_ids=class_ids,
    )
    
    print(f"Postprocessing parameters:")
    print(f"  smooth_k: {params.smooth_k}")
    print(f"  threshold: {params.threshold}")
    print(f"  merge_gap_sec: {params.merge_gap_sec}")
    print(f"  min_duration_sec: {params.min_duration_sec}")
    print(f"  score_reducer: {params.score_reducer}")
    print(f"  class_ids: {params.class_ids}")
    
    # Convert to segments
    segments_df = window_scores_to_segments(df, params)
    
    print(f"\nGenerated {len(segments_df)} segments")
    if not segments_df.empty:
        for class_id in class_ids:
            class_segs = segments_df[segments_df["class_id"] == class_id]
            print(f"  Class {class_id}: {len(class_segs)} segments")
    
    # Save
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    segments_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved to: {args.out_csv}")


if __name__ == "__main__":
    main()

