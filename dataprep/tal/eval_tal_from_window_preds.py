#!/usr/bin/env python3
"""
End-to-end TAL evaluation from window-level predictions.

This CLI script:
1. Loads window-level predictions (CSV with score_class0..N columns)
2. Postprocesses into predicted segments (threshold, smooth, merge)
3. Loads GT segments from split CSVs
4. Computes mAP at tIoU thresholds {0.3, 0.5, 0.7}
5. Writes outputs: pred_segments.csv, metrics.json, per_class_ap.csv, report.txt

Example usage:
    # Evaluate V-JEPA predictions
    python eval_tal_from_window_preds.py \
        --window-preds /path/to/window_level_preds.csv \
        --splits-root /path/to/actreg/dataprep/splits \
        --task 4class \
        --out-dir /path/to/eval_output
    
    # With custom postprocessing parameters
    python eval_tal_from_window_preds.py \
        --window-preds predictions.csv \
        --out-dir output \
        --smooth-k 5 \
        --thr 0.4 \
        --merge-gap-sec 1.5

Standardized window prediction CSV schema (required columns):
    - window_id, video_key, start_sec, end_sec
    - score_class0, score_class1, score_class2, score_class3, [score_class4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Import our modules
from window_to_segments import (
    PostprocessParams,
    parse_threshold_per_class,
    window_scores_to_segments,
)
from tal_map_eval import (
    ID2LABEL_4CLASS,
    ID2LABEL_5CLASS,
    compute_map,
    format_metrics_report,
    load_gt_segments,
    save_metrics,
)


# Resolve paths relative to the repo root so this runs from any clone location.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TAL from window-level predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    python eval_tal_from_window_preds.py \\
        --window-preds window_level_preds.csv \\
        --out-dir eval_output

    # With custom thresholds
    python eval_tal_from_window_preds.py \\
        --window-preds preds.csv \\
        --out-dir output \\
        --thr 0.4 \\
        --thr-per-class "0:0.5,1:0.4,2:0.3,3:0.6"
        """,
    )
    
    # Input/Output
    parser.add_argument(
        "--window-preds",
        type=Path,
        required=True,
        help="Path to window-level predictions CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for evaluation results.",
    )
    
    # GT loading
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=_REPO_ROOT / "dataprep/splits",
        help="Root directory containing split CSVs (default: actreg/dataprep/splits).",
    )
    parser.add_argument(
        "--task",
        choices=["4class", "5class"],
        default="4class",
        help="Task type (default: 4class).",
    )
    parser.add_argument(
        "--mode",
        choices=["cv", "single"],
        default="cv",
        help="Split mode for GT loading (default: cv).",
    )
    
    # Evaluation thresholds
    parser.add_argument(
        "--tiou-thresholds",
        type=str,
        default="0.3,0.5,0.7",
        help="Comma-separated tIoU thresholds (default: 0.3,0.5,0.7).",
    )
    
    # Postprocessing parameters
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
    
    # Misc
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # 1. Load window predictions
    # =========================================================================
    if not args.quiet:
        print("=" * 60)
        print("TAL Evaluation from Window Predictions")
        print("=" * 60)
        print(f"\nLoading window predictions from: {args.window_preds}")
    
    df = pd.read_csv(args.window_preds)
    
    if not args.quiet:
        print(f"  Loaded {len(df)} windows from {df['video_key'].nunique()} videos")
    
    # Validate required columns
    required_cols = ["video_key", "start_sec", "end_sec"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for score columns
    score_cols = [c for c in df.columns if c.startswith("score_class")]
    if not score_cols:
        raise ValueError("No score_class* columns found in predictions CSV")
    
    if not args.quiet:
        print(f"  Score columns found: {score_cols}")
    
    # =========================================================================
    # 2. Set up postprocessing parameters
    # =========================================================================
    threshold_per_class = None
    if args.thr_per_class:
        threshold_per_class = parse_threshold_per_class(args.thr_per_class)
    
    # Determine class IDs based on task
    if args.task == "4class":
        class_ids = [0, 1, 2, 3]
        id2label = ID2LABEL_4CLASS
    else:
        class_ids = [0, 1, 2, 3, 4]
        id2label = ID2LABEL_5CLASS
    
    params = PostprocessParams(
        smooth_k=args.smooth_k,
        threshold=args.thr,
        threshold_per_class=threshold_per_class,
        merge_gap_sec=args.merge_gap_sec,
        min_duration_sec=args.min_duration_sec,
        score_reducer=args.score_reducer,
        class_ids=class_ids,
    )
    
    if not args.quiet:
        print(f"\nPostprocessing parameters:")
        print(f"  smooth_k: {params.smooth_k}")
        print(f"  threshold: {params.threshold}")
        if threshold_per_class:
            print(f"  threshold_per_class: {threshold_per_class}")
        print(f"  merge_gap_sec: {params.merge_gap_sec}")
        print(f"  min_duration_sec: {params.min_duration_sec}")
        print(f"  score_reducer: {params.score_reducer}")
        print(f"  class_ids: {params.class_ids}")
    
    # =========================================================================
    # 3. Postprocess: windows -> segments
    # =========================================================================
    if not args.quiet:
        print(f"\nPostprocessing windows to segments...")
    
    segments_df = window_scores_to_segments(df, params)
    
    if not args.quiet:
        print(f"  Generated {len(segments_df)} predicted segments")
        if not segments_df.empty:
            for class_id in class_ids:
                class_segs = segments_df[segments_df["class_id"] == class_id]
                label = id2label.get(class_id, f"class_{class_id}")
                print(f"    {label}: {len(class_segs)} segments")
    
    # =========================================================================
    # 4. Load GT segments
    # =========================================================================
    if not args.quiet:
        print(f"\nLoading GT segments from: {args.splits_root}")
        print(f"  Task: {args.task}, Mode: {args.mode}")
    
    gt_by_video = load_gt_segments(args.splits_root, args.task, args.mode)
    
    if not args.quiet:
        n_gt_total = sum(len(segs) for segs in gt_by_video.values())
        print(f"  Loaded {n_gt_total} GT segments from {len(gt_by_video)} videos")
    
    # =========================================================================
    # 5. Compute mAP
    # =========================================================================
    tiou_thresholds = [float(x.strip()) for x in args.tiou_thresholds.split(",")]
    
    if not args.quiet:
        print(f"\nComputing mAP at tIoU thresholds: {tiou_thresholds}")
    
    metrics = compute_map(segments_df, gt_by_video, class_ids, tiou_thresholds)
    
    # =========================================================================
    # 6. Print report
    # =========================================================================
    report = format_metrics_report(metrics, class_ids, id2label)
    print("\n" + report)
    
    # =========================================================================
    # 7. Save outputs
    # =========================================================================
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predicted segments
    seg_path = args.out_dir / "pred_segments.csv"
    segments_df.to_csv(seg_path, index=False)
    if not args.quiet:
        print(f"\nPredicted segments saved to: {seg_path}")
    
    # Save metrics (JSON, per_class_ap.csv, report.txt)
    save_metrics(metrics, args.out_dir, class_ids, id2label)
    if not args.quiet:
        print(f"Metrics saved to: {args.out_dir}")
    
    # Save evaluation config for reproducibility
    config = {
        "window_preds": str(args.window_preds),
        "splits_root": str(args.splits_root),
        "task": args.task,
        "mode": args.mode,
        "tiou_thresholds": tiou_thresholds,
        "postprocess": {
            "smooth_k": params.smooth_k,
            "threshold": params.threshold,
            "threshold_per_class": params.threshold_per_class,
            "merge_gap_sec": params.merge_gap_sec,
            "min_duration_sec": params.min_duration_sec,
            "score_reducer": params.score_reducer,
            "class_ids": params.class_ids,
        },
    }
    config_path = args.out_dir / "eval_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    if not args.quiet:
        print(f"Config saved to: {config_path}")
        print("\nDone!")
    
    return metrics


if __name__ == "__main__":
    main()

