#!/usr/bin/env python3
"""
Run TAL evaluation across all CV folds and aggregate results.

This script:
1. Discovers available folds (0, 1, 2)
2. Finds prediction files for each model (V-JEPA, PoseC3D, STGCN++)
3. Runs evaluation on each fold
4. Aggregates results into a CV summary report (mean ± std)

Example usage:
    # Evaluate all available folds for V-JEPA
    python run_tal_eval_cv.py \
        --model vjepa \
        --preds-root /path/to/vjepa/runs/tal_cv \
        --out-dir /path/to/eval_results/vjepa_cv
    
    # Evaluate PoseC3D with custom postprocessing
    python run_tal_eval_cv.py \
        --model posec3d \
        --preds-root /path/to/pyskl/work_dirs/posec3d/tal \
        --out-dir eval_results/posec3d_cv \
        --thr 0.4

Supported models:
    - vjepa: V-JEPA2 (expects window_level_preds.csv in fold directories)
    - posec3d: PoseC3D (expects result.pkl from pyskl test.py)
    - stgcn: STGCN++ (expects result.pkl from pyskl test.py)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import our modules
from window_to_segments import PostprocessParams, window_scores_to_segments
from tal_map_eval import (
    ID2LABEL_4CLASS,
    ID2LABEL_5CLASS,
    compute_map,
    format_metrics_report,
    load_gt_segments,
    save_metrics,
)
from export_pyskl_window_preds import export_window_preds

# Resolve paths relative to the repo root so this runs from any clone location.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())


@dataclass
class FoldResult:
    """Results for a single fold."""
    fold_idx: int
    model: str
    metrics: Dict[str, Any]
    n_windows: int
    n_segments: int


def find_vjepa_preds(preds_root: Path, fold_idx: int) -> Optional[Path]:
    """Find V-JEPA prediction file for a fold."""
    # Common patterns
    patterns = [
        f"fold{fold_idx}/window_level_preds.csv",
        f"fold_{fold_idx}/window_level_preds.csv",
        f"tal_cv_fold{fold_idx}/window_level_preds.csv",
    ]
    
    for pattern in patterns:
        path = preds_root / pattern
        if path.exists():
            return path
    
    return None


def find_pyskl_preds(preds_root: Path, fold_idx: int, model: str) -> Optional[Tuple[Path, Path]]:
    """
    Find pyskl prediction files for a fold.
    
    Returns:
        Tuple of (scores_pkl, ann_pkl) or None if not found.
    """
    # Common patterns for work_dirs structure
    patterns = [
        (f"fold{fold_idx}/result.pkl", f"fold{fold_idx}"),
        (f"fold_{fold_idx}/result.pkl", f"fold_{fold_idx}"),
        (f"cv_4class_5class_bgsub/fold{fold_idx}/result.pkl", f"cv_4class_5class_bgsub/fold{fold_idx}"),
    ]
    
    for scores_pattern, work_dir in patterns:
        scores_path = preds_root / scores_pattern
        if scores_path.exists():
            return (scores_path, work_dir)
    
    return None


def load_or_export_pyskl_preds(
    preds_root: Path,
    fold_idx: int,
    model: str,
    ann_pkl_root: Path,
    window_csv_root: Path,
    cache_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    Load or export pyskl predictions for a fold.
    
    Args:
        preds_root: Root directory for pyskl work_dirs.
        fold_idx: Fold index.
        model: Model name (posec3d, stgcn).
        ann_pkl_root: Root for annotation pickle files.
        window_csv_root: Root for TAL window CSVs.
        cache_dir: Directory to cache exported CSVs.
    
    Returns:
        DataFrame with window predictions or None if not available.
    """
    result = find_pyskl_preds(preds_root, fold_idx, model)
    if result is None:
        return None
    
    scores_pkl, work_dir = result
    
    # Check for cached export
    cached_csv = cache_dir / f"{model}_fold{fold_idx}_window_preds.csv"
    if cached_csv.exists():
        print(f"  Loading cached predictions: {cached_csv}")
        return pd.read_csv(cached_csv)
    
    # Find annotation pickle
    ann_pkl = ann_pkl_root / f"fold{fold_idx}.pkl"
    if not ann_pkl.exists():
        print(f"  Warning: Annotation pickle not found: {ann_pkl}")
        return None
    
    # Find window CSV for metadata
    window_csv = window_csv_root / f"fold_{fold_idx}_val_windows.csv"
    if not window_csv.exists():
        print(f"  Warning: Window CSV not found: {window_csv}")
        window_csv = None
    
    # Export
    print(f"  Exporting predictions from: {scores_pkl}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        df = export_window_preds(
            ann_pkl=ann_pkl,
            scores_pkl=scores_pkl,
            out_csv=cached_csv,
            window_csv=window_csv,
            apply_softmax=False,  # pyskl typically outputs probs already
        )
        return df
    except Exception as e:
        print(f"  Error exporting predictions: {e}")
        return None


def evaluate_fold(
    preds_df: pd.DataFrame,
    splits_root: Path,
    task: str,
    mode: str,
    params: PostprocessParams,
    tiou_thresholds: List[float],
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate predictions for a single fold.
    
    Args:
        preds_df: Window predictions DataFrame.
        splits_root: Root for split CSVs.
        task: Task type (4class, 5class).
        mode: Split mode (cv, single).
        params: Postprocessing parameters.
        tiou_thresholds: tIoU thresholds.
        out_dir: Optional output directory.
    
    Returns:
        Metrics dictionary.
    """
    # Load GT
    gt_by_video = load_gt_segments(splits_root, task, mode)
    
    # Postprocess to segments
    segments_df = window_scores_to_segments(preds_df, params)
    
    # Determine class IDs
    if task == "4class":
        class_ids = [0, 1, 2, 3]
        id2label = ID2LABEL_4CLASS
    else:
        class_ids = [0, 1, 2, 3, 4]
        id2label = ID2LABEL_5CLASS
    
    # Compute metrics
    metrics = compute_map(segments_df, gt_by_video, class_ids, tiou_thresholds)
    
    # Add summary stats
    metrics["n_windows"] = len(preds_df)
    metrics["n_segments"] = len(segments_df)
    metrics["n_videos"] = preds_df["video_key"].nunique()
    
    # Save if requested
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        segments_df.to_csv(out_dir / "pred_segments.csv", index=False)
        save_metrics(metrics, out_dir, class_ids, id2label)
    
    return metrics


def aggregate_cv_results(
    fold_results: List[FoldResult],
    tiou_thresholds: List[float],
) -> Dict[str, Any]:
    """
    Aggregate results across CV folds.
    
    Args:
        fold_results: List of FoldResult objects.
        tiou_thresholds: tIoU thresholds.
    
    Returns:
        Aggregated metrics dictionary with mean and std.
    """
    if not fold_results:
        return {"error": "No fold results available"}
    
    # Collect values for each metric
    metrics_by_key = {}
    
    for fr in fold_results:
        for key, value in fr.metrics.items():
            if key.startswith("mAP@") or key == "avg_mAP":
                if key not in metrics_by_key:
                    metrics_by_key[key] = []
                if not np.isnan(value):
                    metrics_by_key[key].append(value)
    
    # Compute mean and std
    agg = {
        "n_folds": len(fold_results),
        "folds_evaluated": [fr.fold_idx for fr in fold_results],
        "model": fold_results[0].model,
    }
    
    for key, values in metrics_by_key.items():
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
            agg[f"{key}_values"] = values
        else:
            agg[f"{key}_mean"] = None
            agg[f"{key}_std"] = None
    
    # Total stats
    agg["total_windows"] = sum(fr.n_windows for fr in fold_results)
    agg["total_segments"] = sum(fr.n_segments for fr in fold_results)
    
    return agg


def format_cv_summary(agg: Dict[str, Any]) -> str:
    """Format CV summary as human-readable report."""
    lines = [
        "=" * 60,
        "Cross-Validation Summary",
        "=" * 60,
        f"Model: {agg.get('model', 'unknown')}",
        f"Folds evaluated: {agg.get('folds_evaluated', [])}",
        f"Total windows: {agg.get('total_windows', 0)}",
        f"Total predicted segments: {agg.get('total_segments', 0)}",
        "",
        "Results (mean ± std):",
    ]
    
    # mAP results
    for key in sorted(agg.keys()):
        if key.endswith("_mean") and "mAP" in key:
            base_key = key.replace("_mean", "")
            mean_val = agg.get(f"{base_key}_mean")
            std_val = agg.get(f"{base_key}_std", 0)
            
            if mean_val is not None:
                lines.append(f"  {base_key}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                lines.append(f"  {base_key}: N/A")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run TAL evaluation across CV folds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model specification
    parser.add_argument(
        "--model",
        choices=["vjepa", "posec3d", "stgcn"],
        required=True,
        help="Model type to evaluate.",
    )
    parser.add_argument(
        "--preds-root",
        type=Path,
        required=True,
        help="Root directory containing fold predictions.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for evaluation results.",
    )
    
    # Data paths
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=_REPO_ROOT / "dataprep/splits",
        help="Root directory containing split CSVs.",
    )
    parser.add_argument(
        "--ann-pkl-root",
        type=Path,
        default=_REPO_ROOT / "pyskl/data/sails/tal/cv_4class/5class_windows_conf04",
        help="Root directory for pyskl annotation pickles.",
    )
    parser.add_argument(
        "--window-csv-root",
        type=Path,
        default=_REPO_ROOT / "dataprep/tal/splits_cv_4class",
        help="Root directory for TAL window CSVs.",
    )
    parser.add_argument(
        "--task",
        choices=["4class", "5class"],
        default="4class",
        help="Task type (default: 4class).",
    )
    
    # Evaluation settings
    parser.add_argument(
        "--tiou-thresholds",
        type=str,
        default="0.3,0.5,0.7",
        help="Comma-separated tIoU thresholds.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0,1,2",
        help="Comma-separated fold indices to evaluate.",
    )
    
    # Postprocessing parameters
    parser.add_argument("--smooth-k", type=int, default=3)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--thr-per-class", type=str, default=None)
    parser.add_argument("--merge-gap-sec", type=float, default=1.0)
    parser.add_argument("--min-duration-sec", type=float, default=0.0)
    parser.add_argument("--score-reducer", choices=["max", "mean"], default="max")
    
    args = parser.parse_args()
    
    # Parse thresholds and folds
    tiou_thresholds = [float(x.strip()) for x in args.tiou_thresholds.split(",")]
    fold_indices = [int(x.strip()) for x in args.folds.split(",")]
    
    # Set up postprocessing params
    threshold_per_class = None
    if args.thr_per_class:
        threshold_per_class = {}
        for pair in args.thr_per_class.split(","):
            cid, thr = pair.split(":")
            threshold_per_class[int(cid)] = float(thr)
    
    class_ids = [0, 1, 2, 3] if args.task == "4class" else [0, 1, 2, 3, 4]
    
    params = PostprocessParams(
        smooth_k=args.smooth_k,
        threshold=args.thr,
        threshold_per_class=threshold_per_class,
        merge_gap_sec=args.merge_gap_sec,
        min_duration_sec=args.min_duration_sec,
        score_reducer=args.score_reducer,
        class_ids=class_ids,
    )
    
    print("=" * 60)
    print(f"TAL CV Evaluation: {args.model.upper()}")
    print("=" * 60)
    print(f"Predictions root: {args.preds_root}")
    print(f"Output directory: {args.out_dir}")
    print(f"Folds to evaluate: {fold_indices}")
    print(f"tIoU thresholds: {tiou_thresholds}")
    print()
    
    # Evaluate each fold
    fold_results: List[FoldResult] = []
    cache_dir = args.out_dir / "cache"
    
    for fold_idx in fold_indices:
        print(f"\n{'='*40}")
        print(f"Fold {fold_idx}")
        print(f"{'='*40}")
        
        preds_df = None
        
        if args.model == "vjepa":
            # Load V-JEPA predictions directly
            preds_path = find_vjepa_preds(args.preds_root, fold_idx)
            if preds_path:
                print(f"  Loading predictions: {preds_path}")
                preds_df = pd.read_csv(preds_path)
            else:
                print(f"  No predictions found for fold {fold_idx}")
        else:
            # Export pyskl predictions
            preds_df = load_or_export_pyskl_preds(
                preds_root=args.preds_root,
                fold_idx=fold_idx,
                model=args.model,
                ann_pkl_root=args.ann_pkl_root,
                window_csv_root=args.window_csv_root,
                cache_dir=cache_dir,
            )
        
        if preds_df is None or preds_df.empty:
            print(f"  Skipping fold {fold_idx}: no predictions available")
            continue
        
        print(f"  Loaded {len(preds_df)} window predictions")
        
        # Evaluate
        fold_out = args.out_dir / f"fold{fold_idx}"
        metrics = evaluate_fold(
            preds_df=preds_df,
            splits_root=args.splits_root,
            task=args.task,
            mode="cv",
            params=params,
            tiou_thresholds=tiou_thresholds,
            out_dir=fold_out,
        )
        
        # Print fold summary
        print(f"\n  Fold {fold_idx} Results:")
        for thr in tiou_thresholds:
            map_val = metrics.get(f"mAP@{thr}", float("nan"))
            if not np.isnan(map_val):
                print(f"    mAP@{thr}: {map_val:.4f}")
        
        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            model=args.model,
            metrics=metrics,
            n_windows=metrics.get("n_windows", 0),
            n_segments=metrics.get("n_segments", 0),
        ))
    
    # Aggregate CV results
    print("\n" + "=" * 60)
    print("Aggregating CV Results")
    print("=" * 60)
    
    agg = aggregate_cv_results(fold_results, tiou_thresholds)
    
    # Print and save summary
    summary = format_cv_summary(agg)
    print("\n" + summary)
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary JSON
    summary_path = args.out_dir / "cv_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj) if not np.isnan(obj) else None
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(agg), f, indent=2)
    print(f"\nCV summary saved to: {summary_path}")
    
    # Save summary text
    report_path = args.out_dir / "cv_summary.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(summary)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

