#!/usr/bin/env python3
"""
Evaluate TAL models with best postprocessing parameters from grid search.

This script:
1. Reads best_params_per_model.json from grid search
2. For each model/fold, runs evaluation with optimal params
3. Writes per-fold artifacts (metrics.json, pred_segments.csv, report.txt)
4. Aggregates into cv_summary.json

Usage:
    # Evaluate all models with best params
    python scripts/eval_best_postprocess.py
    
    # Evaluate specific models only
    python scripts/eval_best_postprocess.py --models vjepa_posec3d_mlp_logp vjepa_stgcnpp_mlp_logp
    
    # With custom paths
    python scripts/eval_best_postprocess.py --best-params-json /path/to/best_params.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from window_to_segments import PostprocessParams, window_scores_to_segments
from tal_map_eval import (
    ID2LABEL_4CLASS,
    ID2LABEL_5CLASS,
    compute_map,
    format_metrics_report,
    load_gt_segments,
    save_metrics,
)

# Import model configs from grid search
from scripts.grid_search_postprocessing import MODELS, CLASS_IDS, TIOU_THRESHOLDS


# =============================================================================
# Path Configuration
# =============================================================================

TAL_DIR = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = TAL_DIR / "eval_results"
SPLITS_ROOT = TAL_DIR.parent / "dataprep" / "splits"


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_fold_with_best_params(
    pred_csv: Path,
    gt_by_video: Dict,
    best_params: Dict,
    out_dir: Path,
    id2label: Dict[int, str],
) -> Dict:
    """
    Evaluate a single fold with the best postprocessing parameters.
    
    Args:
        pred_csv: Path to TAL-format prediction CSV
        gt_by_video: GT segments by video key
        best_params: Dict with thr, smooth_k, merge_gap_sec
        out_dir: Output directory for artifacts
        id2label: Class ID to label name mapping
    
    Returns:
        Metrics dict
    """
    # Load predictions
    df = pd.read_csv(pred_csv)
    
    # Build postprocess params
    pp_params = PostprocessParams(
        smooth_k=best_params["smooth_k"],
        threshold=best_params["thr"],
        merge_gap_sec=best_params["merge_gap_sec"],
        min_duration_sec=0.0,
        score_reducer="max",
        class_ids=CLASS_IDS,
    )
    
    # Postprocess to segments
    segments_df = window_scores_to_segments(df, pp_params)
    
    # Compute metrics
    metrics = compute_map(segments_df, gt_by_video, CLASS_IDS, TIOU_THRESHOLDS)
    
    # Add summary stats
    metrics["n_windows"] = len(df)
    metrics["n_segments"] = len(segments_df)
    metrics["n_videos"] = df["video_key"].nunique()
    metrics["postprocess_params"] = best_params
    
    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predicted segments
    segments_df.to_csv(out_dir / "pred_segments.csv", index=False)
    
    # Save metrics using existing function
    save_metrics(metrics, out_dir, CLASS_IDS, id2label)
    
    # Save evaluation config
    config = {
        "pred_csv": str(pred_csv),
        "postprocess": best_params,
        "class_ids": CLASS_IDS,
        "tiou_thresholds": TIOU_THRESHOLDS,
    }
    with open(out_dir / "eval_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return metrics


def aggregate_cv_results(
    fold_metrics: Dict[int, Dict],
    model_name: str,
) -> Dict:
    """
    Aggregate metrics across CV folds.
    
    Args:
        fold_metrics: Dict mapping fold_idx -> metrics dict
        model_name: Display name for the model
    
    Returns:
        Aggregated summary dict
    """
    if not fold_metrics:
        return {"error": "No fold results available"}
    
    # Keys to aggregate
    agg_keys = ["mAP@0.3", "mAP@0.5", "mAP@0.7", "avg_mAP"]
    
    agg = {
        "model": model_name,
        "n_folds": len(fold_metrics),
        "folds_evaluated": sorted(fold_metrics.keys()),
    }
    
    for key in agg_keys:
        values = [
            m[key] for m in fold_metrics.values()
            if key in m and not np.isnan(m[key])
        ]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
            agg[f"{key}_values"] = values
        else:
            agg[f"{key}_mean"] = None
            agg[f"{key}_std"] = None
    
    # Total stats
    agg["total_windows"] = sum(m.get("n_windows", 0) for m in fold_metrics.values())
    agg["total_segments"] = sum(m.get("n_segments", 0) for m in fold_metrics.values())
    
    return agg


def format_cv_summary(agg: Dict) -> str:
    """Format CV summary as human-readable report."""
    lines = [
        "=" * 60,
        f"Cross-Validation Summary: {agg.get('model', 'unknown')}",
        "=" * 60,
        f"Folds evaluated: {agg.get('folds_evaluated', [])}",
        f"Total windows: {agg.get('total_windows', 0)}",
        f"Total predicted segments: {agg.get('total_segments', 0)}",
        "",
        "Results (mean ± std):",
    ]
    
    for key in ["mAP@0.3", "mAP@0.5", "mAP@0.7", "avg_mAP"]:
        mean_val = agg.get(f"{key}_mean")
        std_val = agg.get(f"{key}_std", 0)
        
        if mean_val is not None:
            lines.append(f"  {key}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            lines.append(f"  {key}: N/A")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def run_eval_for_model(
    model_key: str,
    best_params: Dict,
    eval_results_dir: Path,
    gt_by_video: Dict,
    id2label: Dict[int, str],
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Run evaluation for a single model across all available folds.
    
    Args:
        model_key: Key from MODELS dict
        best_params: Best postprocessing parameters
        eval_results_dir: Root directory for TAL-format predictions
        gt_by_video: GT segments by video key
        id2label: Class ID to label name mapping
        verbose: Print progress
    
    Returns:
        CV summary dict or None if no folds available
    """
    if model_key not in MODELS:
        print(f"Warning: Unknown model '{model_key}', skipping")
        return None
    
    config = MODELS[model_key]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Model: {config.name}")
        print(f"{'='*60}")
        print(f"Best params: thr={best_params['thr']}, smooth_k={best_params['smooth_k']}, merge_gap={best_params['merge_gap_sec']}")
    
    # Find available folds
    fold_metrics = {}
    
    for fold_idx in config.folds:
        pred_csv = eval_results_dir / config.pred_csv_pattern.format(fold=fold_idx)
        
        if not pred_csv.exists():
            if verbose:
                print(f"  Fold {fold_idx}: Skipping (CSV not found: {pred_csv})")
            continue
        
        # Output directory for this fold
        out_dir = eval_results_dir / config.pred_csv_pattern.format(fold=fold_idx).replace("tal_format_preds.csv", "best_eval")
        out_dir = out_dir.parent / "best_eval"
        
        if verbose:
            print(f"  Fold {fold_idx}: Evaluating...")
        
        try:
            metrics = evaluate_fold_with_best_params(
                pred_csv=pred_csv,
                gt_by_video=gt_by_video,
                best_params=best_params,
                out_dir=out_dir,
                id2label=id2label,
            )
            fold_metrics[fold_idx] = metrics
            
            if verbose:
                print(f"    mAP@0.5: {metrics.get('mAP@0.5', 0):.4f}, avg_mAP: {metrics.get('avg_mAP', 0):.4f}")
                print(f"    Artifacts saved to: {out_dir}")
                
        except Exception as e:
            print(f"  Fold {fold_idx}: Error - {e}")
    
    if not fold_metrics:
        if verbose:
            print(f"  No folds available for {config.name}")
        return None
    
    # Aggregate CV results
    cv_summary = aggregate_cv_results(fold_metrics, config.name)
    
    # Save CV summary
    model_out_dir = eval_results_dir / config.pred_csv_pattern.format(fold=0).split("/")[0]
    
    cv_summary_json = model_out_dir / "cv_summary.json"
    with open(cv_summary_json, "w") as f:
        json.dump(cv_summary, f, indent=2)
    
    cv_summary_txt = model_out_dir / "cv_summary.txt"
    with open(cv_summary_txt, "w") as f:
        f.write(format_cv_summary(cv_summary))
    
    if verbose:
        print(f"\n{format_cv_summary(cv_summary)}")
        print(f"CV summary saved to: {cv_summary_json}")
    
    return cv_summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TAL models with best postprocessing parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--best-params-json",
        type=Path,
        default=EVAL_RESULTS_DIR / "best_params_per_model.json",
        help="Path to best_params_per_model.json from grid search.",
    )
    parser.add_argument(
        "--eval-results-dir",
        type=Path,
        default=EVAL_RESULTS_DIR,
        help="Root directory for TAL-format predictions.",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=SPLITS_ROOT,
        help="Root directory for GT segment CSVs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to evaluate (default: all in best_params_json).",
    )
    parser.add_argument(
        "--task",
        choices=["4class", "5class"],
        default="4class",
        help="Task type for GT loading.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Load best params
    if not args.best_params_json.exists():
        print(f"Error: best_params_json not found: {args.best_params_json}")
        print("Run grid search first: python scripts/grid_search_postprocessing.py")
        sys.exit(1)
    
    with open(args.best_params_json) as f:
        all_best_params = json.load(f)
    
    if verbose:
        print("=" * 60)
        print("TAL Evaluation with Best Postprocessing Parameters")
        print("=" * 60)
        print(f"Best params file: {args.best_params_json}")
        print(f"Eval results dir: {args.eval_results_dir}")
        print(f"Splits root: {args.splits_root}")
    
    # Determine which models to evaluate
    if args.models:
        models_to_eval = args.models
    else:
        models_to_eval = list(all_best_params.keys())
    
    if verbose:
        print(f"Models to evaluate: {models_to_eval}")
    
    # Load GT segments
    if verbose:
        print("\nLoading GT segments...")
    
    gt_by_video = load_gt_segments(args.splits_root, args.task, "cv")
    
    if verbose:
        n_gt = sum(len(v) for v in gt_by_video.values())
        print(f"  Loaded {n_gt} GT segments from {len(gt_by_video)} videos")
    
    # Get id2label
    id2label = ID2LABEL_4CLASS if args.task == "4class" else ID2LABEL_5CLASS
    
    # Evaluate each model
    all_summaries = {}
    
    for model_key in models_to_eval:
        if model_key not in all_best_params:
            print(f"\nWarning: No best params found for '{model_key}', skipping")
            continue
        
        best_params = all_best_params[model_key]["best_params"]
        
        cv_summary = run_eval_for_model(
            model_key=model_key,
            best_params=best_params,
            eval_results_dir=args.eval_results_dir,
            gt_by_video=gt_by_video,
            id2label=id2label,
            verbose=verbose,
        )
        
        if cv_summary:
            all_summaries[model_key] = cv_summary
    
    # Print final comparison
    if verbose and all_summaries:
        print("\n" + "=" * 70)
        print("FINAL COMPARISON (avg_mAP with best postprocessing)")
        print("=" * 70)
        
        # Sort by avg_mAP
        sorted_models = sorted(
            all_summaries.items(),
            key=lambda x: x[1].get("avg_mAP_mean", 0) or 0,
            reverse=True,
        )
        
        for rank, (model_key, summary) in enumerate(sorted_models, 1):
            avg_map = summary.get("avg_mAP_mean", 0) or 0
            avg_map_std = summary.get("avg_mAP_std", 0) or 0
            map05 = summary.get("mAP@0.5_mean", 0) or 0
            
            print(f"{rank}. {summary['model']}")
            print(f"   avg_mAP: {avg_map:.4f} ± {avg_map_std:.4f}")
            print(f"   mAP@0.5: {map05:.4f}")
            print()
    
    print("Done!")


if __name__ == "__main__":
    main()










