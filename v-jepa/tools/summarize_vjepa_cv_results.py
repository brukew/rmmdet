#!/usr/bin/env python
"""
Summarize cross-validation results for V-JEPA TAL/RMM experiments.

This script computes CV summaries (mean ± std across folds) from metrics.json files
in each fold directory.

Usage:
    # Summarize V-JEPA TAL results
    python tools/summarize_vjepa_cv_results.py \
        --work-dir runs/vjepa2_tal_cv_5class_bgsub \
        --task tal

    # Summarize V-JEPA RMM results  
    python tools/summarize_vjepa_cv_results.py \
        --work-dir runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop \
        --task rmm

    # Add metadata
    python tools/summarize_vjepa_cv_results.py \
        --work-dir runs/vjepa2_tal_cv_5class_bgsub \
        --task tal \
        --metadata "class_prob=[1.0,1.0,1.93,9.12,0.1]" "patience=5"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


def load_fold_metrics(fold_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metrics.json from a fold directory."""
    metrics_file = fold_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    
    # Fallback: try to find eval_val/metrics.json (if structure differs)
    alt_metrics_file = fold_dir / "eval_val" / "metrics.json"
    if alt_metrics_file.exists():
        with open(alt_metrics_file) as f:
            return json.load(f)
    
    return None


def compute_cv_summary(
    work_dir: Path,
    task: str,
    num_folds: int = 3,
    metadata: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute CV summary (mean ± std) for V-JEPA results.
    
    Args:
        work_dir: Directory containing fold_0, fold_1, fold_2 subdirectories.
        task: Task type ('tal' or 'rmm').
        num_folds: Number of CV folds.
        metadata: Additional metadata to include in summary.
    
    Returns:
        Summary dict with per-fold results and aggregated stats.
    """
    folds = []
    
    print(f"\nLoading fold metrics from: {work_dir}")
    
    for fold in range(num_folds):
        fold_dir = work_dir / f"fold_{fold}"
        if not fold_dir.exists():
            # Also try without underscore
            fold_dir = work_dir / f"fold{fold}"
        
        metrics = load_fold_metrics(fold_dir)
        
        if metrics:
            metrics["fold"] = fold
            folds.append(metrics)
            print(f"  Fold {fold}: loaded")
        else:
            print(f"  Fold {fold}: metrics not found")
    
    if not folds:
        print("No fold metrics found!")
        return None
    
    # Build summary
    summary: Dict[str, Any] = {
        "model": "V-JEPA2",
        "task": task,
        "per_fold": folds,
    }
    
    if metadata:
        summary.update(metadata)
    
    # Compute mean/std for numeric metrics
    # Skip non-numeric fields
    skip_fields = {"fold", "split", "task", "checkpoint", "train_csv", "val_csv"}
    metric_keys = [k for k in folds[0].keys() if k not in skip_fields]
    
    for key in metric_keys:
        vals = [f[key] for f in folds if key in f and f[key] is not None]
        if vals and isinstance(vals[0], (int, float)):
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
    
    return summary


def print_summary(summary: Dict[str, Any], task: str) -> None:
    """Print formatted summary to console."""
    print("\n" + "=" * 60)
    print("CV Summary")
    print("=" * 60)
    print(f"Model: {summary.get('model', 'V-JEPA2')}")
    print(f"Task: {task.upper()}")
    print(f"Folds: {len(summary.get('per_fold', []))}")
    
    # Key metrics
    print("\nKey Metrics:")
    print(f"  Clip Top-1: {summary.get('clip_top1_acc_mean', 0):.2f} ± {summary.get('clip_top1_acc_std', 0):.2f}%")
    print(f"  Clip Top-2: {summary.get('clip_top2_acc_mean', 0):.2f} ± {summary.get('clip_top2_acc_std', 0):.2f}%")
    print(f"  Macro-F1: {summary.get('clip_macro_f1_mean', 0):.2f} ± {summary.get('clip_macro_f1_std', 0):.2f}%")
    print(f"  Macro-Precision: {summary.get('clip_macro_precision_mean', 0):.2f}%")
    print(f"  Macro-Recall: {summary.get('clip_macro_recall_mean', 0):.2f}%")
    
    if task == "tal":
        print("\nRMM vs Background:")
        print(f"  Accuracy: {summary.get('clip_rmm_vs_bg_accuracy_mean', 0):.2f}%")
        print(f"  F1: {summary.get('clip_rmm_vs_bg_f1_mean', 0):.2f} ± {summary.get('clip_rmm_vs_bg_f1_std', 0):.2f}%")
        print(f"  Precision: {summary.get('clip_rmm_vs_bg_precision_mean', 0):.2f}%")
        print(f"  Recall: {summary.get('clip_rmm_vs_bg_recall_mean', 0):.2f}%")
        print(f"  AUC: {summary.get('clip_rmm_vs_bg_auc_mean', 0):.2f}%")
        print(f"  BG False Alarm Rate: {summary.get('clip_bg_false_alarm_rate_mean', 0):.2f}%")
        print(f"  RMM Miss Rate: {summary.get('clip_rmm_miss_rate_mean', 0):.2f}%")
        print("\nRMM-Only (excluding background):")
        print(f"  Macro-F1: {summary.get('clip_rmm_only_macro_f1_mean', 0):.2f} ± {summary.get('clip_rmm_only_macro_f1_std', 0):.2f}%")
        print(f"  Macro-Precision: {summary.get('clip_rmm_only_macro_precision_mean', 0):.2f}%")
        print(f"  Macro-Recall: {summary.get('clip_rmm_only_macro_recall_mean', 0):.2f}%")
    
    # Per-class metrics
    per_class_keys = [k for k in summary.keys() if k.startswith("clip_f1_") and k.endswith("_mean")]
    if per_class_keys:
        print("\nPer-Class F1:")
        for key in sorted(per_class_keys):
            class_name = key.replace("clip_f1_", "").replace("_mean", "")
            mean_val = summary.get(key, 0)
            std_key = key.replace("_mean", "_std")
            std_val = summary.get(std_key, 0)
            print(f"  {class_name}: {mean_val:.2f} ± {std_val:.2f}%")
    
    print("=" * 60)


def parse_metadata(metadata_args: Optional[List[str]]) -> Dict[str, str]:
    """Parse metadata key=value pairs from command line."""
    if not metadata_args:
        return {}
    
    result = {}
    for item in metadata_args:
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
        else:
            print(f"Warning: Ignoring invalid metadata format: {item}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Summarize cross-validation results for V-JEPA TAL/RMM experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Base work directory containing fold subdirectories",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["tal", "rmm"],
        help="Task type",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=3,
        help="Number of CV folds (default: 3)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        nargs="*",
        help="Additional metadata as key=value pairs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for cv_summary.json (default: work_dir/cv_summary.json)",
    )
    
    args = parser.parse_args()
    
    # Validate work directory
    if not args.work_dir.exists():
        print(f"Error: Work directory does not exist: {args.work_dir}")
        sys.exit(1)
    
    # Parse metadata
    metadata = parse_metadata(args.metadata)
    
    # Compute summary
    summary = compute_cv_summary(
        args.work_dir,
        args.task,
        args.num_folds,
        metadata,
    )
    
    if summary is None:
        print("Failed to compute summary")
        sys.exit(1)
    
    # Print summary
    print_summary(summary, args.task)
    
    # Save summary
    output_path = args.output or (args.work_dir / "cv_summary.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {output_path}")
    
    # Also save CSV version
    csv_path = output_path.with_suffix(".csv")
    try:
        import pandas as pd
        df = pd.DataFrame(summary["per_fold"])
        df.to_csv(csv_path, index=False)
        print(f"Saved per-fold CSV to: {csv_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()










