#!/usr/bin/env python3
"""
Aggregate per-class Precision/Recall metrics across CV folds.

This script processes per_class_metrics.csv files from TAL evaluation results
and computes mean ± std for each class across folds.

Usage:
    python aggregate_per_class_pr.py --model vjepa_balanced --tiou 0.3
    python aggregate_per_class_pr.py --model vjepa_balanced --all-tiou
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Eval results directory
EVAL_RESULTS_DIR = ACTREG_ROOT / "tal" / "eval_results"

# Class names for TAL (4-class)
TAL_CLASSES = ["hands flapping", "jumping", "rocking", "spinning"]


def load_per_class_metrics(model_name: str) -> pd.DataFrame:
    """
    Load per-class metrics from all available folds for a model.
    
    Args:
        model_name: Name of the model directory under eval_results/
        
    Returns:
        DataFrame with all per-class metrics across folds
    """
    model_dir = EVAL_RESULTS_DIR / model_name
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    all_data = []
    fold_dirs = sorted(model_dir.glob("fold*"))
    
    for fold_dir in fold_dirs:
        metrics_file = fold_dir / "best_eval" / "per_class_metrics.csv"
        
        if not metrics_file.exists():
            print(f"Warning: Metrics file not found: {metrics_file}")
            continue
            
        df = pd.read_csv(metrics_file)
        # Extract fold number from directory name
        fold_num = int(fold_dir.name.replace("fold", ""))
        df["fold"] = fold_num
        all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No per-class metrics files found for model: {model_name}")
        
    return pd.concat(all_data, ignore_index=True)


def aggregate_metrics(df: pd.DataFrame, tiou_threshold: str = None) -> pd.DataFrame:
    """
    Aggregate metrics across folds, computing mean and std.
    
    Args:
        df: DataFrame with per-class metrics from all folds
        tiou_threshold: Optional filter for specific tIoU (e.g., "tIoU=0.3")
        
    Returns:
        DataFrame with aggregated mean ± std for each class and tIoU
    """
    if tiou_threshold:
        df = df[df["tiou_threshold"] == tiou_threshold]
    
    # Group by tIoU threshold and class
    grouped = df.groupby(["tiou_threshold", "class_id", "class_name"])
    
    # Compute statistics for precision and recall
    agg_results = []
    
    for (tiou, class_id, class_name), group in grouped:
        n_folds = len(group)
        
        result = {
            "tiou_threshold": tiou,
            "class_id": class_id,
            "class_name": class_name,
            "n_folds": n_folds,
            "precision_mean": group["precision"].mean(),
            "precision_std": group["precision"].std(),
            "recall_mean": group["recall"].mean(),
            "recall_std": group["recall"].std(),
            "AP_mean": group["AP"].mean(),
            "AP_std": group["AP"].std(),
            "n_gt_total": group["n_gt"].sum(),
            "n_pred_total": group["n_pred"].sum(),
            "n_tp_total": group["n_tp"].sum(),
        }
        agg_results.append(result)
    
    return pd.DataFrame(agg_results)


def format_metric(mean: float, std: float, as_percent: bool = True) -> str:
    """Format a metric as mean ± std string."""
    if as_percent:
        return f"{mean*100:.1f}% ± {std*100:.1f}%"
    return f"{mean:.3f} ± {std:.3f}"


def print_table_5_6(agg_df: pd.DataFrame, tiou: str = "tIoU=0.3"):
    """
    Print Table 5.6: Per-class P/R for Window-level Detection (V-JEPA Balanced).
    
    Args:
        agg_df: Aggregated metrics DataFrame
        tiou: tIoU threshold to report (default: tIoU=0.3)
    """
    print("\n" + "="*80)
    print(f"Table 5.6: Per-class Precision/Recall for Window-level Detection")
    print(f"Model: V-JEPA Balanced | tIoU threshold: {tiou}")
    print("="*80)
    
    subset = agg_df[agg_df["tiou_threshold"] == tiou].sort_values("class_id")
    
    print(f"\n| Class | Precision | Recall | AP |")
    print(f"|-------|-----------|--------|-----|")
    
    for _, row in subset.iterrows():
        prec = format_metric(row["precision_mean"], row["precision_std"])
        rec = format_metric(row["recall_mean"], row["recall_std"])
        ap = format_metric(row["AP_mean"], row["AP_std"])
        print(f"| {row['class_name']:<14} | {prec:<18} | {rec:<18} | {ap:<18} |")
    
    print()


def print_all_tiou_summary(agg_df: pd.DataFrame):
    """Print summary across all tIoU thresholds."""
    print("\n" + "="*80)
    print("Per-class Metrics Summary (All tIoU Thresholds)")
    print("="*80)
    
    for tiou in ["tIoU=0.3", "tIoU=0.5", "tIoU=0.7"]:
        print(f"\n### {tiou}")
        subset = agg_df[agg_df["tiou_threshold"] == tiou].sort_values("class_id")
        
        print(f"\n| Class | Precision | Recall | AP |")
        print(f"|-------|-----------|--------|-----|")
        
        for _, row in subset.iterrows():
            prec = format_metric(row["precision_mean"], row["precision_std"])
            rec = format_metric(row["recall_mean"], row["recall_std"])
            ap = format_metric(row["AP_mean"], row["AP_std"])
            print(f"| {row['class_name']:<14} | {prec:<18} | {rec:<18} | {ap:<18} |")


def save_results(agg_df: pd.DataFrame, output_path: Path):
    """Save aggregated results to CSV and JSON."""
    # Save CSV
    csv_path = output_path.with_suffix(".csv")
    agg_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")
    
    # Save JSON with formatted strings
    json_data = {}
    for tiou in agg_df["tiou_threshold"].unique():
        json_data[tiou] = {}
        subset = agg_df[agg_df["tiou_threshold"] == tiou]
        for _, row in subset.iterrows():
            json_data[tiou][row["class_name"]] = {
                "precision": {
                    "mean": row["precision_mean"],
                    "std": row["precision_std"],
                    "formatted": format_metric(row["precision_mean"], row["precision_std"])
                },
                "recall": {
                    "mean": row["recall_mean"],
                    "std": row["recall_std"],
                    "formatted": format_metric(row["recall_mean"], row["recall_std"])
                },
                "AP": {
                    "mean": row["AP_mean"],
                    "std": row["AP_std"],
                    "formatted": format_metric(row["AP_mean"], row["AP_std"])
                }
            }
    
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-class P/R metrics across CV folds"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="vjepa_balanced",
        help="Model name under eval_results/ (default: vjepa_balanced)"
    )
    parser.add_argument(
        "--tiou",
        type=str,
        default="0.3",
        help="tIoU threshold to report (default: 0.3)"
    )
    parser.add_argument(
        "--all-tiou",
        action="store_true",
        help="Print summary for all tIoU thresholds"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: insights/paper_plots/table_5_6_{model}.csv)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading per-class metrics for model: {args.model}")
    df = load_per_class_metrics(args.model)
    print(f"Loaded {len(df)} rows from {df['fold'].nunique()} folds")
    
    # Aggregate
    agg_df = aggregate_metrics(df)
    
    # Print results
    tiou_str = f"tIoU={args.tiou}"
    
    if args.all_tiou:
        print_all_tiou_summary(agg_df)
    else:
        print_table_5_6(agg_df, tiou=tiou_str)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ACTREG_ROOT / "insights" / "paper_plots" / f"table_5_6_{args.model}"
    
    save_results(agg_df, output_path)


if __name__ == "__main__":
    main()
