#!/usr/bin/env python3
"""
Figure 5.2: Top-1 vs Top-2 Recall Bar Chart for Window-level Detection.

This script computes per-class Top-1 and Top-2 recall for the best V-JEPA
window-level detection model, showing how often the correct class appears
in the model's top predictions.

Usage:
    python fig_5_2_top1_top2_recall.py
    python fig_5_2_top1_top2_recall.py --output fig_5_2.png
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Paths
EVAL_RESULTS_DIR = ACTREG_ROOT / "tal" / "eval_results"
DATAPREP_DIR = ACTREG_ROOT / "dataprep" / "tal"
OUTPUT_DIR = ACTREG_ROOT / "insights" / "paper_plots"

# Class mapping for 5-class detection (including background)
# Predictions: class0-3 = RMM classes, class4 = background
# GT labels in splits: 0-3 = RMM classes, -1 = background
CLASS_NAMES = {
    0: "Hands Flapping",
    1: "Jumping",
    2: "Rocking",
    3: "Spinning",
    4: "Background"
}

# Map GT primary_label to 5-class prediction index
GT_TO_PRED_CLASS = {
    0: 0,  # hands_flapping
    1: 1,  # jumping
    2: 2,  # rocking
    3: 3,  # spinning
    -1: 4  # background
}


def load_predictions(model_name: str, fold: int) -> pd.DataFrame:
    """
    Load window-level predictions from tal_format_preds.csv.
    
    Args:
        model_name: Model directory name under eval_results/
        fold: Fold number (0, 1, or 2)
        
    Returns:
        DataFrame with window_id and per-class scores
    """
    pred_file = EVAL_RESULTS_DIR / model_name / f"fold{fold}" / "tal_format_preds.csv"
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_file}")
    
    df = pd.read_csv(pred_file)
    return df


def load_ground_truth(fold: int) -> pd.DataFrame:
    """
    Load ground truth window labels from the splits CSV.
    
    Args:
        fold: Fold number (0, 1, or 2)
        
    Returns:
        DataFrame with window_id and ground truth labels
    """
    gt_file = DATAPREP_DIR / "splits_cv_4class" / f"fold_{fold}_val_windows.csv"
    
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_file}")
    
    df = pd.read_csv(gt_file)
    
    # Parse labels JSON
    df["labels"] = df["labels"].apply(json.loads)
    
    return df


def compute_topk_recall(
    preds_df: pd.DataFrame, 
    gt_df: pd.DataFrame, 
    k: int = 1
) -> Dict[int, float]:
    """
    Compute per-class Top-k recall.
    
    Top-k recall for a class = (# of windows where true class is in top-k predictions) / (# of windows with that class)
    
    Args:
        preds_df: DataFrame with window predictions (score_class0...score_class4)
        gt_df: DataFrame with ground truth (primary_label, labels)
        k: Number of top predictions to consider
        
    Returns:
        Dictionary mapping class_id to recall value
    """
    # Join predictions with ground truth on window_id
    merged = gt_df.merge(preds_df, on="window_id", how="inner")
    
    if len(merged) == 0:
        raise ValueError("No matching windows between predictions and ground truth")
    
    # Extract score columns
    score_cols = [f"score_class{i}" for i in range(5)]
    scores = merged[score_cols].values  # (n_windows, 5)
    
    # Get top-k predicted classes for each window
    topk_preds = np.argsort(scores, axis=1)[:, -k:]  # (n_windows, k)
    
    # Get ground truth class (using primary_label, mapped to 5-class space)
    gt_classes = merged["primary_label"].map(GT_TO_PRED_CLASS).values
    
    # Compute per-class recall
    recall = {}
    for class_id in range(5):
        # Windows where this class is the ground truth
        mask = gt_classes == class_id
        n_total = mask.sum()
        
        if n_total == 0:
            recall[class_id] = np.nan
            continue
        
        # Check if ground truth class is in top-k predictions
        gt_in_topk = np.any(topk_preds[mask] == class_id, axis=1)
        n_correct = gt_in_topk.sum()
        
        recall[class_id] = n_correct / n_total
    
    return recall


def compute_cv_topk_recall(model_name: str, k: int = 1) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute per-class Top-k recall across all CV folds.
    
    Args:
        model_name: Model directory name
        k: Number of top predictions to consider
        
    Returns:
        Tuple of (mean_recall, std_recall) dictionaries
    """
    fold_recalls = []
    
    for fold in range(3):
        try:
            preds_df = load_predictions(model_name, fold)
            gt_df = load_ground_truth(fold)
            recall = compute_topk_recall(preds_df, gt_df, k=k)
            fold_recalls.append(recall)
        except FileNotFoundError as e:
            print(f"Warning: Skipping fold {fold}: {e}")
            continue
    
    if len(fold_recalls) == 0:
        raise ValueError(f"No valid folds found for model: {model_name}")
    
    # Aggregate across folds
    mean_recall = {}
    std_recall = {}
    
    for class_id in range(5):
        values = [r[class_id] for r in fold_recalls if not np.isnan(r.get(class_id, np.nan))]
        if values:
            mean_recall[class_id] = np.mean(values)
            std_recall[class_id] = np.std(values)
        else:
            mean_recall[class_id] = np.nan
            std_recall[class_id] = np.nan
    
    return mean_recall, std_recall


def plot_top1_vs_top2_recall(
    top1_mean: Dict[int, float],
    top1_std: Dict[int, float],
    top2_mean: Dict[int, float],
    top2_std: Dict[int, float],
    output_path: Path = None,
    title: str = "V-JEPA Balanced: Per-class Top-1 vs Top-2 Recall"
):
    """
    Create a grouped bar chart comparing Top-1 and Top-2 recall per class.
    
    Args:
        top1_mean: Mean Top-1 recall per class
        top1_std: Std Top-1 recall per class
        top2_mean: Mean Top-2 recall per class
        top2_std: Std Top-2 recall per class
        output_path: Path to save the figure
        title: Plot title
    """
    # Only plot RMM classes (not background)
    class_ids = [0, 1, 2, 3]
    class_labels = [CLASS_NAMES[c] for c in class_ids]
    
    top1_vals = [top1_mean[c] * 100 for c in class_ids]
    top1_errs = [top1_std[c] * 100 for c in class_ids]
    top2_vals = [top2_mean[c] * 100 for c in class_ids]
    top2_errs = [top2_std[c] * 100 for c in class_ids]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(class_labels))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(
        x - width/2, top1_vals, width, 
        yerr=top1_errs, 
        label='Top-1 Recall',
        color='#FFB74D',  # Lighter orange
        capsize=5,
        alpha=0.8
    )
    bars2 = ax.bar(
        x + width/2, top2_vals, width, 
        yerr=top2_errs, 
        label='Top-2 Recall',
        color='#81C784',  # Lighter green
        capsize=5,
        alpha=0.8
    )
    
    # Labels and formatting
    ax.set_xlabel('RMM Class', fontsize=12)
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    def add_labels(bars, vals, errs):
        for bar, val, err in zip(bars, vals, errs):
            height = bar.get_height()
            ax.annotate(
                f'{val:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height + err + 1),
                ha='center', va='bottom',
                fontsize=9
            )
    
    add_labels(bars1, top1_vals, top1_errs)
    add_labels(bars2, top2_vals, top2_errs)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {output_path}")
        
        # Also save as PDF for publication
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved PDF: {pdf_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_table(
    top1_mean: Dict[int, float],
    top1_std: Dict[int, float],
    top2_mean: Dict[int, float],
    top2_std: Dict[int, float]
):
    """Print a markdown table summarizing the results."""
    print("\n### Top-1 vs Top-2 Recall Summary (V-JEPA Balanced)")
    print("\n| Class | Top-1 Recall | Top-2 Recall | Gap |")
    print("|-------|--------------|--------------|-----|")
    
    for class_id in range(5):
        name = CLASS_NAMES[class_id]
        t1 = top1_mean[class_id] * 100
        t1_std = top1_std[class_id] * 100
        t2 = top2_mean[class_id] * 100
        t2_std = top2_std[class_id] * 100
        gap = t2 - t1
        
        print(f"| {name:<14} | {t1:.1f}% ± {t1_std:.1f}% | {t2:.1f}% ± {t2_std:.1f}% | +{gap:.1f}% |")


def save_results_json(
    top1_mean: Dict[int, float],
    top1_std: Dict[int, float],
    top2_mean: Dict[int, float],
    top2_std: Dict[int, float],
    output_path: Path
):
    """Save results as JSON for later use."""
    results = {
        "model": "vjepa_balanced",
        "task": "window_level_detection_5class",
        "per_class": {}
    }
    
    for class_id in range(5):
        results["per_class"][CLASS_NAMES[class_id]] = {
            "top1_recall": {
                "mean": top1_mean[class_id],
                "std": top1_std[class_id]
            },
            "top2_recall": {
                "mean": top2_mean[class_id],
                "std": top2_std[class_id]
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Top-1 vs Top-2 Recall bar chart (Figure 5.2)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vjepa_balanced",
        help="Model name under eval_results/ (default: vjepa_balanced)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path (default: insights/paper_plots/fig_5_2_top1_top2_recall.png)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting, only print summary"
    )
    
    args = parser.parse_args()
    
    # Compute Top-1 and Top-2 recall
    print(f"Computing Top-1 recall for model: {args.model}")
    top1_mean, top1_std = compute_cv_topk_recall(args.model, k=1)
    
    print(f"Computing Top-2 recall for model: {args.model}")
    top2_mean, top2_std = compute_cv_topk_recall(args.model, k=2)
    
    # Print summary
    print_summary_table(top1_mean, top1_std, top2_mean, top2_std)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / "fig_5_2_top1_top2_recall.png"
    
    # Save JSON results
    json_path = output_path.with_suffix('.json')
    save_results_json(top1_mean, top1_std, top2_mean, top2_std, json_path)
    
    # Generate plot
    if not args.no_plot:
        plot_top1_vs_top2_recall(
            top1_mean, top1_std,
            top2_mean, top2_std,
            output_path=output_path
        )


if __name__ == "__main__":
    main()
