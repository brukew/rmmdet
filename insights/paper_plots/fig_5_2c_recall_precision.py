#!/usr/bin/env python3
"""
Figure 5.2c: Top-1 vs Top-2 Recall and Precision Analysis for Window-level Detection.

Creates a 1x2 figure:
- Left (A): Grouped bar chart showing Top-1 vs Top-2 Recall per class
- Right (B): Grouped bar chart showing Top-1 vs Top-2 Precision per class

Usage:
    python fig_5_2c_recall_precision.py
    python fig_5_2c_recall_precision.py --output fig_5_2c_recall_precision.png
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Paths
EVAL_RESULTS_DIR = ACTREG_ROOT / "tal" / "eval_results"
DATAPREP_DIR = ACTREG_ROOT / "dataprep" / "tal"
OUTPUT_DIR = ACTREG_ROOT / "insights" / "figs"

# Class mapping for 5-class detection (including background)
CLASS_NAMES = {
    0: "Hands Flapping",
    1: "Jumping",
    2: "Rocking",
    3: "Spinning",
    4: "Background"
}

# Shorter names for plot labels
CLASS_LABELS_SHORT = {
    0: "Hands\nFlapping",
    1: "Jumping",
    2: "Rocking",
    3: "Spinning",
}

# Map GT primary_label to 5-class prediction index
GT_TO_PRED_CLASS = {
    0: 0,  # hands_flapping
    1: 1,  # jumping
    2: 2,  # rocking
    3: 3,  # spinning
    -1: 4  # background
}

# Consistent color scheme (lighter palette)
COLORS = {
    "top1": "#FFB74D",       # Lighter orange
    "top2": "#81C784",       # Lighter green
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
    
    Recall = (# samples with GT=class AND class in top-k predictions) / (# samples with GT=class)
    
    Args:
        preds_df: DataFrame with window predictions (score_class0...score_class4)
        gt_df: DataFrame with ground truth (primary_label, labels)
        k: Number of top predictions to consider
        
    Returns:
        Dictionary mapping class_id to recall value
    """
    merged = gt_df.merge(preds_df, on="window_id", how="inner")
    
    if len(merged) == 0:
        raise ValueError("No matching windows between predictions and ground truth")
    
    score_cols = [f"score_class{i}" for i in range(5)]
    scores = merged[score_cols].values
    
    topk_preds = np.argsort(scores, axis=1)[:, -k:]
    gt_classes = merged["primary_label"].map(GT_TO_PRED_CLASS).values
    
    recall = {}
    for class_id in range(5):
        mask = gt_classes == class_id
        n_total = mask.sum()
        
        if n_total == 0:
            recall[class_id] = np.nan
            continue
        
        gt_in_topk = np.any(topk_preds[mask] == class_id, axis=1)
        n_correct = gt_in_topk.sum()
        
        recall[class_id] = n_correct / n_total
    
    return recall


def compute_topk_precision(
    preds_df: pd.DataFrame, 
    gt_df: pd.DataFrame, 
    k: int = 1
) -> Dict[int, float]:
    """
    Compute per-class Top-k precision.
    
    Precision = (# samples with GT=class AND class in top-k predictions) / (# samples with class in top-k predictions)
    
    Args:
        preds_df: DataFrame with window predictions (score_class0...score_class4)
        gt_df: DataFrame with ground truth (primary_label, labels)
        k: Number of top predictions to consider
        
    Returns:
        Dictionary mapping class_id to precision value
    """
    merged = gt_df.merge(preds_df, on="window_id", how="inner")
    
    if len(merged) == 0:
        raise ValueError("No matching windows between predictions and ground truth")
    
    score_cols = [f"score_class{i}" for i in range(5)]
    scores = merged[score_cols].values
    
    topk_preds = np.argsort(scores, axis=1)[:, -k:]
    gt_classes = merged["primary_label"].map(GT_TO_PRED_CLASS).values
    
    precision = {}
    for class_id in range(5):
        # Samples where this class is in top-k predictions
        class_in_topk = np.any(topk_preds == class_id, axis=1)
        n_predicted = class_in_topk.sum()
        
        if n_predicted == 0:
            precision[class_id] = np.nan
            continue
        
        # Of those, how many have GT = this class
        correct_mask = class_in_topk & (gt_classes == class_id)
        n_correct = correct_mask.sum()
        
        precision[class_id] = n_correct / n_predicted
    
    return precision


def compute_cv_topk_recall(model_name: str, k: int = 1) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute per-class Top-k recall across all CV folds.
    
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


def compute_cv_topk_precision(model_name: str, k: int = 1) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute per-class Top-k precision across all CV folds.
    
    Returns:
        Tuple of (mean_precision, std_precision) dictionaries
    """
    fold_precisions = []
    
    for fold in range(3):
        try:
            preds_df = load_predictions(model_name, fold)
            gt_df = load_ground_truth(fold)
            precision = compute_topk_precision(preds_df, gt_df, k=k)
            fold_precisions.append(precision)
        except FileNotFoundError as e:
            print(f"Warning: Skipping fold {fold}: {e}")
            continue
    
    if len(fold_precisions) == 0:
        raise ValueError(f"No valid folds found for model: {model_name}")
    
    mean_precision = {}
    std_precision = {}
    
    for class_id in range(5):
        values = [p[class_id] for p in fold_precisions if not np.isnan(p.get(class_id, np.nan))]
        if values:
            mean_precision[class_id] = np.mean(values)
            std_precision[class_id] = np.std(values)
        else:
            mean_precision[class_id] = np.nan
            std_precision[class_id] = np.nan
    
    return mean_precision, std_precision


def plot_recall_precision(
    top1_recall_mean: Dict[int, float],
    top1_recall_std: Dict[int, float],
    top2_recall_mean: Dict[int, float],
    top2_recall_std: Dict[int, float],
    top1_precision_mean: Dict[int, float],
    top1_precision_std: Dict[int, float],
    top2_precision_mean: Dict[int, float],
    top2_precision_std: Dict[int, float],
    output_path: Path = None
):
    """
    Create 1x2 figure with Recall and Precision panels.
    
    Args:
        *_mean, *_std: Mean and std for recall/precision at different k values
        output_path: Path to save the figure
    """
    # Only plot RMM classes (not background)
    class_ids = [0, 1, 2, 3]
    class_labels = [CLASS_LABELS_SHORT[c] for c in class_ids]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    x = np.arange(len(class_labels))
    width = 0.35
    
    # ===== Left Panel (A): Recall =====
    recall1_vals = [top1_recall_mean[c] * 100 for c in class_ids]
    recall1_errs = [top1_recall_std[c] * 100 for c in class_ids]
    recall2_vals = [top2_recall_mean[c] * 100 for c in class_ids]
    recall2_errs = [top2_recall_std[c] * 100 for c in class_ids]
    
    bars1 = ax1.bar(
        x - width/2, recall1_vals, width,
        yerr=recall1_errs,
        label='Top-1 Recall',
        color=COLORS["top1"],
        capsize=4,
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5
    )
    bars2 = ax1.bar(
        x + width/2, recall2_vals, width,
        yerr=recall2_errs,
        label='Top-2 Recall',
        color=COLORS["top2"],
        capsize=4,
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax1.set_xlabel('RMM Class', fontsize=11)
    ax1.set_ylabel('Recall (%)', fontsize=11)
    ax1.set_title('(A) Top-1 vs Top-2 Recall', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_labels, fontsize=10)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_ylim(0, 105)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax1.set_axisbelow(True)
    
    # Add value labels BELOW error bars
    for bar, val, err in zip(bars1, recall1_vals, recall1_errs):
        label_y = val - err
        ax1.annotate(f'{val:.0f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, label_y),
                    xytext=(0, -2),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, fontweight='bold')
    for bar, val, err in zip(bars2, recall2_vals, recall2_errs):
        label_y = val - err
        ax1.annotate(f'{val:.0f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, label_y),
                    xytext=(0, -2),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, fontweight='bold')
    
    # ===== Right Panel (B): Precision =====
    prec1_vals = [top1_precision_mean[c] * 100 for c in class_ids]
    prec1_errs = [top1_precision_std[c] * 100 for c in class_ids]
    prec2_vals = [top2_precision_mean[c] * 100 for c in class_ids]
    prec2_errs = [top2_precision_std[c] * 100 for c in class_ids]
    
    bars3 = ax2.bar(
        x - width/2, prec1_vals, width,
        yerr=prec1_errs,
        label='Top-1 Precision',
        color=COLORS["top1"],
        capsize=4,
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5
    )
    bars4 = ax2.bar(
        x + width/2, prec2_vals, width,
        yerr=prec2_errs,
        label='Top-2 Precision',
        color=COLORS["top2"],
        capsize=4,
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax2.set_xlabel('RMM Class', fontsize=11)
    ax2.set_ylabel('Precision (%)', fontsize=11)
    ax2.set_title('(B) Top-1 vs Top-2 Precision', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_labels, fontsize=10)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_ylim(0, 105)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)
    
    # Add value labels BELOW error bars
    for bar, val, err in zip(bars3, prec1_vals, prec1_errs):
        label_y = val - err
        ax2.annotate(f'{val:.0f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, label_y),
                    xytext=(0, -2),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, fontweight='bold')
    for bar, val, err in zip(bars4, prec2_vals, prec2_errs):
        label_y = val - err
        ax2.annotate(f'{val:.0f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, label_y),
                    xytext=(0, -2),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('Window-level Detection Analysis (V-JEPA)', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    # Save
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved figure: {output_path}")
        
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"Saved PDF: {pdf_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(
    top1_recall_mean: Dict[int, float],
    top2_recall_mean: Dict[int, float],
    top1_precision_mean: Dict[int, float],
    top2_precision_mean: Dict[int, float]
):
    """Print summary tables."""
    print("\n### Recall Summary")
    print("\n| Class | Top-1 | Top-2 | Gap |")
    print("|-------|-------|-------|-----|")
    
    for class_id in range(4):
        name = CLASS_NAMES[class_id]
        t1 = top1_recall_mean[class_id] * 100
        t2 = top2_recall_mean[class_id] * 100
        gap = t2 - t1
        print(f"| {name:<14} | {t1:.1f}% | {t2:.1f}% | +{gap:.1f}% |")
    
    print("\n### Precision Summary")
    print("\n| Class | Top-1 | Top-2 | Gap |")
    print("|-------|-------|-------|-----|")
    
    for class_id in range(4):
        name = CLASS_NAMES[class_id]
        t1 = top1_precision_mean[class_id] * 100
        t2 = top2_precision_mean[class_id] * 100
        gap = t2 - t1
        print(f"| {name:<14} | {t1:.1f}% | {t2:.1f}% | {gap:+.1f}% |")


def save_results_json(
    top1_recall_mean: Dict[int, float],
    top1_recall_std: Dict[int, float],
    top2_recall_mean: Dict[int, float],
    top2_recall_std: Dict[int, float],
    top1_precision_mean: Dict[int, float],
    top1_precision_std: Dict[int, float],
    top2_precision_mean: Dict[int, float],
    top2_precision_std: Dict[int, float],
    output_path: Path
):
    """Save all results as JSON."""
    results = {
        "model": "vjepa_balanced",
        "task": "window_level_detection_5class",
        "per_class": {}
    }
    
    for class_id in range(5):
        results["per_class"][CLASS_NAMES[class_id]] = {
            "top1_recall": {
                "mean": float(top1_recall_mean[class_id]) if not np.isnan(top1_recall_mean.get(class_id, np.nan)) else None,
                "std": float(top1_recall_std[class_id]) if not np.isnan(top1_recall_std.get(class_id, np.nan)) else None
            },
            "top2_recall": {
                "mean": float(top2_recall_mean[class_id]) if not np.isnan(top2_recall_mean.get(class_id, np.nan)) else None,
                "std": float(top2_recall_std[class_id]) if not np.isnan(top2_recall_std.get(class_id, np.nan)) else None
            },
            "top1_precision": {
                "mean": float(top1_precision_mean[class_id]) if not np.isnan(top1_precision_mean.get(class_id, np.nan)) else None,
                "std": float(top1_precision_std[class_id]) if not np.isnan(top1_precision_std.get(class_id, np.nan)) else None
            },
            "top2_precision": {
                "mean": float(top2_precision_mean[class_id]) if not np.isnan(top2_precision_mean.get(class_id, np.nan)) else None,
                "std": float(top2_precision_std[class_id]) if not np.isnan(top2_precision_std.get(class_id, np.nan)) else None
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Top-1 vs Top-2 Recall and Precision figure (Figure 5.2c)"
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
        help="Output figure path (default: insights/figs/fig_5_2c_recall_precision.png)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting, only print summary"
    )
    
    args = parser.parse_args()
    
    # Compute Recall
    print(f"Computing Top-1 recall for model: {args.model}")
    top1_recall_mean, top1_recall_std = compute_cv_topk_recall(args.model, k=1)
    
    print(f"Computing Top-2 recall for model: {args.model}")
    top2_recall_mean, top2_recall_std = compute_cv_topk_recall(args.model, k=2)
    
    # Compute Precision
    print(f"Computing Top-1 precision for model: {args.model}")
    top1_prec_mean, top1_prec_std = compute_cv_topk_precision(args.model, k=1)
    
    print(f"Computing Top-2 precision for model: {args.model}")
    top2_prec_mean, top2_prec_std = compute_cv_topk_precision(args.model, k=2)
    
    # Print summary
    print_summary(top1_recall_mean, top2_recall_mean, top1_prec_mean, top2_prec_mean)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / "fig_5_2c_recall_precision.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    json_path = output_path.parent.parent / "tables" / "fig_5_2c_recall_precision_data.json"
    save_results_json(
        top1_recall_mean, top1_recall_std,
        top2_recall_mean, top2_recall_std,
        top1_prec_mean, top1_prec_std,
        top2_prec_mean, top2_prec_std,
        json_path
    )
    
    # Generate plot
    if not args.no_plot:
        plot_recall_precision(
            top1_recall_mean, top1_recall_std,
            top2_recall_mean, top2_recall_std,
            top1_prec_mean, top1_prec_std,
            top2_prec_mean, top2_prec_std,
            output_path=output_path
        )


if __name__ == "__main__":
    main()
