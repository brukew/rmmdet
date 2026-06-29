#!/usr/bin/env python3
"""
Figure 5.2: Top-1 vs Top-2 Recall Analysis for Window-level Detection.

Creates a single grouped bar chart showing Top-1 vs Top-2 Recall per RMM class.

Usage:
    python fig_5_2_composite.py
    python fig_5_2_composite.py --output fig_5_2_top1_top2_composite.png
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
    "correct": "#81C784",    # Lighter green (same as top2)
    "recovered": "#FFB74D",  # Lighter orange (same as top1)
    "missed": "#BA68C8",     # Lighter purple
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


def compute_topk_breakdown(
    preds_df: pd.DataFrame,
    gt_df: pd.DataFrame
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-class breakdown: Correct Top-1 | Recovered Top-2 | Missed.
    
    For each class, returns the proportion of windows that fall into each category:
    - correct_top1: GT class = argmax(scores)
    - recovered_top2: GT != Top-1 but GT is in Top-2
    - missed: GT not in Top-2
    
    Args:
        preds_df: DataFrame with window predictions
        gt_df: DataFrame with ground truth
        
    Returns:
        Dict mapping class_id -> {correct_top1, recovered_top2, missed} proportions
    """
    merged = gt_df.merge(preds_df, on="window_id", how="inner")
    
    if len(merged) == 0:
        raise ValueError("No matching windows between predictions and ground truth")
    
    score_cols = [f"score_class{i}" for i in range(5)]
    scores = merged[score_cols].values
    
    # Get Top-1 and Top-2 predictions
    top1_preds = np.argmax(scores, axis=1)
    top2_preds = np.argsort(scores, axis=1)[:, -2:]  # Last 2 columns (top 2)
    
    gt_classes = merged["primary_label"].map(GT_TO_PRED_CLASS).values
    
    breakdown = {}
    for class_id in range(5):
        mask = gt_classes == class_id
        n_total = mask.sum()
        
        if n_total == 0:
            breakdown[class_id] = {
                "correct_top1": np.nan,
                "recovered_top2": np.nan,
                "missed": np.nan
            }
            continue
        
        # Correct Top-1: GT == Top-1 prediction
        correct_top1 = (top1_preds[mask] == class_id).sum()
        
        # GT in Top-2 (includes correct Top-1)
        gt_in_top2 = np.any(top2_preds[mask] == class_id, axis=1)
        
        # Recovered Top-2: GT in Top-2 but not Top-1
        recovered_top2 = gt_in_top2.sum() - correct_top1
        
        # Missed: GT not in Top-2
        missed = n_total - gt_in_top2.sum()
        
        breakdown[class_id] = {
            "correct_top1": correct_top1 / n_total,
            "recovered_top2": recovered_top2 / n_total,
            "missed": missed / n_total
        }
    
    return breakdown


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


def compute_cv_breakdown(model_name: str) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    Compute per-class breakdown across all CV folds.
    
    Returns:
        Tuple of (mean_breakdown, std_breakdown) dictionaries
    """
    fold_breakdowns = []
    
    for fold in range(3):
        try:
            preds_df = load_predictions(model_name, fold)
            gt_df = load_ground_truth(fold)
            breakdown = compute_topk_breakdown(preds_df, gt_df)
            fold_breakdowns.append(breakdown)
        except FileNotFoundError as e:
            print(f"Warning: Skipping fold {fold}: {e}")
            continue
    
    if len(fold_breakdowns) == 0:
        raise ValueError(f"No valid folds found for model: {model_name}")
    
    # Aggregate across folds
    mean_breakdown = {}
    std_breakdown = {}
    
    for class_id in range(5):
        mean_breakdown[class_id] = {}
        std_breakdown[class_id] = {}
        
        for category in ["correct_top1", "recovered_top2", "missed"]:
            values = [b[class_id][category] for b in fold_breakdowns 
                     if not np.isnan(b[class_id].get(category, np.nan))]
            if values:
                mean_breakdown[class_id][category] = np.mean(values)
                std_breakdown[class_id][category] = np.std(values)
            else:
                mean_breakdown[class_id][category] = np.nan
                std_breakdown[class_id][category] = np.nan
    
    return mean_breakdown, std_breakdown


def plot_composite(
    top1_mean: Dict[int, float],
    top1_std: Dict[int, float],
    top2_mean: Dict[int, float],
    top2_std: Dict[int, float],
    breakdown_mean: Dict[int, Dict[str, float]],
    breakdown_std: Dict[int, Dict[str, float]],
    output_path: Path = None
):
    """
    Create single panel grouped bar chart for Top-1 vs Top-2 Recall.
    
    Args:
        top1_mean, top1_std: Top-1 recall statistics
        top2_mean, top2_std: Top-2 recall statistics
        breakdown_mean, breakdown_std: Breakdown statistics (not used in single panel)
        output_path: Path to save the figure
    """
    # Only plot RMM classes (not background)
    class_ids = [0, 1, 2, 3]
    class_labels = [CLASS_LABELS_SHORT[c] for c in class_ids]
    
    # Create single panel figure
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    x = np.arange(len(class_labels))
    width = 0.35
    
    # Prepare data
    top1_vals = [top1_mean[c] * 100 for c in class_ids]
    top1_errs = [top1_std[c] * 100 for c in class_ids]
    top2_vals = [top2_mean[c] * 100 for c in class_ids]
    top2_errs = [top2_std[c] * 100 for c in class_ids]
    
    # Plot bars
    bars1 = ax.bar(
        x - width/2, top1_vals, width,
        yerr=top1_errs,
        label='Top-1 Recall',
        color=COLORS["top1"],
        capsize=4,
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5
    )
    bars2 = ax.bar(
        x + width/2, top2_vals, width,
        yerr=top2_errs,
        label='Top-2 Recall',
        color=COLORS["top2"],
        capsize=4,
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.set_xlabel('RMM Class', fontsize=11)
    ax.set_ylabel('Recall (%)', fontsize=11)
    ax.set_title('Window-level Detection Analysis (V-JEPA)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Add value labels BELOW error bars
    for bar, val, err in zip(bars1, top1_vals, top1_errs):
        label_y = val - err  # Position below the error bar
        ax.annotate(f'{val:.0f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, label_y),
                    xytext=(0, -2),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, fontweight='bold')
    for bar, val, err in zip(bars2, top2_vals, top2_errs):
        label_y = val - err  # Position below the error bar
        ax.annotate(f'{val:.0f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, label_y),
                    xytext=(0, -2),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
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
    top1_mean: Dict[int, float],
    top2_mean: Dict[int, float],
    breakdown_mean: Dict[int, Dict[str, float]]
):
    """Print summary tables."""
    print("\n### Top-1 vs Top-2 Recall Summary")
    print("\n| Class | Top-1 | Top-2 | Gap |")
    print("|-------|-------|-------|-----|")
    
    for class_id in range(4):
        name = CLASS_NAMES[class_id]
        t1 = top1_mean[class_id] * 100
        t2 = top2_mean[class_id] * 100
        gap = t2 - t1
        print(f"| {name:<14} | {t1:.1f}% | {t2:.1f}% | +{gap:.1f}% |")
    
    print("\n### Prediction Outcome Breakdown")
    print("\n| Class | Correct Top-1 | Recovered Top-2 | Missed |")
    print("|-------|---------------|-----------------|--------|")
    
    for class_id in range(4):
        name = CLASS_NAMES[class_id]
        c = breakdown_mean[class_id]["correct_top1"] * 100
        r = breakdown_mean[class_id]["recovered_top2"] * 100
        m = breakdown_mean[class_id]["missed"] * 100
        print(f"| {name:<14} | {c:.1f}% | {r:.1f}% | {m:.1f}% |")


def save_results_json(
    top1_mean: Dict[int, float],
    top1_std: Dict[int, float],
    top2_mean: Dict[int, float],
    top2_std: Dict[int, float],
    breakdown_mean: Dict[int, Dict[str, float]],
    breakdown_std: Dict[int, Dict[str, float]],
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
                "mean": float(top1_mean[class_id]) if not np.isnan(top1_mean[class_id]) else None,
                "std": float(top1_std[class_id]) if not np.isnan(top1_std[class_id]) else None
            },
            "top2_recall": {
                "mean": float(top2_mean[class_id]) if not np.isnan(top2_mean[class_id]) else None,
                "std": float(top2_std[class_id]) if not np.isnan(top2_std[class_id]) else None
            },
            "breakdown": {
                "correct_top1": {
                    "mean": float(breakdown_mean[class_id]["correct_top1"]) if not np.isnan(breakdown_mean[class_id]["correct_top1"]) else None,
                    "std": float(breakdown_std[class_id]["correct_top1"]) if not np.isnan(breakdown_std[class_id]["correct_top1"]) else None
                },
                "recovered_top2": {
                    "mean": float(breakdown_mean[class_id]["recovered_top2"]) if not np.isnan(breakdown_mean[class_id]["recovered_top2"]) else None,
                    "std": float(breakdown_std[class_id]["recovered_top2"]) if not np.isnan(breakdown_std[class_id]["recovered_top2"]) else None
                },
                "missed": {
                    "mean": float(breakdown_mean[class_id]["missed"]) if not np.isnan(breakdown_mean[class_id]["missed"]) else None,
                    "std": float(breakdown_std[class_id]["missed"]) if not np.isnan(breakdown_std[class_id]["missed"]) else None
                }
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate composite Top-1 vs Top-2 analysis figure (Figure 5.2)"
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
        help="Output figure path (default: insights/figs/fig_5_2_top1_top2_composite.png)"
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
    
    # Compute breakdown
    print(f"Computing prediction breakdown for model: {args.model}")
    breakdown_mean, breakdown_std = compute_cv_breakdown(args.model)
    
    # Print summary
    print_summary(top1_mean, top2_mean, breakdown_mean)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / "fig_5_2_top1_top2_composite.png"
    
    # Save JSON results
    json_path = output_path.parent.parent / "tables" / "fig_5_2_composite_data.json"
    save_results_json(top1_mean, top1_std, top2_mean, top2_std, 
                      breakdown_mean, breakdown_std, json_path)
    
    # Generate plot
    if not args.no_plot:
        plot_composite(
            top1_mean, top1_std,
            top2_mean, top2_std,
            breakdown_mean, breakdown_std,
            output_path=output_path
        )


if __name__ == "__main__":
    main()
