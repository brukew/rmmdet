#!/usr/bin/env python3
"""
Figure 5.2b: Top-1 vs Top-2 Confusion Matrices for V-JEPA Balanced.

Creates a 1x2 figure showing:
- Left: Top-1 confusion matrix (standard argmax predictions)
- Right: Top-2 confusion matrix (prediction = GT if GT in top-2, else top-1)

Usage:
    python fig_5_2b_confusion_matrices.py
    python fig_5_2b_confusion_matrices.py --output fig_5_2b_confusion_matrices.png
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Paths
EVAL_RESULTS_DIR = ACTREG_ROOT / "tal" / "eval_results"
DATAPREP_DIR = ACTREG_ROOT / "dataprep" / "tal"
OUTPUT_DIR = ACTREG_ROOT / "insights" / "figs"

# Class names (5-class including background)
CLASS_NAMES = ["Hands\nFlapping", "Jumping", "Rocking", "Spinning", "Background"]
CLASS_NAMES_SHORT = ["HF", "J", "R", "S", "BG"]

# Map GT primary_label to 5-class prediction index
GT_TO_PRED_CLASS = {
    0: 0,  # hands_flapping
    1: 1,  # jumping
    2: 2,  # rocking
    3: 3,  # spinning
    -1: 4  # background
}

# Colors - use classic Blues colormap
CMAP = "Blues"


def load_predictions(model_name: str, fold: int) -> pd.DataFrame:
    """Load window-level predictions from tal_format_preds.csv."""
    pred_file = EVAL_RESULTS_DIR / model_name / f"fold{fold}" / "tal_format_preds.csv"
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_file}")
    
    return pd.read_csv(pred_file)


def load_ground_truth(fold: int) -> pd.DataFrame:
    """Load ground truth window labels from the splits CSV."""
    gt_file = DATAPREP_DIR / "splits_cv_4class" / f"fold_{fold}_val_windows.csv"
    
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_file}")
    
    df = pd.read_csv(gt_file)
    df["labels"] = df["labels"].apply(json.loads)
    
    return df


def compute_confusion_matrices(
    preds_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    n_classes: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Top-1 and Top-2 confusion matrices.
    
    Top-1 CM: Standard confusion matrix using argmax predictions
    Top-2 CM: "Relaxed" CM where prediction = GT if GT is in top-2, else top-1
    
    Args:
        preds_df: DataFrame with predictions
        gt_df: DataFrame with ground truth
        n_classes: Number of classes
        
    Returns:
        Tuple of (top1_cm, top2_cm) as numpy arrays
    """
    merged = gt_df.merge(preds_df, on="window_id", how="inner")
    
    if len(merged) == 0:
        raise ValueError("No matching windows")
    
    score_cols = [f"score_class{i}" for i in range(n_classes)]
    scores = merged[score_cols].values
    
    # Get predictions
    top1_preds = np.argmax(scores, axis=1)
    top2_preds = np.argsort(scores, axis=1)[:, -2:]  # Last 2 columns
    
    gt_classes = merged["primary_label"].map(GT_TO_PRED_CLASS).values
    
    # Top-1 confusion matrix
    top1_cm = np.zeros((n_classes, n_classes), dtype=int)
    for gt, pred in zip(gt_classes, top1_preds):
        top1_cm[gt, pred] += 1
    
    # Top-2 "relaxed" confusion matrix
    # If GT is in top-2, count as correct (on diagonal)
    # Otherwise, count as top-1 prediction
    top2_cm = np.zeros((n_classes, n_classes), dtype=int)
    for i, (gt, t1_pred, t2_preds) in enumerate(zip(gt_classes, top1_preds, top2_preds)):
        if gt in t2_preds:
            # GT is in top-2, count on diagonal (correct)
            top2_cm[gt, gt] += 1
        else:
            # GT not in top-2, use top-1 prediction
            top2_cm[gt, t1_pred] += 1
    
    return top1_cm, top2_cm


def aggregate_cv_confusion_matrices(model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate confusion matrices across CV folds.
    
    Returns:
        Tuple of (top1_cm, top2_cm) summed across folds
    """
    top1_total = None
    top2_total = None
    
    for fold in range(3):
        try:
            preds_df = load_predictions(model_name, fold)
            gt_df = load_ground_truth(fold)
            top1_cm, top2_cm = compute_confusion_matrices(preds_df, gt_df)
            
            if top1_total is None:
                top1_total = top1_cm
                top2_total = top2_cm
            else:
                top1_total += top1_cm
                top2_total += top2_cm
                
        except FileNotFoundError as e:
            print(f"Warning: Skipping fold {fold}: {e}")
            continue
    
    if top1_total is None:
        raise ValueError(f"No valid folds found for model: {model_name}")
    
    return top1_total, top2_total


def plot_confusion_matrices(
    top1_cm: np.ndarray,
    top2_cm: np.ndarray,
    output_path: Path = None,
    normalize: bool = True
):
    """
    Create 1x2 figure with Top-1 and Top-2 confusion matrices.
    
    Args:
        top1_cm: Top-1 confusion matrix (counts)
        top2_cm: Top-2 confusion matrix (counts)
        output_path: Path to save figure
        normalize: Whether to normalize by row (show percentages)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Normalize if requested
    if normalize:
        row_sums_1 = top1_cm.sum(axis=1, keepdims=True)
        row_sums_2 = top2_cm.sum(axis=1, keepdims=True)
        top1_norm = top1_cm / row_sums_1 * 100
        top2_norm = top2_cm / row_sums_2 * 100
        fmt = '.1f'
        vmax = 100
    else:
        top1_norm = top1_cm.astype(float)
        top2_norm = top2_cm.astype(float)
        fmt = 'd'
        vmax = None
    
    # Plot Top-1 CM
    sns.heatmap(
        top1_norm, 
        annot=True, 
        fmt=fmt,
        cmap=CMAP,
        xticklabels=CLASS_NAMES_SHORT,
        yticklabels=CLASS_NAMES_SHORT,
        ax=ax1,
        vmin=0,
        vmax=vmax,
        cbar_kws={'label': 'Percentage (%)' if normalize else 'Count'}
    )
    ax1.set_xlabel('Predicted Class', fontsize=11)
    ax1.set_ylabel('True Class', fontsize=11)
    ax1.set_title('(A) Top-1 Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Calculate accuracy (for JSON, not displayed)
    top1_acc = np.diag(top1_cm).sum() / top1_cm.sum() * 100
    
    # Plot Top-2 CM
    sns.heatmap(
        top2_norm,
        annot=True,
        fmt=fmt,
        cmap=CMAP,
        xticklabels=CLASS_NAMES_SHORT,
        yticklabels=CLASS_NAMES_SHORT,
        ax=ax2,
        vmin=0,
        vmax=vmax,
        cbar_kws={'label': 'Percentage (%)' if normalize else 'Count'}
    )
    ax2.set_xlabel('Predicted Class', fontsize=11)
    ax2.set_ylabel('True Class', fontsize=11)
    ax2.set_title('(B) Top-2 Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Calculate accuracy (for JSON, not displayed)
    top2_acc = np.diag(top2_cm).sum() / top2_cm.sum() * 100
    
    # Add class name legend
    legend_text = "HF=Hands Flapping, J=Jumping, R=Rocking, S=Spinning, BG=Background"
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=9, style='italic')
    
    plt.suptitle('Window-level Detection Confusion Matrices (V-JEPA)', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {output_path}")
        
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved PDF: {pdf_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(top1_cm: np.ndarray, top2_cm: np.ndarray):
    """Print summary statistics."""
    top1_acc = np.diag(top1_cm).sum() / top1_cm.sum() * 100
    top2_acc = np.diag(top2_cm).sum() / top2_cm.sum() * 100
    
    print("\n### Confusion Matrix Summary")
    print(f"\nTop-1 Overall Accuracy: {top1_acc:.1f}%")
    print(f"Top-2 Overall Accuracy: {top2_acc:.1f}%")
    print(f"Accuracy Gain: +{top2_acc - top1_acc:.1f}%")
    
    # Per-class accuracy
    print("\n### Per-class Recall (diagonal)")
    print("\n| Class | Top-1 Recall | Top-2 Recall | Gain |")
    print("|-------|--------------|--------------|------|")
    
    class_names = ["Hands Flapping", "Jumping", "Rocking", "Spinning", "Background"]
    for i, name in enumerate(class_names):
        t1_recall = top1_cm[i, i] / top1_cm[i].sum() * 100
        t2_recall = top2_cm[i, i] / top2_cm[i].sum() * 100
        gain = t2_recall - t1_recall
        print(f"| {name:<14} | {t1_recall:.1f}% | {t2_recall:.1f}% | +{gain:.1f}% |")


def save_results_json(
    top1_cm: np.ndarray,
    top2_cm: np.ndarray,
    output_path: Path
):
    """Save confusion matrices to JSON."""
    results = {
        "model": "vjepa_balanced",
        "top1_confusion_matrix": top1_cm.tolist(),
        "top2_confusion_matrix": top2_cm.tolist(),
        "class_names": ["hands_flapping", "jumping", "rocking", "spinning", "background"],
        "top1_accuracy": float(np.diag(top1_cm).sum() / top1_cm.sum()),
        "top2_accuracy": float(np.diag(top2_cm).sum() / top2_cm.sum()),
        "per_class_recall": {
            "top1": [float(top1_cm[i, i] / top1_cm[i].sum()) for i in range(5)],
            "top2": [float(top2_cm[i, i] / top2_cm[i].sum()) for i in range(5)]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Top-1 vs Top-2 confusion matrices (Figure 5.2b)"
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
        help="Output figure path (default: insights/figs/fig_5_2b_confusion_matrices.png)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Show raw counts instead of percentages"
    )
    
    args = parser.parse_args()
    
    print(f"Computing confusion matrices for model: {args.model}")
    top1_cm, top2_cm = aggregate_cv_confusion_matrices(args.model)
    
    # Print summary
    print_summary(top1_cm, top2_cm)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / "fig_5_2b_confusion_matrices.png"
    
    # Save JSON
    json_path = output_path.parent.parent / "tables" / "fig_5_2b_confusion_matrices.json"
    save_results_json(top1_cm, top2_cm, json_path)
    
    # Generate plot
    plot_confusion_matrices(
        top1_cm, top2_cm,
        output_path=output_path,
        normalize=not args.no_normalize
    )


if __name__ == "__main__":
    main()
