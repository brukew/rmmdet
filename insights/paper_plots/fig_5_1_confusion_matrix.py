#!/usr/bin/env python3
"""
Figure 5.1: 4-Class Classification Confusion Matrix.

Generates a confusion matrix for the best fusion model (3-Way MLP)
with a modified plasma colormap that has a lighter purple at the low end.

Usage:
    python fig_5_1_confusion_matrix.py
    python fig_5_1_confusion_matrix.py --output fig_5_1_confusion_matrix.png
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

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Paths
FUSION_RUNS_DIR = ACTREG_ROOT / "fusion" / "runs" / "4class_cv_3way"
OUTPUT_DIR = ACTREG_ROOT / "insights" / "figs"

# Class names for 4-class task (full names)
CLASS_NAMES = ["Hands Flapping", "Jumping", "Rocking", "Spinning"]


def get_cmap():
    """Get the colormap for confusion matrix visualization."""
    return "Blues"


def load_predictions(fold: int) -> pd.DataFrame:
    """Load clip-level predictions for a fold."""
    pred_file = FUSION_RUNS_DIR / f"fold_{fold}" / "predictions_clip.csv"
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_file}")
    
    return pd.read_csv(pred_file)


def compute_confusion_matrix(preds_df: pd.DataFrame, n_classes: int = 4) -> np.ndarray:
    """
    Compute confusion matrix from predictions DataFrame.
    
    Args:
        preds_df: DataFrame with 'label_id' and 'pred_id' columns
        n_classes: Number of classes
        
    Returns:
        Confusion matrix as numpy array [n_classes, n_classes]
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for _, row in preds_df.iterrows():
        gt = int(row['label_id'])
        pred = int(row['pred_id'])
        if 0 <= gt < n_classes and 0 <= pred < n_classes:
            cm[gt, pred] += 1
    
    return cm


def aggregate_confusion_matrices() -> np.ndarray:
    """
    Aggregate confusion matrices across all CV folds.
    
    Returns:
        Summed confusion matrix across folds
    """
    total_cm = None
    
    for fold in range(3):
        try:
            preds_df = load_predictions(fold)
            cm = compute_confusion_matrix(preds_df)
            
            if total_cm is None:
                total_cm = cm
            else:
                total_cm += cm
                
        except FileNotFoundError as e:
            print(f"Warning: Skipping fold {fold}: {e}")
            continue
    
    if total_cm is None:
        raise ValueError("No valid folds found")
    
    return total_cm


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path = None,
    normalize: bool = True
):
    """
    Plot confusion matrix with modified plasma colormap.
    
    Args:
        cm: Confusion matrix (counts)
        output_path: Path to save figure
        normalize: Whether to normalize by row (show percentages)
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    # Use classic Blues colormap
    cmap = get_cmap()
    
    # Normalize if requested
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / row_sums * 100
        fmt = '.1f'
        vmax = 100
    else:
        cm_norm = cm.astype(float)
        fmt = 'd'
        vmax = None
    
    # Plot heatmap with full class names
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        vmin=0,
        vmax=vmax,
        cbar_kws={'label': 'Percentage (%)' if normalize else 'Count'},
        annot_kws={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title('4-Class Classification Confusion Matrix\n(3-Way Fusion)', 
                 fontsize=13, fontweight='bold')
    
    # Rotate x-axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Calculate and display accuracy
    accuracy = np.diag(cm).sum() / cm.sum() * 100
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved figure: {output_path}")
        
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"Saved PDF: {pdf_path}")
    else:
        plt.show()
    
    plt.close()
    
    return accuracy


def print_summary(cm: np.ndarray):
    """Print summary statistics."""
    accuracy = np.diag(cm).sum() / cm.sum() * 100
    
    print("\n### Confusion Matrix Summary")
    print(f"\nOverall Accuracy: {accuracy:.1f}%")
    
    # Per-class recall (diagonal)
    print("\n### Per-class Recall")
    print("\n| Class | Recall |")
    print("|-------|--------|")
    
    class_names = ["Hands Flapping", "Jumping", "Rocking", "Spinning"]
    for i, name in enumerate(class_names):
        recall = cm[i, i] / cm[i].sum() * 100
        print(f"| {name:<14} | {recall:.1f}% |")


def save_results_json(cm: np.ndarray, output_path: Path):
    """Save confusion matrix data to JSON."""
    results = {
        "model": "3-way_fusion",
        "task": "4-class_classification",
        "confusion_matrix": cm.tolist(),
        "class_names": ["hands_flapping", "jumping", "rocking", "spinning"],
        "accuracy": float(np.diag(cm).sum() / cm.sum()),
        "per_class_recall": [float(cm[i, i] / cm[i].sum()) for i in range(4)],
        "per_class_precision": [float(cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() > 0 else 0.0 
                                for i in range(4)]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 4-class confusion matrix (Figure 5.1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path (default: insights/figs/fig_5_1_confusion_matrix.png)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Show raw counts instead of percentages"
    )
    
    args = parser.parse_args()
    
    print("Aggregating confusion matrices across folds...")
    cm = aggregate_confusion_matrices()
    
    # Print summary
    print_summary(cm)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / "fig_5_1_confusion_matrix.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_path.parent.parent / "tables" / "fig_5_1_confusion_matrix.json"
    save_results_json(cm, json_path)
    
    # Generate plot
    plot_confusion_matrix(
        cm,
        output_path=output_path,
        normalize=not args.no_normalize
    )


if __name__ == "__main__":
    main()
