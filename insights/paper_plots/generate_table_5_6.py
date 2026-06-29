#!/usr/bin/env python3
"""
Table 5.6: Macro F1 Gain (Top-1 vs Top-2) for Best Window-level Models.

Computes Top-1 and Top-2 Macro F1 for the best model from each family:
- RGB: vjepa_balanced
- Pose: posec3d_ce
- Skeleton: stgcnpp_ce_4stream
- Fusion: vjepa_balanced_posec3d_mlp_logp

For Top-2 F1:
- A prediction is "correct" if the GT class is in the Top-2 predictions
- This gives higher recall and potentially different precision characteristics

Usage:
    python generate_table_5_6.py
    python generate_table_5_6.py --output table_5_6_macro_f1_gain.md
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Paths
EVAL_RESULTS_DIR = ACTREG_ROOT / "tal" / "eval_results"
DATAPREP_DIR = ACTREG_ROOT / "dataprep" / "tal"
OUTPUT_DIR = ACTREG_ROOT / "insights" / "tables"

# Best models per family (for main table)
BEST_MODELS = {
    "RGB": {
        "name": "V-JEPA Balanced",
        "dir": "vjepa_balanced"
    },
    "Pose": {
        "name": "PoseC3D (CE)",
        "dir": "posec3d_ce"
    },
    "Skeleton": {
        "name": "STGCN++ (CE)",
        "dir": "stgcnpp_ce_4stream"
    },
    "Fusion": {
        "name": "V-JEPA + PoseC3D (MLP)",
        "dir": "vjepa_balanced_posec3d_mlp_logp"
    }
}

# All available 5-class models (for JSON export)
# Note: Binary models excluded as they have different class structure
ALL_MODELS = {
    # RGB models
    "vjepa": {"family": "RGB", "name": "V-JEPA"},
    "vjepa_balanced": {"family": "RGB", "name": "V-JEPA Balanced"},
    # Pose models
    "posec3d_ce": {"family": "Pose", "name": "PoseC3D (CE)"},
    "posec3d_focal": {"family": "Pose", "name": "PoseC3D (Focal)"},
    "posec3d_ce_balanced": {"family": "Pose", "name": "PoseC3D (CE Balanced)"},
    # Skeleton models
    "stgcnpp_ce_4stream": {"family": "Skeleton", "name": "STGCN++ (CE)"},
    "stgcnpp_focal_4stream": {"family": "Skeleton", "name": "STGCN++ (Focal)"},
    # Fusion models
    "vjepa_posec3d_mlp_logp": {"family": "Fusion", "name": "V-JEPA + PoseC3D (MLP)"},
    "vjepa_stgcnpp_mlp_logp": {"family": "Fusion", "name": "V-JEPA + STGCN++ (MLP)"},
    "vjepa_posec3d_bal_mlp_logp": {"family": "Fusion", "name": "V-JEPA + PoseC3D Bal (MLP)"},
    "vjepa_balanced_posec3d_mlp_logp": {"family": "Fusion", "name": "V-JEPA Bal + PoseC3D (MLP)"},
}

# Class mapping
CLASS_NAMES = {
    0: "hands_flapping",
    1: "jumping",
    2: "rocking",
    3: "spinning",
    4: "background"
}

GT_TO_PRED_CLASS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    -1: 4
}


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


def compute_macro_f1(
    preds_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    n_classes: int = 5
) -> Tuple[float, Dict[int, Dict[str, float]]]:
    """
    Compute standard Top-1 Macro F1.
    
    Args:
        preds_df: DataFrame with predictions
        gt_df: DataFrame with ground truth
        n_classes: Number of classes (5 for window detection)
        
    Returns:
        Tuple of (macro_f1, per_class_metrics)
    """
    merged = gt_df.merge(preds_df, on="window_id", how="inner")
    
    if len(merged) == 0:
        raise ValueError("No matching windows")
    
    score_cols = [f"score_class{i}" for i in range(n_classes)]
    scores = merged[score_cols].values
    
    top1_preds = np.argmax(scores, axis=1)
    gt_classes = merged["primary_label"].map(GT_TO_PRED_CLASS).values
    
    per_class = {}
    
    for class_id in range(n_classes):
        gt_is_this_class = gt_classes == class_id
        pred_is_this_class = top1_preds == class_id
        
        tp = np.sum(gt_is_this_class & pred_is_this_class)
        fp = np.sum(~gt_is_this_class & pred_is_this_class)
        fn = np.sum(gt_is_this_class & ~pred_is_this_class)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn)
        }
    
    # Macro F1 (exclude background for RMM-focused analysis)
    class_f1s = [per_class[c]["f1"] for c in range(4)]  # Only RMM classes
    macro_f1 = np.mean(class_f1s)
    
    return macro_f1, per_class


def compute_topk_accuracy(
    preds_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    k: int = 1,
    n_classes: int = 5
) -> float:
    """
    Compute Top-k accuracy (percentage of windows with GT in top-k predictions).
    
    Args:
        preds_df: DataFrame with predictions
        gt_df: DataFrame with ground truth
        k: Number of top predictions to consider
        n_classes: Number of classes
        
    Returns:
        Top-k accuracy as a float
    """
    merged = gt_df.merge(preds_df, on="window_id", how="inner")
    
    if len(merged) == 0:
        raise ValueError("No matching windows")
    
    score_cols = [f"score_class{i}" for i in range(n_classes)]
    scores = merged[score_cols].values
    
    topk_preds = np.argsort(scores, axis=1)[:, -k:]
    gt_classes = merged["primary_label"].map(GT_TO_PRED_CLASS).values
    
    # Check if GT is in top-k for each window
    correct = np.array([gt in topk for gt, topk in zip(gt_classes, topk_preds)])
    
    return correct.mean()


def compute_cv_macro_f1(model_dir: str) -> Tuple[float, float]:
    """
    Compute Top-1 Macro F1 across CV folds.
    
    Returns:
        Tuple of (mean_f1, std_f1)
    """
    fold_f1s = []
    
    for fold in range(3):
        try:
            preds_df = load_predictions(model_dir, fold)
            gt_df = load_ground_truth(fold)
            macro_f1, _ = compute_macro_f1(preds_df, gt_df)
            fold_f1s.append(macro_f1)
        except FileNotFoundError as e:
            print(f"  Warning: Skipping fold {fold}: {e}")
            continue
    
    if len(fold_f1s) == 0:
        return np.nan, np.nan
    
    return np.mean(fold_f1s), np.std(fold_f1s)


def compute_cv_topk_accuracy(model_dir: str, k: int = 1) -> Tuple[float, float]:
    """
    Compute Top-k accuracy across CV folds.
    
    Returns:
        Tuple of (mean_acc, std_acc)
    """
    fold_accs = []
    
    for fold in range(3):
        try:
            preds_df = load_predictions(model_dir, fold)
            gt_df = load_ground_truth(fold)
            acc = compute_topk_accuracy(preds_df, gt_df, k=k)
            fold_accs.append(acc)
        except FileNotFoundError as e:
            print(f"  Warning: Skipping fold {fold}: {e}")
            continue
    
    if len(fold_accs) == 0:
        return np.nan, np.nan
    
    return np.mean(fold_accs), np.std(fold_accs)


def generate_table(results: Dict) -> str:
    """Generate markdown table from results."""
    lines = [
        "# Table 5.6: Window-level Detection Metrics Comparison",
        "",
        "Comparison of Top-1 Macro F1, Top-1 Accuracy, and Top-2 Accuracy for best window-level detection models.",
        "",
        "| Model Family | Model | Top-1 Macro F1 | Top-1 Acc | Top-2 Acc | Acc Gain |",
        "|--------------|-------|----------------|-----------|-----------|----------|"
    ]
    
    for family in ["RGB", "Pose", "Skeleton", "Fusion"]:
        if family not in results:
            continue
            
        data = results[family]
        model_name = data["model_name"]
        f1_mean = data["top1_f1"]["mean"] * 100
        f1_std = data["top1_f1"]["std"] * 100
        t1_mean = data["top1_acc"]["mean"] * 100
        t1_std = data["top1_acc"]["std"] * 100
        t2_mean = data["top2_acc"]["mean"] * 100
        t2_std = data["top2_acc"]["std"] * 100
        gain = t2_mean - t1_mean
        
        f1_str = f"{f1_mean:.1f}% ± {f1_std:.1f}%"
        t1_str = f"{t1_mean:.1f}% ± {t1_std:.1f}%"
        t2_str = f"{t2_mean:.1f}% ± {t2_std:.1f}%"
        gain_str = f"+{gain:.1f}%"
        
        lines.append(f"| {family} | {model_name} | {f1_str} | {t1_str} | {t2_str} | {gain_str} |")
    
    lines.extend([
        "",
        "**Notes:**",
        "- Top-1 Macro F1: Standard macro-averaged F1 over 4 RMM classes",
        "- Top-1/Top-2 Acc: Percentage of windows where GT class is in top-1/top-2 predictions",
        "- Acc Gain: Improvement from considering top-2 predictions instead of top-1",
        "- Values are mean ± std across CV folds"
    ])
    
    return "\n".join(lines)


def compute_model_metrics(model_dir: str, model_name: str, verbose: bool = True) -> Dict:
    """
    Compute all metrics for a single model.
    
    Returns:
        Dict with model metrics or None if no data available
    """
    try:
        if verbose:
            print(f"  Computing Top-1 Macro F1...")
        f1_mean, f1_std = compute_cv_macro_f1(model_dir)
        
        if np.isnan(f1_mean):
            return None
        
        if verbose:
            print(f"  Computing Top-1 Accuracy...")
        t1_mean, t1_std = compute_cv_topk_accuracy(model_dir, k=1)
        
        if verbose:
            print(f"  Computing Top-2 Accuracy...")
        t2_mean, t2_std = compute_cv_topk_accuracy(model_dir, k=2)
        
        return {
            "model_name": model_name,
            "model_dir": model_dir,
            "top1_f1": {"mean": float(f1_mean), "std": float(f1_std)},
            "top1_acc": {"mean": float(t1_mean), "std": float(t1_std)},
            "top2_acc": {"mean": float(t2_mean), "std": float(t2_std)}
        }
    except Exception as e:
        if verbose:
            print(f"  Error processing model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate Table 5.6: Window-level Detection Metrics"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown path (default: insights/tables/table_5_6_macro_f1_gain.md)"
    )
    
    args = parser.parse_args()
    
    # Results for markdown table (best per family)
    table_results = {}
    
    # Results for JSON (all models)
    all_results = {"best_per_family": {}, "all_models": {}}
    
    # Process best models for table
    print("="*60)
    print("Processing BEST models per family (for table)")
    print("="*60)
    
    for family, model_info in BEST_MODELS.items():
        model_dir = model_info["dir"]
        model_name = model_info["name"]
        
        print(f"\n{family}: {model_name} ({model_dir})")
        
        metrics = compute_model_metrics(model_dir, model_name)
        
        if metrics:
            table_results[family] = metrics
            all_results["best_per_family"][family] = metrics
            
            print(f"  Top-1 Macro F1: {metrics['top1_f1']['mean']*100:.1f}% ± {metrics['top1_f1']['std']*100:.1f}%")
            print(f"  Top-1 Acc: {metrics['top1_acc']['mean']*100:.1f}% ± {metrics['top1_acc']['std']*100:.1f}%")
            print(f"  Top-2 Acc: {metrics['top2_acc']['mean']*100:.1f}% ± {metrics['top2_acc']['std']*100:.1f}%")
            print(f"  Acc Gain: +{(metrics['top2_acc']['mean'] - metrics['top1_acc']['mean'])*100:.1f}%")
        else:
            print(f"  Warning: No valid data for {model_name}")
    
    # Process ALL models for JSON
    print("\n" + "="*60)
    print("Processing ALL models (for JSON)")
    print("="*60)
    
    for model_dir, model_info in ALL_MODELS.items():
        model_name = model_info["name"]
        family = model_info["family"]
        
        print(f"\n{model_dir}: {model_name}")
        
        metrics = compute_model_metrics(model_dir, model_name, verbose=False)
        
        if metrics:
            metrics["family"] = family
            all_results["all_models"][model_dir] = metrics
            print(f"  Top-1 F1: {metrics['top1_f1']['mean']*100:.1f}%, Top-1 Acc: {metrics['top1_acc']['mean']*100:.1f}%, Top-2 Acc: {metrics['top2_acc']['mean']*100:.1f}%")
        else:
            print(f"  No data available")
    
    # Generate markdown table (best per family only)
    md_content = generate_table(table_results)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / "table_5_6_macro_f1_gain.md"
    
    # Save markdown
    with open(output_path, 'w') as f:
        f.write(md_content)
    print(f"\nSaved markdown: {output_path}")
    
    # Save JSON with ALL models
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved JSON (all models): {json_path}")
    
    # Print table
    print("\n" + "="*70)
    print(md_content)


if __name__ == "__main__":
    main()
