#!/usr/bin/env python3
"""
Figure 5.4: Qualitative Failure Analysis for V-JEPA Balanced Window-level Detection.

Analyzes prediction errors including:
- False positives (high-confidence wrong predictions)
- False negatives (missed RMM events)
- Class confusions

Usage:
    python fig_5_4_failure_analysis.py
    python fig_5_4_failure_analysis.py --output fig_5_4.png
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Paths
SEGMENTS_FILE = ACTREG_ROOT / "dataprep" / "rmm_segments.csv"
EVAL_RESULTS_DIR = ACTREG_ROOT / "tal" / "eval_results"
OUTPUT_DIR = ACTREG_ROOT / "insights" / "paper_plots"

# Class names
CLASS_NAMES = {
    0: "hands flapping",
    1: "jumping",
    2: "rocking",
    3: "spinning"
}

CLASS_COLORS = {
    "hands flapping": "#FFB74D",  # Lighter orange
    "jumping": "#81C784",         # Lighter green
    "rocking": "#BA68C8",         # Lighter purple
    "spinning": "#FFF176"         # Lighter yellow
}


def load_ground_truth() -> pd.DataFrame:
    """Load ground truth segments."""
    df = pd.read_csv(SEGMENTS_FILE)
    
    # Merge one_hand_flap into hands_flapping
    df["rmm_type_4class"] = df["rmm_type"].apply(
        lambda x: "hands flapping" if x == "one hand flap" else x
    )
    
    # Filter to 4-class
    rmm_types = {"hands flapping", "jumping", "rocking", "spinning"}
    df = df[df["rmm_type_4class"].isin(rmm_types)].copy()
    
    # Normalize video key
    df["video_key_normalized"] = df["video_file"].apply(normalize_video_key)
    
    return df


def normalize_video_key(video_file: str) -> str:
    """Normalize video file path for matching."""
    return video_file.replace("\\", "/")


def load_predictions(model_name: str = "vjepa_balanced") -> pd.DataFrame:
    """Load predicted segments from all folds."""
    all_preds = []
    
    for fold in range(3):
        pred_file = EVAL_RESULTS_DIR / model_name / f"fold{fold}" / "best_eval" / "pred_segments.csv"
        
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            df["fold"] = fold
            all_preds.append(df)
    
    if not all_preds:
        raise ValueError(f"No predictions found for model: {model_name}")
    
    return pd.concat(all_preds, ignore_index=True)


def compute_tiou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Compute temporal Intersection over Union."""
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)
    
    pred_len = pred_end - pred_start
    gt_len = gt_end - gt_start
    union = pred_len + gt_len - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def analyze_errors(
    preds_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    tiou_threshold: float = 0.3
) -> Dict:
    """
    Analyze prediction errors.
    
    Returns dict with:
    - false_positives: high-confidence predictions that don't match any GT
    - false_negatives: GT segments that weren't detected
    - class_confusions: predictions with correct localization but wrong class
    """
    results = {
        "false_positives": [],
        "false_negatives": [],
        "class_confusions": [],
        "true_positives": []
    }
    
    # Group GT by video
    gt_by_video = defaultdict(list)
    for _, row in gt_df.iterrows():
        gt_by_video[row["video_key_normalized"]].append({
            "class": row["rmm_type_4class"],
            "start": row["start_sec"],
            "end": row["end_sec"] + 1.0  # Convert to half-open
        })
    
    # Track matched GT segments
    matched_gt = defaultdict(set)
    
    # Process predictions
    for _, pred in preds_df.iterrows():
        video_key = pred["video_key"]
        pred_class = CLASS_NAMES.get(pred["class_id"], "unknown")
        pred_start = pred["start_sec"]
        pred_end = pred["end_sec"]
        score = pred["score"]
        
        # Find matching GT
        gt_segments = gt_by_video.get(video_key, [])
        best_match = None
        best_tiou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_segments):
            tiou = compute_tiou(pred_start, pred_end, gt["start"], gt["end"])
            if tiou > best_tiou:
                best_tiou = tiou
                best_match = gt
                best_gt_idx = i
        
        if best_tiou >= tiou_threshold and best_match is not None:
            # Check if class matches
            if pred_class == best_match["class"]:
                results["true_positives"].append({
                    "video_key": video_key,
                    "pred_class": pred_class,
                    "pred_start": pred_start,
                    "pred_end": pred_end,
                    "score": score,
                    "tiou": best_tiou
                })
                matched_gt[video_key].add(best_gt_idx)
            else:
                # Class confusion
                results["class_confusions"].append({
                    "video_key": video_key,
                    "pred_class": pred_class,
                    "gt_class": best_match["class"],
                    "pred_start": pred_start,
                    "pred_end": pred_end,
                    "gt_start": best_match["start"],
                    "gt_end": best_match["end"],
                    "score": score,
                    "tiou": best_tiou
                })
                matched_gt[video_key].add(best_gt_idx)
        else:
            # False positive
            results["false_positives"].append({
                "video_key": video_key,
                "pred_class": pred_class,
                "pred_start": pred_start,
                "pred_end": pred_end,
                "score": score,
                "best_tiou": best_tiou
            })
    
    # Find false negatives (unmatched GT)
    for video_key, gt_segments in gt_by_video.items():
        for i, gt in enumerate(gt_segments):
            if i not in matched_gt.get(video_key, set()):
                results["false_negatives"].append({
                    "video_key": video_key,
                    "gt_class": gt["class"],
                    "gt_start": gt["start"],
                    "gt_end": gt["end"]
                })
    
    return results


def plot_error_distribution(errors: Dict, output_path: Path = None):
    """Create a visualization of error distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error type distribution (pie chart)
    ax = axes[0, 0]
    labels = ["True Positives", "False Positives", "False Negatives", "Class Confusions"]
    sizes = [
        len(errors["true_positives"]),
        len(errors["false_positives"]),
        len(errors["false_negatives"]),
        len(errors["class_confusions"])
    ]
    colors = ["#81C784", "#FFB74D", "#FFF176", "#BA68C8"]  # Lighter green, orange, yellow, purple
    
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Error Type Distribution", fontsize=12, fontweight="bold")
    
    # 2. False positive score distribution
    ax = axes[0, 1]
    fp_scores = [fp["score"] for fp in errors["false_positives"]]
    if fp_scores:
        ax.hist(fp_scores, bins=20, color="#FFB74D", alpha=0.7, edgecolor="black")  # Lighter orange
        ax.axvline(np.mean(fp_scores), color="red", linestyle="--", label=f"Mean: {np.mean(fp_scores):.2f}")
        ax.legend()
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("False Positive Score Distribution", fontsize=12, fontweight="bold")
    
    # 3. False negatives by class
    ax = axes[1, 0]
    fn_by_class = defaultdict(int)
    for fn in errors["false_negatives"]:
        fn_by_class[fn["gt_class"]] += 1
    
    classes = list(CLASS_COLORS.keys())
    counts = [fn_by_class.get(c, 0) for c in classes]
    colors = [CLASS_COLORS[c] for c in classes]
    
    bars = ax.bar(range(len(classes)), counts, color=colors, edgecolor="black")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.title() for c in classes], rotation=15)
    ax.set_ylabel("Count")
    ax.set_title("False Negatives by Class", fontsize=12, fontweight="bold")
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha="center", va="bottom", fontsize=10)
    
    # 4. Class confusion matrix
    ax = axes[1, 1]
    confusion = defaultdict(lambda: defaultdict(int))
    for cc in errors["class_confusions"]:
        confusion[cc["gt_class"]][cc["pred_class"]] += 1
    
    classes = list(CLASS_COLORS.keys())
    matrix = np.zeros((len(classes), len(classes)))
    for i, gt_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = confusion[gt_class][pred_class]
    
    im = ax.imshow(matrix, cmap="Reds")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([c.split()[0].title() for c in classes], rotation=45)
    ax.set_yticklabels([c.split()[0].title() for c in classes])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Ground Truth Class")
    ax.set_title("Class Confusion Matrix (Localized but Misclassified)", fontsize=12, fontweight="bold")
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            if matrix[i, j] > 0:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", 
                       color="white" if matrix[i, j] > matrix.max()/2 else "black")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {output_path}")
        plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        print(f"Saved PDF: {output_path.with_suffix('.pdf')}")
    
    plt.close()


def print_summary(errors: Dict):
    """Print error analysis summary."""
    print("\n" + "="*60)
    print("V-JEPA Balanced Window-level Detection - Error Analysis")
    print("="*60)
    
    total_preds = len(errors["true_positives"]) + len(errors["false_positives"]) + len(errors["class_confusions"])
    total_gt = len(errors["true_positives"]) + len(errors["false_negatives"]) + len(errors["class_confusions"])
    
    print(f"\nTotal predictions: {total_preds}")
    print(f"Total ground truth segments: {total_gt}")
    
    print(f"\n### Error Breakdown:")
    print(f"  - True Positives: {len(errors['true_positives'])}")
    print(f"  - False Positives: {len(errors['false_positives'])}")
    print(f"  - False Negatives: {len(errors['false_negatives'])}")
    print(f"  - Class Confusions: {len(errors['class_confusions'])}")
    
    # High-confidence false positives
    high_conf_fp = [fp for fp in errors["false_positives"] if fp["score"] >= 0.5]
    print(f"\n### High-confidence False Positives (score >= 0.5): {len(high_conf_fp)}")
    
    # Top 5 worst false positives
    if errors["false_positives"]:
        print("\n### Top 5 Highest-confidence False Positives:")
        sorted_fp = sorted(errors["false_positives"], key=lambda x: -x["score"])[:5]
        for i, fp in enumerate(sorted_fp):
            print(f"  {i+1}. {fp['pred_class']} at {fp['pred_start']:.1f}-{fp['pred_end']:.1f}s "
                  f"(score={fp['score']:.3f}, video={fp['video_key'][:30]}...)")
    
    # Class confusion breakdown
    if errors["class_confusions"]:
        print(f"\n### Class Confusion Breakdown:")
        conf_counts = defaultdict(int)
        for cc in errors["class_confusions"]:
            key = f"{cc['gt_class']} -> {cc['pred_class']}"
            conf_counts[key] += 1
        
        for pair, count in sorted(conf_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  - {pair}: {count}")


def save_results_json(errors: Dict, output_path: Path):
    """Save error analysis results to JSON."""
    # Convert to serializable format
    results = {
        "summary": {
            "true_positives": len(errors["true_positives"]),
            "false_positives": len(errors["false_positives"]),
            "false_negatives": len(errors["false_negatives"]),
            "class_confusions": len(errors["class_confusions"])
        },
        "high_confidence_fps": [
            fp for fp in errors["false_positives"] if fp["score"] >= 0.5
        ][:20],
        "class_confusion_counts": {}
    }
    
    conf_counts = defaultdict(int)
    for cc in errors["class_confusions"]:
        key = f"{cc['gt_class']} -> {cc['pred_class']}"
        conf_counts[key] += 1
    results["class_confusion_counts"] = dict(conf_counts)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate qualitative failure analysis (Figure 5.4)"
    )
    parser.add_argument("--model", type=str, default="vjepa_balanced")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tiou", type=float, default=0.3)
    
    args = parser.parse_args()
    
    # Load data
    print("Loading ground truth...")
    gt_df = load_ground_truth()
    print(f"Loaded {len(gt_df)} ground truth segments")
    
    print("Loading predictions...")
    preds_df = load_predictions(args.model)
    print(f"Loaded {len(preds_df)} predictions from {preds_df['fold'].nunique()} folds")
    
    # Analyze errors
    print("Analyzing errors...")
    errors = analyze_errors(preds_df, gt_df, args.tiou)
    
    # Print summary
    print_summary(errors)
    
    # Generate plots
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "fig_5_4_failure_analysis.png"
    plot_error_distribution(errors, output_path)
    
    # Save JSON results
    save_results_json(errors, output_path.with_suffix(".json"))


if __name__ == "__main__":
    main()
