#!/usr/bin/env python3
"""
Figure 5.3: TAL Timeline Visualization.

Shows ground truth segments vs ActionFormer and Fusion (V-JEPA+PoseC3D) predictions 
for selected videos.

Usage:
    python fig_5_3_tal_timeline.py
    python fig_5_3_tal_timeline.py --output fig_5_3.png
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Paths
SEGMENTS_FILE = ACTREG_ROOT / "dataprep" / "rmm_segments.csv"
ACTIONFORMER_MULTICLASS = ACTREG_ROOT / "tal" / "eval_results" / "actionformer_vjepa"
FUSION_MODEL_DIR = ACTREG_ROOT / "tal" / "eval_results" / "vjepa_posec3d_mlp_logp"
OUTPUT_DIR = ACTREG_ROOT / "insights" / "figs"

# Class colors for visualization (lighter palette)
CLASS_COLORS = {
    "hands flapping": "#FFB74D",  # Lighter orange
    "jumping": "#81C784",         # Lighter green
    "rocking": "#BA68C8",         # Lighter purple
    "spinning": "#FFF176",        # Lighter yellow
    "background": "#E0E0E0"       # Light gray
}

# Label map
LABEL_MAP_4CLASS = {
    0: "hands flapping",
    1: "jumping",
    2: "rocking",
    3: "spinning",
    4: "background"
}


def load_ground_truth() -> pd.DataFrame:
    """Load ground truth segments."""
    df = pd.read_csv(SEGMENTS_FILE)
    
    # Merge one_hand_flap into hands_flapping
    df["rmm_type_4class"] = df["rmm_type"].apply(
        lambda x: "hands flapping" if x == "one hand flap" else x
    )
    
    # Create normalized video key
    def normalize_key(video_file):
        video_file = video_file.replace("\\", "/")
        parts = video_file.split("/")
        child_id = None
        for part in parts:
            if "AMES_" in part:
                child_id = part.split("AMES_")[-1]
                break
        if child_id is None:
            child_id = parts[0] if parts else "UNKNOWN"
        filename = parts[-1]
        video_stem = Path(filename).stem
        return f"{child_id}_{video_stem}"
    
    df["video_key"] = df["video_file"].apply(normalize_key)
    
    return df


def load_actionformer_predictions(fold: int = 0) -> Dict[str, List[Dict]]:
    """Load ActionFormer multiclass predictions."""
    pred_file = ACTIONFORMER_MULTICLASS / f"fold{fold}" / "predictions.json"
    
    with open(pred_file) as f:
        data = json.load(f)
    
    return data.get("results", data)


def load_fusion_predictions(fold: int = 0) -> pd.DataFrame:
    """Load fusion model window predictions."""
    pred_file = FUSION_MODEL_DIR / f"fold{fold}" / "tal_format_preds.csv"
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Fusion predictions not found: {pred_file}")
    
    return pd.read_csv(pred_file)


def convert_windows_to_segments(
    preds_df: pd.DataFrame,
    score_threshold: float = 0.3,
    merge_gap_sec: float = 1.0
) -> Dict[str, List[Dict]]:
    """
    Convert window-level predictions to segment predictions.
    
    Uses simple postprocessing:
    1. Take argmax of scores for each window
    2. Filter by score threshold
    3. Merge adjacent windows with same class
    
    Args:
        preds_df: DataFrame with window predictions
        score_threshold: Minimum score for a detection
        merge_gap_sec: Maximum gap to merge adjacent segments
        
    Returns:
        Dictionary mapping video_key to list of segment predictions
    """
    predictions = {}
    
    # Extract video key from window_id
    def get_video_key(window_id):
        # Format: VIDEO_KEY__tSTART_END
        parts = window_id.rsplit("__t", 1)
        return parts[0] if len(parts) > 1 else window_id
    
    preds_df = preds_df.copy()
    preds_df["video_key_short"] = preds_df["window_id"].apply(get_video_key)
    
    score_cols = [f"score_class{i}" for i in range(5)]
    
    for video_key, group in preds_df.groupby("video_key_short"):
        group = group.sort_values("start_sec")
        
        video_segments = []
        
        for _, row in group.iterrows():
            scores = [row[col] for col in score_cols]
            pred_class = int(np.argmax(scores))
            max_score = scores[pred_class]
            
            # Skip background (class 4) and low confidence
            if pred_class == 4 or max_score < score_threshold:
                continue
            
            segment = {
                "segment": [row["start_sec"], row["end_sec"]],
                "label": LABEL_MAP_4CLASS[pred_class],
                "score": max_score
            }
            
            # Try to merge with previous segment
            if video_segments and video_segments[-1]["label"] == segment["label"]:
                prev = video_segments[-1]
                if segment["segment"][0] - prev["segment"][1] <= merge_gap_sec:
                    # Merge
                    prev["segment"][1] = segment["segment"][1]
                    prev["score"] = max(prev["score"], segment["score"])
                    continue
            
            video_segments.append(segment)
        
        if video_segments:
            predictions[video_key] = video_segments
    
    return predictions


def filter_predictions(predictions: List[Dict], score_threshold: float = 0.3) -> List[Dict]:
    """Filter predictions by score threshold."""
    return [p for p in predictions if p["score"] >= score_threshold]


def select_videos_for_comparison(
    gt_df: pd.DataFrame,
    actionformer_preds: Dict[str, List[Dict]],
    fusion_preds: Dict[str, List[Dict]],
    score_threshold: float = 0.3
) -> Dict[str, Dict]:
    """
    Select videos where both models produce at least one segment.
    
    Selects 3 videos:
    - single_segment: Video with exactly 1 GT RMM segment
    - multi_same_class: Video with multiple GT segments of same class
    - multi_class: Video with GT segments of different classes
    
    Args:
        gt_df: Ground truth DataFrame
        actionformer_preds: ActionFormer predictions
        fusion_preds: Fusion predictions
        score_threshold: Minimum score threshold
        
    Returns:
        Selected videos dictionary
    """
    # Get video-level GT info
    video_info = {}
    for video_key, group in gt_df.groupby("video_key"):
        segments = []
        class_counts = {}
        
        for _, row in group.iterrows():
            rmm_type = row["rmm_type_4class"]
            if rmm_type not in ["hands flapping", "jumping", "rocking", "spinning"]:
                continue
            
            segments.append({
                "class": rmm_type,
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"]
            })
            class_counts[rmm_type] = class_counts.get(rmm_type, 0) + 1
        
        if segments:
            video_info[video_key] = {
                "segments": segments,
                "n_segments": len(segments),
                "n_classes": len(class_counts),
                "class_counts": class_counts
            }
    
    # Find videos where both models have predictions
    candidates = {
        "single_segment": [],
        "multi_same_class": [],
        "multi_class": []
    }
    
    for video_key, info in video_info.items():
        # Check if both models have predictions for this video
        af_preds = filter_predictions(
            actionformer_preds.get(video_key, []), score_threshold
        )
        fusion_segs = filter_predictions(
            fusion_preds.get(video_key, []), score_threshold
        )
        
        if not af_preds or not fusion_segs:
            continue
        
        # Categorize
        n_segs = info["n_segments"]
        n_classes = info["n_classes"]
        
        candidate = {
            "video_key": video_key,
            "n_segments": n_segs,
            "n_classes": n_classes,
            "class_counts": info["class_counts"],
            "segments": info["segments"],
            "n_af_preds": len(af_preds),
            "n_fusion_preds": len(fusion_segs)
        }
        
        if n_segs == 1:
            candidates["single_segment"].append(candidate)
        elif n_classes == 1 and n_segs > 1:
            candidates["multi_same_class"].append(candidate)
        elif n_classes > 1:
            candidates["multi_class"].append(candidate)
    
    # Select best candidate from each category
    # Prefer videos with more predictions from both models
    # Skip first candidate for single_segment and multi_same_class to get variety
    selected = {}
    
    for category, cands in candidates.items():
        if cands:
            # Sort by sum of predictions from both models
            cands.sort(key=lambda x: x["n_af_preds"] + x["n_fusion_preds"], reverse=True)
            
            # For single_segment and multi_same_class, pick 2nd candidate if available
            if category in ["single_segment", "multi_same_class"] and len(cands) > 1:
                selected[category] = cands[1]
            else:
                selected[category] = cands[0]
    
    return selected


def plot_timeline(
    video_key: str,
    video_duration: float,
    gt_segments: List[Dict],
    actionformer_preds: List[Dict],
    fusion_preds: List[Dict],
    title: str = "",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot a single video's timeline showing GT vs predictions.
    
    Args:
        video_key: Video identifier
        video_duration: Total video duration in seconds
        gt_segments: List of GT segments [{"class": str, "start": float, "end": float}]
        actionformer_preds: ActionFormer predictions
        fusion_preds: Fusion predictions
        title: Plot title
        ax: Matplotlib axes (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    
    row_height = 0.3
    y_positions = {
        "gt": 1.0,
        "actionformer": 0.5,
        "fusion": 0.0
    }
    
    # Plot ground truth
    for seg in gt_segments:
        start, end = seg["start_sec"], seg["end_sec"]
        color = CLASS_COLORS.get(seg["class"], "#757575")
        ax.barh(
            y_positions["gt"], end - start, left=start, height=row_height,
            color=color, edgecolor="black", linewidth=0.5, alpha=0.9
        )
    
    # Plot ActionFormer predictions
    for pred in actionformer_preds:
        start, end = pred["segment"]
        label = pred["label"]
        color = CLASS_COLORS.get(label, "#757575")
        ax.barh(
            y_positions["actionformer"], end - start, left=start, height=row_height,
            color=color, edgecolor="black", linewidth=0.5, alpha=0.7
        )
    
    # Plot Fusion predictions
    for pred in fusion_preds:
        start, end = pred["segment"]
        label = pred["label"]
        color = CLASS_COLORS.get(label, "#757575")
        ax.barh(
            y_positions["fusion"], end - start, left=start, height=row_height,
            color=color, edgecolor="black", linewidth=0.5, alpha=0.7
        )
    
    # Formatting
    ax.set_xlim(0, video_duration)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    
    # Y-axis labels
    yticks = [y_positions["gt"], y_positions["actionformer"], y_positions["fusion"]]
    yticklabels = ["Ground Truth", "ActionFormer", "Fusion"]
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=10)
    ax.set_ylim(-0.3, 1.3)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    
    # Add grid
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    
    return ax


def main():
    parser = argparse.ArgumentParser(
        description="Generate TAL timeline visualization (Figure 5.3)"
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    
    args = parser.parse_args()
    
    # Load data
    print("Loading ground truth segments...")
    gt_df = load_ground_truth()
    
    print("Loading ActionFormer predictions...")
    actionformer_preds = load_actionformer_predictions(args.fold)
    
    print("Loading Fusion predictions...")
    fusion_df = load_fusion_predictions(args.fold)
    fusion_preds = convert_windows_to_segments(
        fusion_df, 
        score_threshold=args.score_threshold,
        merge_gap_sec=1.0
    )
    
    print(f"ActionFormer: {len(actionformer_preds)} videos with predictions")
    print(f"Fusion: {len(fusion_preds)} videos with predictions")
    
    # Select videos
    print("Selecting videos for comparison...")
    selected = select_videos_for_comparison(
        gt_df, actionformer_preds, fusion_preds, args.score_threshold
    )
    
    if len(selected) < 3:
        print(f"Warning: Only found {len(selected)} video categories")
        print("Available categories:", list(selected.keys()))
    
    # Create figure with 3 subplots (one per selected video)
    n_videos = len(selected)
    fig, axes = plt.subplots(n_videos, 1, figsize=(14, 3.5 * n_videos))
    
    if n_videos == 1:
        axes = [axes]
    
    titles = {
        "single_segment": "Single RMM Segment",
        "multi_same_class": "Multiple Segments (Same Class)",
        "multi_class": "Multiple Classes"
    }
    
    for idx, (category, video_info) in enumerate(selected.items()):
        video_key = video_info["video_key"]
        
        # Get video duration from GT
        video_gt = gt_df[gt_df["video_key"] == video_key]
        if "video_duration" in video_gt.columns and len(video_gt) > 0:
            video_duration = video_gt["video_duration"].iloc[0]
        else:
            # Estimate from segments
            max_end = max(s["end_sec"] for s in video_info["segments"])
            video_duration = max_end + 5
        
        # Get predictions
        af_preds = filter_predictions(
            actionformer_preds.get(video_key, []),
            args.score_threshold
        )
        fusion_segs = filter_predictions(
            fusion_preds.get(video_key, []),
            args.score_threshold
        )
        
        # Format GT segments
        gt_segs = [
            {"class": seg["class"], "start_sec": seg["start_sec"], "end_sec": seg["end_sec"]}
            for seg in video_info["segments"]
        ]
        
        # Title without video ID
        title = titles.get(category, category)
        
        # Plot
        plot_timeline(
            video_key, video_duration,
            gt_segs, af_preds, fusion_segs,
            title=title, ax=axes[idx]
        )
    
    # Add overall title and legend
    fig.suptitle("TAL Prediction Timelines", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    # Create legend
    handles = []
    for class_name in ["hands flapping", "jumping", "rocking", "spinning"]:
        handles.append(mpatches.Patch(color=CLASS_COLORS[class_name], 
                                      label=class_name.title()))
    
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    # Save
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "fig_5_3_tal_timeline.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {output_path}")
    
    # Also save PDF
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved PDF: {pdf_path}")
    
    plt.close()
    
    # Save selected videos info
    json_path = output_path.parent.parent / "tables" / "fig_5_3_selected_videos.json"
    with open(json_path, 'w') as f:
        json.dump(selected, f, indent=2)
    print(f"Saved selected videos: {json_path}")


if __name__ == "__main__":
    main()
