#!/usr/bin/env python3
"""
Video Example Analysis Script.

Analyzes clip-level, window-level, and TAL predictions for 10 specific videos
to understand model behavior on interesting cases (rocking toys, multi-label,
False_in_Context, etc.).

Usage:
    python video_example_analysis.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ACTREG_ROOT = Path(__file__).parent.parent.parent

# Ground truth
SEGMENTS_CSV = ACTREG_ROOT / "dataprep" / "rmm_segments.csv"
TAL_SPLITS_DIR = ACTREG_ROOT / "dataprep" / "tal" / "splits_cv_4class"

# Clip-level predictions
FUSION_3WAY_CLIP_DIR = ACTREG_ROOT / "fusion" / "runs" / "4class_cv_3way"
FUSION_MLP_CLIP_DIR = ACTREG_ROOT / "fusion" / "runs" / "4class_cv_mlp"
STGCNPP_CLIP_DIR = ACTREG_ROOT / "pyskl" / "work_dirs" / "stgcnpp" / "cv" / "4class_conf04_weighted_4stream"
POSEC3D_CLIP_DIR = ACTREG_ROOT / "pyskl" / "work_dirs" / "posec3d" / "cv" / "4class_conf04"  # non-weighted
VJEPA_CLIP_DIR = ACTREG_ROOT / "v-jepa" / "runs" / "vjepa2_rmm_cv" / "f64_lr1e-5_bs1_acc8_ep20_crop_4cls"

# Window-level TAL predictions
TAL_EVAL_DIR = ACTREG_ROOT / "tal" / "eval_results"

# ActionFormer predictions
OPENTAD_DIR = ACTREG_ROOT / "OpenTAD" / "exps" / "sails_rmm"

# Output
OUTPUT_JSON = ACTREG_ROOT / "insights" / "tables" / "video_example_analysis.json"
OUTPUT_MD = ACTREG_ROOT / "insights" / "writeup" / "video_example_analysis.md"

# Class names
CLASS_NAMES_4 = ["hands_flapping", "jumping", "rocking", "spinning"]
CLASS_NAMES_5 = ["hands_flapping", "jumping", "rocking", "spinning", "background"]


# -----------------------------------------------------------------------------
# Video definitions
# -----------------------------------------------------------------------------

@dataclass
class VideoInfo:
    """Information about a target video for analysis."""
    name: str  # Short descriptive name
    filename: str  # Original filename
    video_key_pattern: str  # Pattern to match in predictions (child_id + filename stem)
    user_description: str  # What the user described
    gt_video_id: str  # video_id in rmm_segments.csv
    val_folds: List[int] = field(default_factory=list)  # Folds where this is val data


VIDEOS = [
    VideoInfo(
        name="rocking_toy",
        filename="12927071_1683482288587213_48606419_n.mp4",
        video_key_pattern="S7V5F9U7N6_12927071",
        user_description="rocking toy",
        gt_video_id="S7V5F9U7N6_unknown_33",
        val_folds=[2],
    ),
    VideoInfo(
        name="spinning_long",
        filename="IMG_4359.MOV",
        video_key_pattern="S2C4T1Y7V7_IMG_4359",
        user_description="long term spinning",
        gt_video_id="S2C4T1Y7V7_36_month_4",
        val_folds=[0],
    ),
    VideoInfo(
        name="spinning_fan",
        filename="06-29-2021 (7).mov",
        video_key_pattern="J3J0V4T8C3_06-29-2021 (7)",
        user_description="child spinning fan (NOT IN GT)",
        gt_video_id="",  # Not annotated
        val_folds=[],
    ),
    VideoInfo(
        name="waving_flag",
        filename="0722221810.mp4",
        video_key_pattern="G7P6U6U2J5_0722221810",
        user_description="waving flag (False_in_Context)",
        gt_video_id="G7P6U6U2J5_36_month_200",
        val_folds=[0],
    ),
    VideoInfo(
        name="lamp_cord",
        filename="3-1-19.MOV",
        video_key_pattern="M2N1C8G1C8_3-1-19",
        user_description="trying to grab lamp cord (False_in_Context)",
        gt_video_id="M2N1C8G1C8_36_month_225",
        val_folds=[0],
    ),
    VideoInfo(
        name="splashing_water",
        filename="5.4.19 2nd clip.MP4",
        video_key_pattern="A2M5H0V6E3_5.4.19 2nd clip",
        user_description="splashing water",
        gt_video_id="A2M5H0V6E3_14_month_238",
        val_folds=[2],
    ),
    VideoInfo(
        name="trampoline",
        filename="04 28 2017.mov",
        video_key_pattern="L4M1H7J7G3_04 28 2017",
        user_description="trampoline (1-sec windows)",
        gt_video_id="L4M1H7J7G3_14_month_313",
        val_folds=[0],
    ),
    VideoInfo(
        name="game_jumping",
        filename="08.09.2022.MP4",
        video_key_pattern="W2E0Z9B7Z8_08.09.2022",
        user_description="game with jumping (False_in_Context)",
        gt_video_id="W2E0Z9B7Z8_36_month_331",
        val_folds=[1],
    ),
    VideoInfo(
        name="jumping_flapping_overlap",
        filename="IMG_4620.MOV",
        video_key_pattern="S2C4T1Y7V7_IMG_4620",
        user_description="jumping and arm flapping overlap",
        gt_video_id="S2C4T1Y7V7_36_month_8",
        val_folds=[0],
    ),
    VideoInfo(
        name="subtle_rocking",
        filename="20181015_094544.mp4",
        video_key_pattern="G4J8F6F4X0_20181015_094544",
        user_description="subtle rocking",
        gt_video_id="G4J8F6F4X0_14_month_103",
        val_folds=[2],
    ),
]


# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------

def load_gt_segments(video_id: str) -> List[Dict]:
    """
    Load ground truth segments for a video from rmm_segments.csv.
    
    Handles 1-second window convention: if start==end, expand to [start, start+1].
    """
    if not video_id:
        return []
    
    df = pd.read_csv(SEGMENTS_CSV)
    video_df = df[df["video_id"] == video_id].copy()
    
    if len(video_df) == 0:
        return []
    
    segments = []
    for _, row in video_df.iterrows():
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        duration = float(row["duration"])
        
        # Handle 1-second window convention
        if start == end or duration == 0:
            end = start + 1.0
            duration = 1.0
        
        segments.append({
            "segment_id": row["segment_id"],
            "rmm_type": row["rmm_type"],
            "start_sec": start,
            "end_sec": end,
            "duration": duration,
            "quality_rating": row.get("quality_rating"),
            "annotator_label": row.get("annotator_label"),
        })
    
    return segments


def load_clip_predictions_fusion_3way(video_info: VideoInfo, fold: int) -> Optional[Dict]:
    """Load 3-way MLP fusion (V-JEPA2 + PoseC3D + STGCN++) clip-level predictions."""
    pred_path = FUSION_3WAY_CLIP_DIR / f"fold_{fold}" / "predictions_clip.csv"
    if not pred_path.exists():
        return None
    
    df = pd.read_csv(pred_path)
    video_df = df[df["segment_id"].str.startswith(video_info.gt_video_id)]
    
    if len(video_df) == 0:
        return None
    
    return {
        "n_clips": len(video_df),
        "clips": [
            {
                "segment_id": row["segment_id"],
                "label": row["label_name"],
                "pred": row["pred_name"],
                "correct": row["label_name"] == row["pred_name"],
                "scores": {
                    CLASS_NAMES_4[i]: float(row[f"fused_score_class{i}"])
                    for i in range(4)
                },
            }
            for _, row in video_df.iterrows()
        ],
    }


def load_clip_predictions_vjepa(video_info: VideoInfo, fold: int) -> Optional[Dict]:
    """Load V-JEPA2 + SAM3 crop clip-level predictions."""
    pred_path = VJEPA_CLIP_DIR / f"fold_{fold}" / "clip_level_preds.csv"
    if not pred_path.exists():
        return None
    
    df = pd.read_csv(pred_path)
    video_df = df[df["segment_id"].str.startswith(video_info.gt_video_id)]
    
    if len(video_df) == 0:
        return None
    
    return {
        "n_clips": len(video_df),
        "clips": [
            {
                "segment_id": row["segment_id"],
                "label": row["label_name"],
                "pred": row["pred_name"],
                "correct": row["label_name"] == row["pred_name"],
                "scores": {
                    CLASS_NAMES_4[i]: float(row[f"score_class{i}"])
                    for i in range(4)
                },
            }
            for _, row in video_df.iterrows()
        ],
    }


def load_clip_predictions_stgcnpp(video_info: VideoInfo, fold: int) -> Optional[Dict]:
    """Load STGCN++ clip-level predictions for a video."""
    pred_path = STGCNPP_CLIP_DIR / f"fold{fold}" / "predictions_clip.csv"
    if not pred_path.exists():
        return None
    
    df = pd.read_csv(pred_path)
    video_df = df[df["segment_id"].str.startswith(video_info.gt_video_id)]
    
    if len(video_df) == 0:
        return None
    
    return {
        "n_clips": len(video_df),
        "clips": [
            {
                "segment_id": row["segment_id"],
                "label": row["true_class"],
                "pred": row["pred_class"],
                "correct": row["true_class"] == row["pred_class"],
                "scores": {
                    CLASS_NAMES_4[i]: float(row[f"score_class{i}"])
                    for i in range(4)
                },
            }
            for _, row in video_df.iterrows()
        ],
    }


def load_clip_predictions_posec3d(video_info: VideoInfo, fold: int) -> Optional[Dict]:
    """Load PoseC3D (non-weighted) clip-level predictions for a video."""
    pred_path = POSEC3D_CLIP_DIR / f"fold{fold}" / "eval_val" / "predictions_clip.csv"
    if not pred_path.exists():
        return None
    
    df = pd.read_csv(pred_path)
    video_df = df[df["segment_id"].str.startswith(video_info.gt_video_id)]
    
    if len(video_df) == 0:
        return None
    
    return {
        "n_clips": len(video_df),
        "clips": [
            {
                "segment_id": row["segment_id"],
                "label": row["true_class"],
                "pred": row["pred_class"],
                "correct": row["true_class"] == row["pred_class"],
                "scores": {
                    CLASS_NAMES_4[i]: float(row[f"score_class{i}"])
                    for i in range(4)
                },
            }
            for _, row in video_df.iterrows()
        ],
    }


def load_clip_predictions_generic(video_info: VideoInfo, pred_path: Path, score_prefix: str = "score_class") -> Optional[Dict]:
    """Generic loader for pyskl-style clip predictions."""
    if not pred_path.exists():
        return None
    
    df = pd.read_csv(pred_path)
    video_df = df[df["segment_id"].str.startswith(video_info.gt_video_id)]
    
    if len(video_df) == 0:
        return None
    
    return {
        "n_clips": len(video_df),
        "clips": [
            {
                "segment_id": row["segment_id"],
                "label": row["true_class"],
                "pred": row["pred_class"],
                "correct": row["true_class"] == row["pred_class"],
                "scores": {
                    CLASS_NAMES_4[i]: float(row[f"{score_prefix}{i}"])
                    for i in range(4)
                },
            }
            for _, row in video_df.iterrows()
        ],
    }


def load_window_predictions(video_info: VideoInfo, model_name: str, fold: int) -> Optional[Dict]:
    """Load window-level TAL predictions for a video."""
    model_dirs = {
        "vjepa_balanced": TAL_EVAL_DIR / "vjepa_balanced",
        "stgcnpp_ce": TAL_EVAL_DIR / "stgcnpp_ce_4stream",
        "posec3d_focal": TAL_EVAL_DIR / "posec3d_focal",
        "fusion_vjepa_posec3d": TAL_EVAL_DIR / "vjepa_posec3d_mlp_logp",
    }
    
    model_dir = model_dirs.get(model_name)
    if model_dir is None:
        return None
    
    # Try both possible filenames
    pred_path = model_dir / f"fold{fold}" / "tal_format_preds.csv"
    if not pred_path.exists():
        pred_path = model_dir / f"fold{fold}" / "fused_tal_format_preds.csv"
    if not pred_path.exists():
        return None
    
    df = pd.read_csv(pred_path)
    # Match by video_key pattern
    video_df = df[df["window_id"].str.startswith(video_info.video_key_pattern)]
    
    if len(video_df) == 0:
        return None
    
    score_cols = [f"score_class{i}" for i in range(5)]
    
    windows = []
    for _, row in video_df.iterrows():
        scores = {CLASS_NAMES_5[i]: float(row[score_cols[i]]) for i in range(5)}
        pred_id = int(np.argmax([row[c] for c in score_cols]))
        windows.append({
            "window_id": row["window_id"],
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "pred_class": CLASS_NAMES_5[pred_id],
            "pred_score": float(row[score_cols[pred_id]]),
            "scores": scores,
        })
    
    return {
        "n_windows": len(windows),
        "windows": windows,
    }


def load_actionformer_predictions(video_info: VideoInfo, variant: str, fold: int) -> Optional[Dict]:
    """Load ActionFormer segment predictions for a video."""
    if variant == "multiclass":
        result_path = OPENTAD_DIR / f"actionformer_vjepa_balanced_fold{fold}" / "gpu1_id99" / "result_detection.json"
    elif variant == "binary":
        result_path = OPENTAD_DIR / f"actionformer_vjepa_binary_fold{fold}" / "gpu1_id99" / "result_detection.json"
    else:
        return None
    
    if not result_path.exists():
        return None
    
    with open(result_path) as f:
        data = json.load(f)
    
    results = data.get("results", {})
    
    # ActionFormer uses video_key without extension (e.g., "S7V5F9U7N6_12927071...")
    video_key_stem = video_info.video_key_pattern
    
    # Find matching key
    matching_segments = []
    for key, segments in results.items():
        if key.startswith(video_key_stem.split("_")[0]) and video_key_stem.split("_")[1] in key:
            matching_segments.extend(segments)
    
    if not matching_segments:
        # Try a more lenient match
        for key, segments in results.items():
            if video_info.filename.split(".")[0] in key:
                matching_segments.extend(segments)
    
    if not matching_segments:
        return None
    
    return {
        "n_segments": len(matching_segments),
        "segments": [
            {
                "start_sec": seg["segment"][0],
                "end_sec": seg["segment"][1],
                "label": seg["label"],
                "score": seg["score"],
            }
            for seg in matching_segments
        ],
    }


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------

def compute_segment_iou(seg1_start: float, seg1_end: float, seg2_start: float, seg2_end: float) -> float:
    """Compute IoU between two segments."""
    inter_start = max(seg1_start, seg2_start)
    inter_end = min(seg1_end, seg2_end)
    
    if inter_start >= inter_end:
        return 0.0
    
    intersection = inter_end - inter_start
    union = (seg1_end - seg1_start) + (seg2_end - seg2_start) - intersection
    
    return intersection / union if union > 0 else 0.0


def analyze_video(video_info: VideoInfo) -> Dict:
    """Analyze a single video across all models and folds."""
    result = {
        "name": video_info.name,
        "filename": video_info.filename,
        "user_description": video_info.user_description,
        "gt_video_id": video_info.gt_video_id,
        "val_folds": video_info.val_folds,
    }
    
    # Load GT segments
    gt_segments = load_gt_segments(video_info.gt_video_id)
    result["gt_segments"] = gt_segments
    result["gt_summary"] = {
        "n_segments": len(gt_segments),
        "classes": list(set(s["rmm_type"] for s in gt_segments)),
        "total_duration": sum(s["duration"] for s in gt_segments),
    }
    
    # Skip if no validation folds
    if not video_info.val_folds:
        result["clip_level"] = {}
        result["window_level"] = {}
        result["actionformer"] = {}
        return result
    
    # Clip-level predictions (only for folds where this video is in validation)
    clip_level = {}
    for fold in video_info.val_folds:
        fold_results = {}
        
        # 3-Way MLP Fusion (V-JEPA2 + PoseC3D + STGCN++)
        fusion_3way_preds = load_clip_predictions_fusion_3way(video_info, fold)
        if fusion_3way_preds:
            fold_results["fusion_3way_mlp"] = fusion_3way_preds
        
        # V-JEPA2 + SAM3 crop
        vjepa_preds = load_clip_predictions_vjepa(video_info, fold)
        if vjepa_preds:
            fold_results["vjepa_sam3"] = vjepa_preds
        
        # STGCN++ (4-stream)
        stgcnpp_preds = load_clip_predictions_stgcnpp(video_info, fold)
        if stgcnpp_preds:
            fold_results["stgcnpp_4stream"] = stgcnpp_preds
        
        # PoseC3D (non-weighted)
        posec3d_preds = load_clip_predictions_posec3d(video_info, fold)
        if posec3d_preds:
            fold_results["posec3d_nonweighted"] = posec3d_preds
        
        if fold_results:
            clip_level[f"fold_{fold}"] = fold_results
    
    result["clip_level"] = clip_level
    
    # Window-level predictions
    window_level = {}
    for fold in video_info.val_folds:
        fold_results = {}
        
        for model in ["vjepa_balanced", "stgcnpp_ce", "posec3d_focal", "fusion_vjepa_posec3d"]:
            preds = load_window_predictions(video_info, model, fold)
            if preds:
                fold_results[model] = preds
        
        if fold_results:
            window_level[f"fold_{fold}"] = fold_results
    
    result["window_level"] = window_level
    
    # ActionFormer predictions
    actionformer = {}
    for fold in video_info.val_folds:
        fold_results = {}
        
        for variant in ["multiclass", "binary"]:
            preds = load_actionformer_predictions(video_info, variant, fold)
            if preds:
                fold_results[variant] = preds
        
        if fold_results:
            actionformer[f"fold_{fold}"] = fold_results
    
    result["actionformer"] = actionformer
    
    return result


def generate_markdown_report(results: Dict) -> str:
    """Generate a markdown report from the analysis results."""
    lines = [
        "# Video Example Analysis",
        "",
        "This document analyzes model predictions on 10 specific videos selected to illustrate",
        "various challenges: rocking toys, multi-label overlap, False_in_Context annotations,",
        "and brief events (1-second windows).",
        "",
        "---",
        "",
    ]
    
    for video_key, video_data in results.items():
        lines.append(f"## {video_data['name']}: {video_data['user_description']}")
        lines.append("")
        lines.append(f"**Filename**: `{video_data['filename']}`")
        lines.append(f"**GT Video ID**: `{video_data['gt_video_id']}`")
        lines.append(f"**Validation Folds**: {video_data['val_folds']}")
        lines.append("")
        
        # GT summary
        gt_summary = video_data.get("gt_summary", {})
        lines.append("### Ground Truth")
        lines.append("")
        if gt_summary.get("n_segments", 0) == 0:
            lines.append("*No RMM segments annotated for this video.*")
        else:
            lines.append(f"- **Segments**: {gt_summary['n_segments']}")
            lines.append(f"- **Classes**: {', '.join(gt_summary['classes'])}")
            lines.append(f"- **Total Duration**: {gt_summary['total_duration']:.1f}s")
            lines.append("")
            lines.append("| Segment ID | Class | Start | End | Duration |")
            lines.append("|------------|-------|-------|-----|----------|")
            for seg in video_data.get("gt_segments", []):
                lines.append(
                    f"| {seg['segment_id'][-10:]}... | {seg['rmm_type']} | "
                    f"{seg['start_sec']:.1f}s | {seg['end_sec']:.1f}s | {seg['duration']:.1f}s |"
                )
        lines.append("")
        
        # Clip-level results
        lines.append("### Clip-level Predictions")
        lines.append("")
        clip_level = video_data.get("clip_level", {})
        if not clip_level:
            lines.append("*No clip-level predictions available.*")
        else:
            for fold_key, fold_data in clip_level.items():
                lines.append(f"**{fold_key}**:")
                lines.append("")
                for model, preds in fold_data.items():
                    n_clips = preds["n_clips"]
                    n_correct = sum(1 for c in preds["clips"] if c["correct"])
                    acc = n_correct / n_clips * 100 if n_clips > 0 else 0
                    lines.append(f"- **{model}**: {n_correct}/{n_clips} correct ({acc:.1f}%)")
                lines.append("")
        
        # Window-level results
        lines.append("### Window-level Predictions")
        lines.append("")
        window_level = video_data.get("window_level", {})
        if not window_level:
            lines.append("*No window-level predictions available.*")
        else:
            for fold_key, fold_data in window_level.items():
                lines.append(f"**{fold_key}**:")
                lines.append("")
                for model, preds in fold_data.items():
                    n_windows = preds["n_windows"]
                    # Count RMM vs background predictions
                    n_rmm = sum(1 for w in preds["windows"] if w["pred_class"] != "background")
                    lines.append(f"- **{model}**: {n_windows} windows, {n_rmm} predicted as RMM")
                lines.append("")
        
        # ActionFormer results
        lines.append("### ActionFormer Predictions")
        lines.append("")
        actionformer = video_data.get("actionformer", {})
        if not actionformer:
            lines.append("*No ActionFormer predictions available.*")
        else:
            for fold_key, fold_data in actionformer.items():
                lines.append(f"**{fold_key}**:")
                lines.append("")
                for variant, preds in fold_data.items():
                    n_segs = preds["n_segments"]
                    if n_segs > 0:
                        lines.append(f"- **{variant}**: {n_segs} predicted segments")
                        # Show top predictions
                        top_segs = sorted(preds["segments"], key=lambda x: -x["score"])[:5]
                        for seg in top_segs:
                            lines.append(
                                f"  - [{seg['start_sec']:.1f}-{seg['end_sec']:.1f}s] "
                                f"{seg['label']} (score={seg['score']:.3f})"
                            )
                    else:
                        lines.append(f"- **{variant}**: No predictions")
                lines.append("")
        
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Video example analysis")
    parser.add_argument("--output-json", type=str, default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", type=str, default=str(OUTPUT_MD))
    args = parser.parse_args()
    
    print("Analyzing videos...")
    
    results = {}
    for video_info in VIDEOS:
        print(f"  Processing: {video_info.name} ({video_info.filename})")
        results[video_info.name] = analyze_video(video_info)
    
    # Write JSON
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote: {out_json}")
    
    # Write Markdown
    md_content = generate_markdown_report(results)
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md_content)
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
