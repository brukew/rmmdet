#!/usr/bin/env python3
"""
ActionFormer TAL Failure Analysis.

Analyzes failure modes for ActionFormer multiclass and binary models,
correlating with preprocessing factors, video quality, and annotation ambiguity.

Usage:
    python actionformer_failure_analysis.py
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# ActionFormer experiment paths
OPENTAD_EXPS = ACTREG_ROOT / "OpenTAD" / "exps" / "sails_rmm"
MULTICLASS_PATTERN = "actionformer_vjepa_balanced_fold{fold}"
BINARY_PATTERN = "actionformer_vjepa_binary_fold{fold}"

# Data paths
SEGMENTS_PATH = ACTREG_ROOT / "dataprep" / "rmm_segments.csv"
VIDEO_RATINGS_PATH = Path("/orcd/data/satra/002/datasets/SAILS/data4analysis/Video Rating Data/SAILS_RATINGS_ALL_8.8.25.xlsx")
SAM3_PARSED_PATH = ACTREG_ROOT / "dataprep" / "rmm_sam3_parsed.csv"
TAL_SPLITS_DIR = ACTREG_ROOT / "dataprep" / "tal" / "splits_cv_4class"

# Output paths
OUTPUT_DIR = ACTREG_ROOT / "insights" / "tal"
FIGS_DIR = ACTREG_ROOT / "insights" / "figs"
TABLES_DIR = ACTREG_ROOT / "insights" / "tables"

# Class names
CLASS_NAMES = ["hands flapping", "jumping", "rocking", "spinning"]
CLASS_NAME_MAP = {
    "hands flapping": "hands_flapping",
    "jumping": "jumping", 
    "rocking": "rocking",
    "spinning": "spinning",
    "rmm": "rmm"  # binary
}

def load_val_video_keys_by_fold() -> Dict[int, List[str]]:
    """
    Load per-fold validation video keys from the TAL window splits.

    ActionFormer outputs are produced per CV fold. To avoid mixing folds, we must
    evaluate each fold's predictions only on that fold's validation videos.

    Returns:
        Dict mapping fold index -> sorted list of validation video keys.
    """
    val_videos_by_fold: Dict[int, List[str]] = {}

    for fold in range(3):
        fold_path = TAL_SPLITS_DIR / f"fold_{fold}_val_windows.csv"
        if not fold_path.exists():
            print(f"Warning: missing TAL split file: {fold_path}")
            continue

        df = pd.read_csv(fold_path)
        if "window_id" not in df.columns:
            raise ValueError(f"Expected 'window_id' column in {fold_path}")

        # window_id format: <video_key>__t<start_ms>_<end_ms>
        video_keys = (
            df["window_id"]
            .astype(str)
            .apply(lambda x: x.rsplit("__t", 1)[0] if "__t" in x else x)
            .unique()
            .tolist()
        )
        val_videos_by_fold[fold] = sorted(video_keys)

    return val_videos_by_fold


def load_ground_truth_segments() -> pd.DataFrame:
    """Load ground truth RMM segment annotations."""
    df = pd.read_csv(SEGMENTS_PATH)
    
    # Normalize RMM type
    df["rmm_type_norm"] = df["rmm_type"].apply(
        lambda x: "hands flapping" if x == "one hand flap" else x
    )
    
    # Create video key for matching
    def normalize_key(video_file):
        if pd.isna(video_file):
            return None
        video_file = str(video_file).replace("\\", "/")
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


def load_actionformer_predictions(model_type: str = "multiclass") -> Dict[int, Dict]:
    """
    Load ActionFormer predictions for all folds.
    
    Args:
        model_type: "multiclass" or "binary"
        
    Returns:
        Dictionary mapping fold -> {video_key: [predictions]}
    """
    pattern = MULTICLASS_PATTERN if model_type == "multiclass" else BINARY_PATTERN
    
    all_predictions = {}
    
    for fold in range(3):
        fold_dir = OPENTAD_EXPS / pattern.format(fold=fold)
        pred_file = fold_dir / "gpu1_id99" / "result_detection.json"
        
        if not pred_file.exists():
            print(f"  Warning: {pred_file} not found")
            continue
        
        with open(pred_file) as f:
            data = json.load(f)
        
        all_predictions[fold] = data.get("results", data)
        print(f"  Fold {fold}: {len(all_predictions[fold])} videos")
    
    return all_predictions


def compute_segment_iou(pred_start: float, pred_end: float, 
                        gt_start: float, gt_end: float) -> float:
    """Compute temporal IoU between predicted and ground truth segments."""
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def match_predictions_to_gt(
    predictions: List[Dict],
    gt_segments: List[Dict],
    iou_threshold: float = 0.3,
    is_binary: bool = False
) -> Dict:
    """
    Match predictions to ground truth segments.
    
    Returns dictionary with:
    - true_positives: matched predictions
    - false_positives: unmatched predictions
    - false_negatives: unmatched GT segments
    """
    matched_gt = set()
    true_positives = []
    false_positives = []
    
    # Sort predictions by score (descending)
    sorted_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)
    
    for pred in sorted_preds:
        pred_start, pred_end = pred["segment"]
        pred_label = pred["label"]
        
        best_iou = 0
        best_gt_idx = None
        
        for i, gt in enumerate(gt_segments):
            if i in matched_gt:
                continue
            
            gt_start = gt["start_sec"]
            gt_end = gt["end_sec"]
            gt_label = gt["rmm_type_norm"]
            
            # For binary, any RMM class is a match
            if is_binary:
                label_match = pred_label == "rmm" and gt_label in CLASS_NAMES
            else:
                label_match = pred_label == gt_label
            
            if not label_match:
                continue
            
            iou = compute_segment_iou(pred_start, pred_end, gt_start, gt_end)
            
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_gt_idx is not None:
            matched_gt.add(best_gt_idx)
            true_positives.append({
                "pred": pred,
                "gt": gt_segments[best_gt_idx],
                "iou": best_iou
            })
        else:
            false_positives.append(pred)
    
    # False negatives: unmatched GT segments
    false_negatives = [gt for i, gt in enumerate(gt_segments) if i not in matched_gt]
    
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "n_tp": len(true_positives),
        "n_fp": len(false_positives),
        "n_fn": len(false_negatives)
    }


def analyze_video_level_performance(
    predictions: Dict[str, List[Dict]],
    gt_df: pd.DataFrame,
    is_binary: bool = False,
    score_threshold: float = 0.1,
    iou_threshold: float = 0.3,
    video_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Analyze performance at the video level.
    
    Returns DataFrame with per-video metrics.
    """
    results = []
    
    if video_keys is None:
        # Default: analyze all videos present in either predictions or GT.
        video_keys = sorted(
            set(predictions.keys()) | set(gt_df["video_key"].dropna().unique())
        )

    for video_key in video_keys:
        preds = predictions.get(video_key, [])
        video_gt = gt_df[gt_df["video_key"] == video_key]
        
        if len(video_gt) == 0:
            # No GT for this video - all predictions are FP
            filtered_preds = [p for p in preds if p["score"] >= score_threshold]
            results.append({
                "video_key": video_key,
                "n_gt_segments": 0,
                "n_predictions": len(filtered_preds),
                "n_tp": 0,
                "n_fp": len(filtered_preds),
                "n_fn": 0,
                "has_gt": False,
                "precision": 0.0 if len(filtered_preds) > 0 else 1.0,
                "recall": 1.0,  # No GT to miss
                "gt_classes": []
            })
            continue
        
        gt_segments = video_gt.to_dict("records")
        filtered_preds = [p for p in preds if p["score"] >= score_threshold]
        
        match_result = match_predictions_to_gt(
            filtered_preds, gt_segments, iou_threshold, is_binary
        )
        
        n_tp = match_result["n_tp"]
        n_fp = match_result["n_fp"]
        n_fn = match_result["n_fn"]
        
        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
        
        gt_classes = list(video_gt["rmm_type_norm"].unique())
        
        results.append({
            "video_key": video_key,
            "n_gt_segments": len(gt_segments),
            "n_predictions": len(filtered_preds),
            "n_tp": n_tp,
            "n_fp": n_fp,
            "n_fn": n_fn,
            "has_gt": True,
            "precision": precision,
            "recall": recall,
            "gt_classes": gt_classes,
            "missed_segments": match_result["false_negatives"]
        })

    return pd.DataFrame(results)


def load_video_metadata() -> pd.DataFrame:
    """Load video quality ratings and metadata."""
    try:
        df = pd.read_excel(VIDEO_RATINGS_PATH)
        
        # Standardize columns
        df = df.rename(columns={
            'ID': 'child_id',
            'FileName': 'filename',
            '#_adults': 'n_adults',
            '#_children': 'n_children'
        })
        
        # Create video key
        df['video_key'] = df['child_id'].astype(str) + '_' + df['filename'].apply(
            lambda x: Path(str(x)).stem if pd.notna(x) else ""
        )
        
        # Quality flags
        quality_cols = [c for c in df.columns if 'Video_Quality' in c]
        for col in quality_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['low_body_visibility'] = df['Video_Quality_Child_Body_Visibility'] <= 5
        df['low_hand_visibility'] = df['Video_Quality_Child_Hand_Visibility'] <= 5
        df['high_motion'] = df['Video_Quality_Motion'] <= 5
        df['multi_child'] = pd.to_numeric(df['n_children'], errors='coerce') > 1
        
        return df
    except Exception as e:
        print(f"Warning: Could not load video ratings: {e}")
        return pd.DataFrame()


def load_sam3_annotations() -> pd.DataFrame:
    """Load SAM3 annotation notes for ambiguity analysis."""
    df = pd.read_csv(SAM3_PARSED_PATH)
    
    # Create video key
    def get_video_key(source_file):
        if pd.isna(source_file):
            return None
        source_file = str(source_file).replace("\\", "/")
        parts = source_file.split("/")
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
    
    df['video_key'] = df['SourceFile'].apply(get_video_key)
    
    # Ambiguity flags
    df['is_debatable'] = df['RMM_Label_Note'].str.contains('Debatable', case=False, na=False)
    df['has_context_issue'] = df['RMM_Label_Note'].str.contains('Has_Context', case=False, na=False)
    df['annotation_ambiguous'] = df['RMM_Notes'].apply(
        lambda x: any(kw in str(x).lower() for kw in ['vs', '?', 'not sure', 'coincide']) if pd.notna(x) else False
    )
    
    return df


def load_preprocessing_info() -> pd.DataFrame:
    """Load preprocessing reliability info from TAL splits."""
    dfs = []
    for fold in range(3):
        fold_path = TAL_SPLITS_DIR / f"fold_{fold}_val_windows.csv"
        if fold_path.exists():
            df = pd.read_csv(fold_path)
            df['fold'] = fold
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Extract video key from window_id
    combined['video_key'] = combined['window_id'].apply(
        lambda x: x.rsplit("__t", 1)[0] if "__t" in x else x
    )
    
    # Aggregate preprocessing issues at video level
    combined['sam3_issue'] = combined['mask_cause'] != 'full'
    combined['pose_issue'] = combined['pose_cause'] != 'full'
    
    video_preproc = combined.groupby('video_key').agg({
        'sam3_issue': 'any',
        'pose_issue': 'any',
        'mask_severity': lambda x: (x == 'severe').any(),
        'pose_severity': lambda x: (x == 'severe').any()
    }).reset_index()
    
    video_preproc.columns = ['video_key', 'has_sam3_issue', 'has_pose_issue', 
                              'sam3_severe', 'pose_severe']
    
    return video_preproc


def analyze_class_performance(
    video_results: pd.DataFrame,
    gt_df: pd.DataFrame
) -> Dict:
    """Analyze performance by RMM class."""
    class_stats = {}
    
    for class_name in CLASS_NAMES:
        class_gt = gt_df[gt_df["rmm_type_norm"] == class_name]
        n_segments = len(class_gt)
        
        if n_segments == 0:
            continue
        
        # Count how many were detected
        n_detected = 0
        n_missed = 0
        
        for _, row in video_results.iterrows():
            if not row['has_gt']:
                continue
            
            if 'missed_segments' in row and isinstance(row['missed_segments'], list):
                for seg in row['missed_segments']:
                    if seg.get('rmm_type_norm') == class_name:
                        n_missed += 1
        
        # Calculate from GT
        video_class_gt = class_gt.groupby('video_key').size().reset_index(name='n_segments')
        
        class_recall_list = []
        for video_key in video_class_gt['video_key']:
            video_row = video_results[video_results['video_key'] == video_key]
            if len(video_row) > 0:
                class_recall_list.append(video_row.iloc[0]['recall'])
        
        class_stats[class_name] = {
            "n_segments": n_segments,
            "n_videos": len(class_gt['video_key'].unique()),
            "avg_recall": float(np.mean(class_recall_list)) if class_recall_list else 0.0
        }
    
    return class_stats


def compare_multiclass_vs_binary(
    multi_results: pd.DataFrame,
    binary_results: pd.DataFrame,
    gt_df: pd.DataFrame
) -> Dict:
    """Compare multiclass and binary model performance."""
    comparison = {
        "overall": {},
        "by_class": {},
        "agreement": {}
    }
    
    # Overall metrics
    multi_with_gt = multi_results[multi_results['has_gt']]
    binary_with_gt = binary_results[binary_results['has_gt']]
    
    comparison["overall"]["multiclass"] = {
        "avg_precision": float(multi_with_gt['precision'].mean()),
        "avg_recall": float(multi_with_gt['recall'].mean()),
        "total_tp": int(multi_with_gt['n_tp'].sum()),
        "total_fp": int(multi_with_gt['n_fp'].sum()),
        "total_fn": int(multi_with_gt['n_fn'].sum())
    }
    
    comparison["overall"]["binary"] = {
        "avg_precision": float(binary_with_gt['precision'].mean()),
        "avg_recall": float(binary_with_gt['recall'].mean()),
        "total_tp": int(binary_with_gt['n_tp'].sum()),
        "total_fp": int(binary_with_gt['n_fp'].sum()),
        "total_fn": int(binary_with_gt['n_fn'].sum())
    }
    
    # Compare on same videos
    common_videos = set(multi_results['video_key']) & set(binary_results['video_key'])
    
    multi_better = 0
    binary_better = 0
    same = 0
    
    for video_key in common_videos:
        multi_row = multi_results[multi_results['video_key'] == video_key].iloc[0]
        binary_row = binary_results[binary_results['video_key'] == video_key].iloc[0]
        
        if multi_row['recall'] > binary_row['recall']:
            multi_better += 1
        elif binary_row['recall'] > multi_row['recall']:
            binary_better += 1
        else:
            same += 1
    
    comparison["agreement"] = {
        "common_videos": len(common_videos),
        "multiclass_better_recall": multi_better,
        "binary_better_recall": binary_better,
        "same_recall": same
    }
    
    return comparison


def analyze_failure_correlations(
    video_results: pd.DataFrame,
    metadata_df: pd.DataFrame,
    preproc_df: pd.DataFrame,
    sam3_df: pd.DataFrame
) -> Dict:
    """Correlate failures with various factors."""
    correlations = {}
    
    # Merge metadata
    merged = video_results.copy()
    
    if not metadata_df.empty:
        merged = merged.merge(
            metadata_df[['video_key', 'low_body_visibility', 'low_hand_visibility', 
                        'high_motion', 'multi_child']].drop_duplicates('video_key'),
            on='video_key', how='left'
        )
    
    if not preproc_df.empty:
        merged = merged.merge(preproc_df, on='video_key', how='left')
    
    if not sam3_df.empty:
        merged = merged.merge(
            sam3_df[['video_key', 'is_debatable', 'annotation_ambiguous']].drop_duplicates('video_key'),
            on='video_key', how='left'
        )
    
    # Fill NaN
    for col in ['low_body_visibility', 'low_hand_visibility', 'high_motion', 
                'multi_child', 'has_sam3_issue', 'has_pose_issue',
                'is_debatable', 'annotation_ambiguous']:
        if col in merged.columns:
            merged[col] = merged[col].fillna(False)
    
    # Filter to videos with GT
    merged = merged[merged['has_gt']]
    
    # Analyze each factor
    factors = [
        ('has_sam3_issue', 'SAM3 Mask Issues'),
        ('has_pose_issue', 'Pose Keypoint Issues'),
        ('low_body_visibility', 'Low Body Visibility'),
        ('low_hand_visibility', 'Low Hand Visibility'),
        ('high_motion', 'High Motion Blur'),
        ('multi_child', 'Multiple Children'),
        ('is_debatable', 'Debatable Annotation'),
        ('annotation_ambiguous', 'Ambiguous Annotation')
    ]
    
    for col, name in factors:
        if col not in merged.columns:
            continue
        
        with_factor = merged[merged[col] == True]
        without_factor = merged[merged[col] == False]
        
        if len(with_factor) == 0 or len(without_factor) == 0:
            continue
        
        correlations[name] = {
            "n_with": len(with_factor),
            "n_without": len(without_factor),
            "recall_with": float(with_factor['recall'].mean()),
            "recall_without": float(without_factor['recall'].mean()),
            "recall_diff": float(with_factor['recall'].mean() - without_factor['recall'].mean()),
            "precision_with": float(with_factor['precision'].mean()),
            "precision_without": float(without_factor['precision'].mean())
        }
    
    return correlations


def run_full_analysis() -> Dict:
    """Run the complete ActionFormer failure analysis."""
    print("=" * 60)
    print("ActionFormer TAL Failure Analysis")
    print("=" * 60)
    
    # Load ground truth
    print("\n1. Loading ground truth segments...")
    gt_df = load_ground_truth_segments()
    print(f"   Loaded {len(gt_df)} segments from {gt_df['video_key'].nunique()} videos")
    
    # Load predictions
    print("\n2. Loading ActionFormer multiclass predictions...")
    multi_preds = load_actionformer_predictions("multiclass")
    
    print("\n3. Loading ActionFormer binary predictions...")
    binary_preds = load_actionformer_predictions("binary")
    
    # Load metadata
    print("\n4. Loading metadata...")
    metadata_df = load_video_metadata()
    print(f"   Video ratings: {len(metadata_df)} videos")
    
    sam3_df = load_sam3_annotations()
    print(f"   SAM3 annotations: {len(sam3_df)} videos")
    
    preproc_df = load_preprocessing_info()
    print(f"   Preprocessing info: {len(preproc_df)} videos")

    # Load fold validation video keys (critical to avoid fold mixing)
    val_videos_by_fold = load_val_video_keys_by_fold()
    all_val_videos: List[str] = sorted(
        {vk for vks in val_videos_by_fold.values() for vk in vks}
    )
    print(f"   Validation videos (union across folds): {len(all_val_videos)}")

    # Restrict GT to validation videos only (for consistency with ActionFormer outputs)
    gt_df_val = gt_df[gt_df["video_key"].isin(all_val_videos)].copy()
    print(
        f"   GT segments in validation videos: {len(gt_df_val)} "
        f"across {gt_df_val['video_key'].nunique()} videos"
    )
    
    results = {
        "multiclass": {"by_fold": {}, "combined": {}},
        "binary": {"by_fold": {}, "combined": {}},
        "comparison": {},
        "failure_correlations": {}
    }
    
    # Analyze multiclass by fold
    print("\n5. Analyzing multiclass model...")
    all_multi_results = []
    for fold, preds in multi_preds.items():
        fold_video_keys = val_videos_by_fold.get(fold, [])
        if not fold_video_keys:
            print(f"  Warning: no validation videos found for fold {fold}; skipping")
            continue

        gt_fold = gt_df_val[gt_df_val["video_key"].isin(fold_video_keys)].copy()
        video_results = analyze_video_level_performance(
            preds,
            gt_fold,
            is_binary=False,
            score_threshold=0.1,
            iou_threshold=0.3,
            video_keys=fold_video_keys,
        )
        all_multi_results.append(video_results)
        
        with_gt = video_results[video_results['has_gt']]
        results["multiclass"]["by_fold"][f"fold_{fold}"] = {
            "n_videos": len(with_gt),
            "avg_precision": float(with_gt['precision'].mean()),
            "avg_recall": float(with_gt['recall'].mean()),
            "total_tp": int(with_gt['n_tp'].sum()),
            "total_fn": int(with_gt['n_fn'].sum())
        }
    
    if all_multi_results:
        combined_multi = pd.concat(all_multi_results, ignore_index=True)
        # Folds are disjoint by construction, but keep this safeguard anyway.
        combined_multi = combined_multi.drop_duplicates('video_key', keep='first')
        
        with_gt = combined_multi[combined_multi['has_gt']]
        results["multiclass"]["combined"] = {
            "n_videos": len(with_gt),
            "avg_precision": float(with_gt['precision'].mean()),
            "avg_recall": float(with_gt['recall'].mean()),
            "total_tp": int(with_gt['n_tp'].sum()),
            "total_fn": int(with_gt['n_fn'].sum())
        }
        
        results["multiclass"]["class_performance"] = analyze_class_performance(
            combined_multi, gt_df_val
        )
    
    # Analyze binary by fold
    print("\n6. Analyzing binary model...")
    all_binary_results = []
    for fold, preds in binary_preds.items():
        fold_video_keys = val_videos_by_fold.get(fold, [])
        if not fold_video_keys:
            print(f"  Warning: no validation videos found for fold {fold}; skipping")
            continue

        gt_fold = gt_df_val[gt_df_val["video_key"].isin(fold_video_keys)].copy()
        video_results = analyze_video_level_performance(
            preds,
            gt_fold,
            is_binary=True,
            score_threshold=0.1,
            iou_threshold=0.3,
            video_keys=fold_video_keys,
        )
        all_binary_results.append(video_results)
        
        with_gt = video_results[video_results['has_gt']]
        results["binary"]["by_fold"][f"fold_{fold}"] = {
            "n_videos": len(with_gt),
            "avg_precision": float(with_gt['precision'].mean()),
            "avg_recall": float(with_gt['recall'].mean()),
            "total_tp": int(with_gt['n_tp'].sum()),
            "total_fn": int(with_gt['n_fn'].sum())
        }
    
    if all_binary_results:
        combined_binary = pd.concat(all_binary_results, ignore_index=True)
        combined_binary = combined_binary.drop_duplicates('video_key', keep='first')
        
        with_gt = combined_binary[combined_binary['has_gt']]
        results["binary"]["combined"] = {
            "n_videos": len(with_gt),
            "avg_precision": float(with_gt['precision'].mean()),
            "avg_recall": float(with_gt['recall'].mean()),
            "total_tp": int(with_gt['n_tp'].sum()),
            "total_fn": int(with_gt['n_fn'].sum())
        }
    
    # Compare multiclass vs binary
    print("\n7. Comparing multiclass vs binary...")
    if all_multi_results and all_binary_results:
        results["comparison"] = compare_multiclass_vs_binary(
            combined_multi, combined_binary, gt_df_val
        )
    
    # Failure correlations
    print("\n8. Analyzing failure correlations...")
    if all_multi_results:
        results["failure_correlations"]["multiclass"] = analyze_failure_correlations(
            combined_multi, metadata_df, preproc_df, sam3_df
        )
    if all_binary_results:
        results["failure_correlations"]["binary"] = analyze_failure_correlations(
            combined_binary, metadata_df, preproc_df, sam3_df
        )
    
    return results


def print_summary(results: Dict):
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n### Multiclass ActionFormer")
    mc = results["multiclass"]["combined"]
    print(f"  Videos with GT: {mc['n_videos']}")
    print(f"  Average Precision: {mc['avg_precision']*100:.1f}%")
    print(f"  Average Recall: {mc['avg_recall']*100:.1f}%")
    print(f"  Total TP: {mc['total_tp']}, FN: {mc['total_fn']}")
    
    print("\n### Binary ActionFormer")
    bi = results["binary"]["combined"]
    print(f"  Videos with GT: {bi['n_videos']}")
    print(f"  Average Precision: {bi['avg_precision']*100:.1f}%")
    print(f"  Average Recall: {bi['avg_recall']*100:.1f}%")
    print(f"  Total TP: {bi['total_tp']}, FN: {bi['total_fn']}")
    
    print("\n### Comparison")
    comp = results["comparison"]
    if "agreement" in comp:
        agr = comp["agreement"]
        print(f"  Common videos: {agr['common_videos']}")
        print(f"  Multiclass better recall: {agr['multiclass_better_recall']}")
        print(f"  Binary better recall: {agr['binary_better_recall']}")
        print(f"  Same recall: {agr['same_recall']}")
    
    print("\n### Per-Class Performance (Multiclass)")
    if "class_performance" in results["multiclass"]:
        for class_name, stats in results["multiclass"]["class_performance"].items():
            print(f"  {class_name}: {stats['n_segments']} segments, "
                  f"avg recall={stats['avg_recall']*100:.1f}%")
    
    print("\n### Failure Correlations (Multiclass)")
    if "multiclass" in results["failure_correlations"]:
        for factor, stats in results["failure_correlations"]["multiclass"].items():
            diff = stats['recall_diff'] * 100
            direction = "↓" if diff < 0 else "↑"
            print(f"  {factor}: recall {direction} {abs(diff):.1f}pp "
                  f"({stats['n_with']} with, {stats['n_without']} without)")


def main():
    parser = argparse.ArgumentParser(description="ActionFormer TAL Failure Analysis")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    
    # Run analysis
    results = run_full_analysis()
    
    # Print summary
    print_summary(results)
    
    # Save results
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    json_path = TABLES_DIR / "actionformer_failure_analysis.json"
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nSaved results: {json_path}")


if __name__ == "__main__":
    main()
