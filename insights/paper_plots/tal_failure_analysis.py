#!/usr/bin/env python3
"""
TAL Failure Analysis Script.

Correlates preprocessing factors (SAM3, pose, video quality) and annotation 
ambiguity with model prediction errors to identify systematic failure modes.

Usage:
    python tal_failure_analysis.py
    python tal_failure_analysis.py --output-dir insights/writeup
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

# Data paths
TAL_SPLITS_DIR = ACTREG_ROOT / "dataprep" / "tal" / "splits_cv_4class"
TAL_SPLITS_5CLASS_DIR = ACTREG_ROOT / "dataprep" / "tal" / "splits_cv_5class"
VIDEO_RATINGS_PATH = Path("/orcd/data/satra/002/datasets/SAILS/data4analysis/Video Rating Data/SAILS_RATINGS_ALL_8.8.25.xlsx")
SEGMENT_ANNOTATIONS_PATH = ACTREG_ROOT / "dataprep" / "rmm_segments.csv"
SAM3_PARSED_PATH = ACTREG_ROOT / "dataprep" / "rmm_sam3_parsed.csv"
TAL_PREDICTIONS_DIR = ACTREG_ROOT / "tal" / "eval_results" / "vjepa_balanced"

# Output paths
OUTPUT_DIR = ACTREG_ROOT / "insights" / "writeup"
FIGS_DIR = ACTREG_ROOT / "insights" / "figs"
TABLES_DIR = ACTREG_ROOT / "insights" / "tables"

# Class names
CLASS_NAMES_4CLASS = {0: "hands_flapping", 1: "jumping", 2: "rocking", 3: "spinning", -1: "background"}
CLASS_NAMES_5CLASS = {0: "hands_flapping", 1: "jumping", 2: "one_hand_flap", 3: "rocking", 4: "spinning", -1: "background"}


def load_tal_splits(n_classes: int = 4) -> pd.DataFrame:
    """
    Load and concatenate all validation windows from TAL splits.
    
    Args:
        n_classes: Number of classes (4 or 5)
        
    Returns:
        DataFrame with all validation windows across folds
    """
    splits_dir = TAL_SPLITS_DIR if n_classes == 4 else TAL_SPLITS_5CLASS_DIR
    
    dfs = []
    for fold in range(3):
        fold_path = splits_dir / f"fold_{fold}_val_windows.csv"
        if fold_path.exists():
            df = pd.read_csv(fold_path)
            df['fold'] = fold
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"No validation windows found in {splits_dir}")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Parse labels column
    combined['labels_parsed'] = combined['labels'].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )
    combined['n_labels'] = combined['labels_parsed'].apply(len)
    
    # Compute preprocessing issue flags
    combined['sam3_issue'] = combined['mask_cause'] != 'full'
    combined['pose_issue'] = combined['pose_cause'] != 'full'
    combined['any_preproc_issue'] = combined['sam3_issue'] | combined['pose_issue']
    
    # Compute multi-label overlap
    tiou_cols = [c for c in combined.columns if c.startswith('tiou_')]
    combined['n_overlapping'] = (combined[tiou_cols] > 0).sum(axis=1)
    combined['has_multi_overlap'] = combined['n_overlapping'] > 1
    
    # RMM vs background flag
    combined['is_rmm'] = combined['primary_label'] != -1
    
    return combined


def load_video_ratings() -> pd.DataFrame:
    """
    Load video quality ratings from Excel file.
    
    Returns:
        DataFrame with video-level quality metrics
    """
    df = pd.read_excel(VIDEO_RATINGS_PATH)
    
    # Standardize column names
    df = df.rename(columns={
        'ID': 'child_id',
        'FileName': 'filename',
        '#_adults': 'n_adults_rating',
        '#_children': 'n_children_rating',
        '#_people_background': 'n_people_background'
    })
    
    # Convert quality columns to numeric
    quality_cols = [c for c in df.columns if 'Video_Quality' in c]
    for col in quality_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute composite quality score
    df['quality_composite'] = df[quality_cols].mean(axis=1)
    
    # Low quality flags (score <= 5 on 1-10 scale)
    df['low_face_visibility'] = df['Video_Quality_Child_Face_Visibility'] <= 5
    df['low_body_visibility'] = df['Video_Quality_Child_Body_Visibility'] <= 5
    df['low_hand_visibility'] = df['Video_Quality_Child_Hand_Visibility'] <= 5
    df['high_motion_blur'] = df['Video_Quality_Motion'] <= 5
    
    # Multi-person scene flags
    df['n_children_rating'] = pd.to_numeric(df['n_children_rating'], errors='coerce')
    df['n_adults_rating'] = pd.to_numeric(df['n_adults_rating'], errors='coerce')
    df['multi_child_scene'] = df['n_children_rating'] > 1
    df['adult_present'] = df['n_adults_rating'] > 0
    
    # Body visibility
    df['upper_body_only'] = df['Body_Parts_Visible'] == 'upper'
    
    # Child clarity
    df['child_unclear'] = df['Child_of_interest_clear'].str.lower() == 'no'
    
    return df


def load_segment_annotations() -> pd.DataFrame:
    """
    Load RMM segment annotations with quality ratings.
    
    Returns:
        DataFrame with segment-level annotations
    """
    df = pd.read_csv(SEGMENT_ANNOTATIONS_PATH)
    
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
    
    df['video_key_norm'] = df['video_file'].apply(normalize_key)
    
    # Quality rating categories
    df['low_quality_segment'] = df['quality_rating'] <= 3
    df['has_quality_rating'] = df['quality_rating'].notna()
    
    return df


def load_sam3_annotations() -> pd.DataFrame:
    """
    Load SAM3 parsed annotations with notes about ambiguity.
    
    Returns:
        DataFrame with annotation ambiguity flags
    """
    df = pd.read_csv(SAM3_PARSED_PATH)
    
    # Extract video key
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
    
    df['video_key_norm'] = df['SourceFile'].apply(get_video_key)
    
    # Annotation ambiguity flags
    df['has_rmm_notes'] = df['RMM_Notes'].notna() & (df['RMM_Notes'] != '')
    df['is_debatable'] = df['RMM_Label_Note'].str.contains('Debatable', case=False, na=False)
    df['has_context_issue'] = df['RMM_Label_Note'].str.contains('Has_Context', case=False, na=False)
    df['no_rmm_identified'] = df['RMM_Label_Note'].str.contains('No_RMM', case=False, na=False)
    df['has_sam3_notes'] = df['SAM3 Notes'].notna() & (df['SAM3 Notes'] != '')
    
    # Check for specific ambiguity keywords in notes
    ambiguity_keywords = ['twisting', 'vs', 'not sure', 'debatable', '?', 'coincide', 'both']
    df['annotation_ambiguous'] = df['RMM_Notes'].apply(
        lambda x: any(kw in str(x).lower() for kw in ambiguity_keywords) if pd.notna(x) else False
    )
    
    return df


def load_model_predictions(fold: int = 0) -> pd.DataFrame:
    """
    Load TAL window-level predictions for error analysis.
    
    Args:
        fold: Fold number (0, 1, or 2)
        
    Returns:
        DataFrame with predictions and correctness flags
    """
    pred_path = TAL_PREDICTIONS_DIR / f"fold{fold}" / "tal_format_preds.csv"
    
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    
    df = pd.read_csv(pred_path)
    
    # Compute predicted class from scores
    score_cols = [f'score_class{i}' for i in range(5)]
    df['pred_id'] = df[score_cols].values.argmax(axis=1)
    
    return df


def merge_all_data(
    tal_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    sam3_df: pd.DataFrame,
    predictions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all data sources into a unified DataFrame for analysis.
    
    Returns:
        Merged DataFrame with all factors and error flags
    """
    # Extract video key from TAL splits for matching
    def extract_video_key(window_id):
        # Format: CHILD_ID_FILENAME__tSTART_END
        parts = window_id.rsplit("__t", 1)
        return parts[0] if len(parts) > 1 else window_id
    
    tal_df = tal_df.copy()
    tal_df['video_key_norm'] = tal_df['window_id'].apply(extract_video_key)
    
    # Extract child_id and filename for ratings merge
    tal_df['child_id'] = tal_df['child_id'].astype(str)
    tal_df['filename_stem'] = tal_df['filename'].apply(lambda x: Path(str(x)).stem if pd.notna(x) else None)
    
    # Merge predictions by window_id
    predictions_df = predictions_df.copy()
    
    # Map primary_label to 5-class prediction space (-1 -> 4 for background)
    tal_df['label_5class'] = tal_df['primary_label'].apply(lambda x: 4 if x == -1 else x)
    
    merged = tal_df.merge(
        predictions_df[['window_id', 'pred_id']],
        on='window_id',
        how='left'
    )
    
    # Compute correctness based on 5-class labels
    merged['is_correct'] = merged['label_5class'] == merged['pred_id']
    merged['is_error'] = ~merged['is_correct'] & merged['pred_id'].notna()
    
    # Merge video ratings (by child_id and filename)
    ratings_df = ratings_df.copy()
    ratings_df['child_id'] = ratings_df['child_id'].astype(str)
    ratings_df['filename_stem'] = ratings_df['filename'].apply(lambda x: Path(str(x)).stem if pd.notna(x) else None)
    
    # Create unique key for ratings merge
    merged['ratings_key'] = merged['child_id'] + '_' + merged['filename_stem'].astype(str)
    ratings_df['ratings_key'] = ratings_df['child_id'] + '_' + ratings_df['filename_stem'].astype(str)
    
    # Select relevant columns from ratings
    ratings_cols = [
        'ratings_key', 'quality_composite', 
        'Video_Quality_Child_Face_Visibility', 'Video_Quality_Child_Body_Visibility',
        'Video_Quality_Child_Hand_Visibility', 'Video_Quality_Motion',
        'low_face_visibility', 'low_body_visibility', 'low_hand_visibility',
        'high_motion_blur', 'multi_child_scene', 'adult_present',
        'upper_body_only', 'child_unclear', 'Body_Parts_Visible', 'Angle_of_Body'
    ]
    ratings_subset = ratings_df[[c for c in ratings_cols if c in ratings_df.columns]].drop_duplicates('ratings_key')
    
    merged = merged.merge(ratings_subset, on='ratings_key', how='left')
    
    # Merge SAM3 annotations (by video_key_norm)
    sam3_cols = [
        'video_key_norm', 'has_rmm_notes', 'is_debatable', 'has_context_issue',
        'no_rmm_identified', 'has_sam3_notes', 'annotation_ambiguous', 'RMM_Notes'
    ]
    sam3_subset = sam3_df[[c for c in sam3_cols if c in sam3_df.columns]].drop_duplicates('video_key_norm')
    
    merged = merged.merge(sam3_subset, on='video_key_norm', how='left', suffixes=('', '_sam3'))
    
    # Fill NaN flags with False
    flag_cols = [
        'sam3_issue', 'pose_issue', 'any_preproc_issue', 'has_multi_overlap',
        'low_face_visibility', 'low_body_visibility', 'low_hand_visibility',
        'high_motion_blur', 'multi_child_scene', 'adult_present',
        'upper_body_only', 'child_unclear', 'has_rmm_notes', 'is_debatable',
        'has_context_issue', 'annotation_ambiguous', 'has_sam3_notes'
    ]
    for col in flag_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(False)
    
    return merged


def compute_error_rates(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Compute error rates grouped by a factor.
    
    Args:
        df: DataFrame with is_error column
        group_col: Column to group by
        
    Returns:
        DataFrame with error rates per group
    """
    # Filter to rows with predictions
    pred_df = df[df['is_error'].notna()].copy()
    
    if len(pred_df) == 0:
        return pd.DataFrame()
    
    grouped = pred_df.groupby(group_col).agg({
        'is_error': ['sum', 'count', 'mean'],
        'window_id': 'count'
    }).reset_index()
    
    grouped.columns = [group_col, 'n_errors', 'n_total', 'error_rate', 'n_windows']
    grouped['accuracy'] = 1 - grouped['error_rate']
    
    return grouped


def analyze_factor_impact(
    df: pd.DataFrame,
    factor_col: str,
    factor_name: str
) -> Dict:
    """
    Analyze impact of a binary factor on error rate.
    
    Args:
        df: Merged DataFrame
        factor_col: Column name for the factor
        factor_name: Human-readable factor name
        
    Returns:
        Dictionary with analysis results
    """
    pred_df = df[df['is_error'].notna()].copy()
    
    if factor_col not in pred_df.columns:
        return {"factor": factor_name, "available": False}
    
    # Overall stats
    with_factor = pred_df[pred_df[factor_col] == True]
    without_factor = pred_df[pred_df[factor_col] == False]
    
    n_with = len(with_factor)
    n_without = len(without_factor)
    
    if n_with == 0 or n_without == 0:
        return {
            "factor": factor_name,
            "available": True,
            "n_with_factor": n_with,
            "n_without_factor": n_without,
            "insufficient_data": True
        }
    
    error_rate_with = with_factor['is_error'].mean()
    error_rate_without = without_factor['is_error'].mean()
    
    # By RMM class
    class_impact = {}
    for label, name in CLASS_NAMES_4CLASS.items():
        class_df = pred_df[pred_df['primary_label'] == label]
        if len(class_df) == 0:
            continue
        
        class_with = class_df[class_df[factor_col] == True]
        class_without = class_df[class_df[factor_col] == False]
        
        if len(class_with) > 0 and len(class_without) > 0:
            class_impact[name] = {
                "n_with": len(class_with),
                "n_without": len(class_without),
                "error_rate_with": float(class_with['is_error'].mean()),
                "error_rate_without": float(class_without['is_error'].mean()),
                "error_rate_diff": float(class_with['is_error'].mean() - class_without['is_error'].mean())
            }
    
    return {
        "factor": factor_name,
        "available": True,
        "n_with_factor": n_with,
        "n_without_factor": n_without,
        "pct_with_factor": float(n_with / len(pred_df) * 100),
        "error_rate_with": float(error_rate_with),
        "error_rate_without": float(error_rate_without),
        "error_rate_diff": float(error_rate_with - error_rate_without),
        "accuracy_with": float(1 - error_rate_with),
        "accuracy_without": float(1 - error_rate_without),
        "by_class": class_impact
    }


def analyze_multi_label_overlap(df: pd.DataFrame) -> Dict:
    """
    Analyze impact of multi-label overlap on errors.
    
    Returns:
        Dictionary with multi-label analysis
    """
    pred_df = df[df['is_error'].notna()].copy()
    
    # Get overlap combinations
    tiou_cols = [c for c in pred_df.columns if c.startswith('tiou_')]
    
    results = {
        "total_with_overlap": int((pred_df['n_overlapping'] > 1).sum()),
        "pct_with_overlap": float((pred_df['n_overlapping'] > 1).mean() * 100),
        "overlap_combinations": {},
        "error_by_n_overlapping": {}
    }
    
    # Error rate by number of overlapping classes
    for n in range(5):
        subset = pred_df[pred_df['n_overlapping'] == n]
        if len(subset) > 0:
            results["error_by_n_overlapping"][n] = {
                "n_windows": len(subset),
                "error_rate": float(subset['is_error'].mean()),
                "n_errors": int(subset['is_error'].sum())
            }
    
    # Identify specific overlap combinations
    multi_overlap = pred_df[pred_df['n_overlapping'] > 1]
    if len(multi_overlap) > 0:
        for _, row in multi_overlap.iterrows():
            overlapping = tuple(sorted([
                c.replace('tiou_', '') for c in tiou_cols if row[c] > 0
            ]))
            if overlapping not in results["overlap_combinations"]:
                results["overlap_combinations"][overlapping] = {"count": 0, "errors": 0}
            results["overlap_combinations"][overlapping]["count"] += 1
            if row['is_error']:
                results["overlap_combinations"][overlapping]["errors"] += 1
        
        # Convert tuples to strings for JSON
        results["overlap_combinations"] = {
            str(k): v for k, v in results["overlap_combinations"].items()
        }
    
    return results


def analyze_annotation_ambiguity(df: pd.DataFrame, sam3_df: pd.DataFrame) -> Dict:
    """
    Analyze annotation ambiguity patterns.
    
    Returns:
        Dictionary with ambiguity analysis
    """
    pred_df = df[df['is_error'].notna()].copy()
    
    # Ambiguity notes analysis
    ambiguity_notes = sam3_df[sam3_df['has_rmm_notes'] | sam3_df['annotation_ambiguous']].copy()
    
    results = {
        "total_debatable": int(pred_df['is_debatable'].sum()) if 'is_debatable' in pred_df.columns else 0,
        "total_has_context": int(pred_df['has_context_issue'].sum()) if 'has_context_issue' in pred_df.columns else 0,
        "total_ambiguous_notes": int(pred_df['annotation_ambiguous'].sum()) if 'annotation_ambiguous' in pred_df.columns else 0,
        "sample_ambiguity_notes": [],
        "error_rate_debatable": None,
        "error_rate_not_debatable": None
    }
    
    # Error rates for debatable annotations
    if 'is_debatable' in pred_df.columns:
        debatable = pred_df[pred_df['is_debatable'] == True]
        not_debatable = pred_df[pred_df['is_debatable'] == False]
        
        if len(debatable) > 0:
            results["error_rate_debatable"] = float(debatable['is_error'].mean())
        if len(not_debatable) > 0:
            results["error_rate_not_debatable"] = float(not_debatable['is_error'].mean())
    
    # Sample notes
    if 'RMM_Notes' in sam3_df.columns:
        sample_notes = sam3_df[sam3_df['RMM_Notes'].notna() & (sam3_df['RMM_Notes'] != '')]['RMM_Notes'].head(20).tolist()
        results["sample_ambiguity_notes"] = sample_notes
    
    return results


def analyze_video_quality_impact(df: pd.DataFrame) -> Dict:
    """
    Analyze impact of video quality factors.
    
    Returns:
        Dictionary with quality analysis
    """
    pred_df = df[df['is_error'].notna()].copy()
    
    results = {
        "quality_factors": {},
        "quality_score_correlation": None
    }
    
    quality_factors = [
        ('low_face_visibility', 'Low Face Visibility'),
        ('low_body_visibility', 'Low Body Visibility'),
        ('low_hand_visibility', 'Low Hand Visibility'),
        ('high_motion_blur', 'High Motion Blur'),
        ('upper_body_only', 'Upper Body Only'),
        ('child_unclear', 'Child of Interest Unclear')
    ]
    
    for col, name in quality_factors:
        if col in pred_df.columns:
            impact = analyze_factor_impact(pred_df, col, name)
            results["quality_factors"][name] = impact
    
    # Correlation with composite quality score
    if 'quality_composite' in pred_df.columns:
        valid_quality = pred_df[pred_df['quality_composite'].notna()]
        if len(valid_quality) > 10:
            correlation = valid_quality['quality_composite'].corr(valid_quality['is_error'].astype(float))
            results["quality_score_correlation"] = float(correlation)
    
    return results


def analyze_scene_complexity(df: pd.DataFrame) -> Dict:
    """
    Analyze impact of scene complexity (multi-person scenes).
    
    Returns:
        Dictionary with scene complexity analysis
    """
    pred_df = df[df['is_error'].notna()].copy()
    
    results = {
        "complexity_factors": {}
    }
    
    complexity_factors = [
        ('multi_child_scene', 'Multiple Children Present'),
        ('adult_present', 'Adult Present in Scene')
    ]
    
    for col, name in complexity_factors:
        if col in pred_df.columns:
            impact = analyze_factor_impact(pred_df, col, name)
            results["complexity_factors"][name] = impact
    
    return results


def analyze_5class_confusion(tal_5class_df: pd.DataFrame) -> Dict:
    """
    Analyze confusion between hands_flapping and one_hand_flap in 5-class task.
    
    Returns:
        Dictionary with 5-class confusion analysis
    """
    results = {
        "hf_ohf_overlap": {},
        "class_distribution": {}
    }
    
    # Class distribution
    for label, name in CLASS_NAMES_5CLASS.items():
        count = (tal_5class_df['primary_label'] == label).sum()
        results["class_distribution"][name] = int(count)
    
    # Overlap between hands_flapping and one_hand_flap via tIoU
    hf_windows = tal_5class_df[tal_5class_df['primary_label'] == 0]  # hands_flapping
    ohf_windows = tal_5class_df[tal_5class_df['primary_label'] == 2]  # one_hand_flap
    
    if 'tiou_one_hand_flap' in tal_5class_df.columns:
        hf_with_ohf_overlap = (hf_windows['tiou_one_hand_flap'] > 0).sum()
        results["hf_ohf_overlap"]["hf_windows_with_ohf_tiou"] = int(hf_with_ohf_overlap)
        results["hf_ohf_overlap"]["hf_windows_total"] = len(hf_windows)
        results["hf_ohf_overlap"]["hf_pct_with_ohf"] = float(hf_with_ohf_overlap / len(hf_windows) * 100) if len(hf_windows) > 0 else 0
    
    if 'tiou_hands_flapping' in tal_5class_df.columns:
        ohf_with_hf_overlap = (ohf_windows['tiou_hands_flapping'] > 0).sum()
        results["hf_ohf_overlap"]["ohf_windows_with_hf_tiou"] = int(ohf_with_hf_overlap)
        results["hf_ohf_overlap"]["ohf_windows_total"] = len(ohf_windows)
        results["hf_ohf_overlap"]["ohf_pct_with_hf"] = float(ohf_with_hf_overlap / len(ohf_windows) * 100) if len(ohf_windows) > 0 else 0
    
    return results


def generate_summary_stats(df: pd.DataFrame) -> Dict:
    """
    Generate overall summary statistics.
    
    Returns:
        Dictionary with summary stats
    """
    pred_df = df[df['is_error'].notna()].copy()
    
    results = {
        "total_windows": len(df),
        "windows_with_predictions": len(pred_df),
        "total_errors": int(pred_df['is_error'].sum()),
        "overall_accuracy": float(1 - pred_df['is_error'].mean()),
        "rmm_windows": int(df['is_rmm'].sum()),
        "background_windows": int((~df['is_rmm']).sum()),
        "preprocessing": {
            "sam3_issues_total": int(df['sam3_issue'].sum()),
            "sam3_issues_pct": float(df['sam3_issue'].mean() * 100),
            "pose_issues_total": int(df['pose_issue'].sum()),
            "pose_issues_pct": float(df['pose_issue'].mean() * 100),
            "sam3_issues_rmm_only": int(df[df['is_rmm']]['sam3_issue'].sum()),
            "sam3_issues_rmm_pct": float(df[df['is_rmm']]['sam3_issue'].mean() * 100),
            "pose_issues_rmm_only": int(df[df['is_rmm']]['pose_issue'].sum()),
            "pose_issues_rmm_pct": float(df[df['is_rmm']]['pose_issue'].mean() * 100),
        },
        "by_class": {}
    }
    
    # Per-class stats
    for label, name in CLASS_NAMES_4CLASS.items():
        class_df = pred_df[pred_df['primary_label'] == label]
        if len(class_df) > 0:
            results["by_class"][name] = {
                "n_windows": len(class_df),
                "n_errors": int(class_df['is_error'].sum()),
                "accuracy": float(1 - class_df['is_error'].mean()),
                "sam3_issue_pct": float(class_df['sam3_issue'].mean() * 100),
                "pose_issue_pct": float(class_df['pose_issue'].mean() * 100)
            }
    
    return results


def create_visualizations(analysis_results: Dict, output_dir: Path):
    """
    Create visualization plots for the analysis.
    
    Args:
        analysis_results: Dictionary with all analysis results
        output_dir: Directory to save figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Factor impact comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    factors = []
    error_diffs = []
    
    factor_analyses = [
        analysis_results.get('preprocessing_impact', {}),
        analysis_results.get('video_quality', {}).get('quality_factors', {}),
        analysis_results.get('scene_complexity', {}).get('complexity_factors', {})
    ]
    
    for factor_dict in factor_analyses:
        for name, data in factor_dict.items():
            if isinstance(data, dict) and 'error_rate_diff' in data:
                factors.append(name)
                error_diffs.append(data['error_rate_diff'] * 100)
    
    if factors:
        colors = ['#E57373' if x > 0 else '#81C784' for x in error_diffs]
        bars = ax.barh(factors, error_diffs, color=colors, alpha=0.8)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Error Rate Difference (percentage points)')
        ax.set_title('Impact of Various Factors on Error Rate\n(Positive = Higher Error When Factor Present)')
        plt.tight_layout()
        plt.savefig(output_dir / 'fig_failure_factor_impact.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'fig_failure_factor_impact.pdf', bbox_inches='tight')
        plt.close()
    
    # 2. Error rate by number of overlapping classes
    multi_label = analysis_results.get('multi_label_overlap', {}).get('error_by_n_overlapping', {})
    if multi_label:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        n_overlap = list(multi_label.keys())
        error_rates = [multi_label[n]['error_rate'] * 100 for n in n_overlap]
        counts = [multi_label[n]['n_windows'] for n in n_overlap]
        
        bars = ax.bar(n_overlap, error_rates, color='#64B5F6', alpha=0.8)
        ax.set_xlabel('Number of Overlapping RMM Classes')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title('Error Rate by Multi-Label Overlap')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            ax.annotate(f'n={count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fig_failure_multi_label.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'fig_failure_multi_label.pdf', bbox_inches='tight')
        plt.close()
    
    print(f"Saved visualizations to {output_dir}")


def run_full_analysis() -> Dict:
    """
    Run the complete failure analysis pipeline.
    
    Returns:
        Dictionary with all analysis results
    """
    print("Loading data sources...")
    
    # Load all data
    print("  Loading TAL splits (4-class)...")
    tal_df = load_tal_splits(n_classes=4)
    print(f"    Loaded {len(tal_df)} windows")
    
    print("  Loading TAL splits (5-class)...")
    tal_5class_df = load_tal_splits(n_classes=5)
    print(f"    Loaded {len(tal_5class_df)} windows")
    
    print("  Loading video ratings...")
    try:
        ratings_df = load_video_ratings()
        print(f"    Loaded {len(ratings_df)} video ratings")
    except Exception as e:
        print(f"    Warning: Could not load video ratings: {e}")
        ratings_df = pd.DataFrame()
    
    print("  Loading segment annotations...")
    segments_df = load_segment_annotations()
    print(f"    Loaded {len(segments_df)} segment annotations")
    
    print("  Loading SAM3 annotations...")
    sam3_df = load_sam3_annotations()
    print(f"    Loaded {len(sam3_df)} SAM3 annotations")
    
    # Load predictions from all folds
    print("  Loading model predictions...")
    all_predictions = []
    for fold in range(3):
        try:
            pred_df = load_model_predictions(fold)
            pred_df['fold'] = fold
            all_predictions.append(pred_df)
            print(f"    Fold {fold}: {len(pred_df)} predictions")
        except FileNotFoundError as e:
            print(f"    Warning: {e}")
    
    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
    else:
        predictions_df = pd.DataFrame()
    
    # Merge all data
    print("\nMerging data sources...")
    merged_df = merge_all_data(tal_df, ratings_df, segments_df, sam3_df, predictions_df)
    print(f"  Merged dataset: {len(merged_df)} rows")
    
    # Run analyses
    print("\nRunning analyses...")
    
    results = {
        "summary": generate_summary_stats(merged_df),
        "preprocessing_impact": {},
        "multi_label_overlap": analyze_multi_label_overlap(merged_df),
        "annotation_ambiguity": analyze_annotation_ambiguity(merged_df, sam3_df),
        "video_quality": analyze_video_quality_impact(merged_df),
        "scene_complexity": analyze_scene_complexity(merged_df),
        "five_class_confusion": analyze_5class_confusion(tal_5class_df)
    }
    
    # Analyze preprocessing factors
    print("  Analyzing preprocessing impact...")
    preproc_factors = [
        ('sam3_issue', 'SAM3 Mask Issues'),
        ('pose_issue', 'Pose Keypoint Issues'),
        ('any_preproc_issue', 'Any Preprocessing Issue')
    ]
    
    for col, name in preproc_factors:
        results["preprocessing_impact"][name] = analyze_factor_impact(merged_df, col, name)
    
    print("\nAnalysis complete!")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="TAL Failure Analysis")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    
    # Run analysis
    results = run_full_analysis()
    
    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = TABLES_DIR / "tal_failure_analysis.json"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved analysis results: {json_path}")
    
    # Create visualizations
    create_visualizations(results, FIGS_DIR)
    
    # Print summary
    print("\n" + "="*60)
    print("FAILURE ANALYSIS SUMMARY")
    print("="*60)
    
    summary = results['summary']
    print(f"\nTotal windows analyzed: {summary['total_windows']}")
    print(f"Windows with predictions: {summary['windows_with_predictions']}")
    print(f"Overall accuracy: {summary['overall_accuracy']*100:.1f}%")
    
    print(f"\nPreprocessing Issues:")
    print(f"  SAM3 issues (all): {summary['preprocessing']['sam3_issues_pct']:.1f}%")
    print(f"  SAM3 issues (RMM only): {summary['preprocessing']['sam3_issues_rmm_pct']:.1f}%")
    print(f"  Pose issues (all): {summary['preprocessing']['pose_issues_pct']:.1f}%")
    print(f"  Pose issues (RMM only): {summary['preprocessing']['pose_issues_rmm_pct']:.1f}%")
    
    print(f"\nMulti-label Overlap:")
    ml = results['multi_label_overlap']
    print(f"  Windows with >1 overlapping classes: {ml['total_with_overlap']} ({ml['pct_with_overlap']:.1f}%)")
    
    print(f"\n5-Class Confusion (hands_flapping vs one_hand_flap):")
    hf_ohf = results['five_class_confusion']['hf_ohf_overlap']
    if hf_ohf:
        print(f"  hands_flapping windows with one_hand_flap tIoU: {hf_ohf.get('hf_pct_with_ohf', 0):.1f}%")
        print(f"  one_hand_flap windows with hands_flapping tIoU: {hf_ohf.get('ohf_pct_with_hf', 0):.1f}%")


if __name__ == "__main__":
    main()
