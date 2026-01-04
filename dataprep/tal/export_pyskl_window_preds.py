#!/usr/bin/env python3
"""
Export pyskl test outputs to standardized window prediction CSV.

This script converts pyskl test.py --out results (scores.pkl) into the
common window_level_preds.csv format used by the TAL evaluation pipeline.

Works with any pyskl model:
- PoseC3D (e.g., SlowOnly R50)
- ST-GCN / STGCN++
- Any other pyskl skeleton-based model

The output format matches V-JEPA's window_level_preds.csv:
    window_id, video_key, start_sec, end_sec, score_class0, score_class1, ...

Example usage:
    # Export PoseC3D scores
    python export_pyskl_window_preds.py \
        --ann-pkl data/sails/tal/cv_4class/5class_windows_conf04/fold0.pkl \
        --scores-pkl work_dirs/posec3d/tal/fold0/result.pkl \
        --window-csv /path/to/tal/splits_cv_4class/fold_0_val_windows.csv \
        --out-csv window_level_preds.csv

    # Export STGCN++ scores (same process)
    python export_pyskl_window_preds.py \
        --ann-pkl data/sails/tal/cv_4class/5class_windows_conf04/fold0.pkl \
        --scores-pkl work_dirs/stgcnpp/tal/fold0/result.pkl \
        --window-csv /path/to/tal/splits_cv_4class/fold_0_val_windows.csv \
        --out-csv window_level_preds.csv
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_pyskl_scores(scores_pkl: Path) -> np.ndarray:
    """
    Load scores from pyskl test.py --out output.
    
    The output format varies slightly:
    - Usually: list of (N,) arrays (one per sample, N=num_classes after softmax)
    - Sometimes: single (M, N) array where M=num_samples
    
    Args:
        scores_pkl: Path to scores pickle file.
    
    Returns:
        NumPy array of shape (num_samples, num_classes).
    """
    with open(scores_pkl, "rb") as f:
        data = pickle.load(f)
    
    if isinstance(data, np.ndarray):
        # Already an array
        if data.ndim == 1:
            raise ValueError(f"Scores array has unexpected shape: {data.shape}")
        return data
    
    if isinstance(data, list):
        # List of arrays
        if len(data) == 0:
            raise ValueError("Empty scores list")
        
        # Check if first element is an array
        first = data[0]
        if isinstance(first, np.ndarray):
            # Stack into (M, N) array
            return np.stack(data, axis=0)
        else:
            raise ValueError(f"Unexpected element type in scores list: {type(first)}")
    
    raise ValueError(f"Unexpected scores format: {type(data)}")


def load_pyskl_annotations(ann_pkl: Path) -> List[Dict[str, Any]]:
    """
    Load annotations from pyskl annotation pickle.
    
    Args:
        ann_pkl: Path to annotation pickle file.
    
    Returns:
        List of annotation dictionaries.
    """
    with open(ann_pkl, "rb") as f:
        data = pickle.load(f)
    
    annotations = data.get("annotations", [])
    split_info = data.get("split", {})
    
    # Get val split sample IDs (used during testing)
    val_ids = set(split_info.get("val", []))
    
    # Filter to val split only
    val_annotations = []
    for ann in annotations:
        frame_dir = ann.get("frame_dir", "")
        if frame_dir in val_ids:
            val_annotations.append(ann)
    
    return val_annotations


def extract_window_id(frame_dir: str) -> str:
    """
    Extract window_id from pyskl frame_dir.
    
    The frame_dir format in TAL pickles is typically:
        {child_id}_{video_stem}__t{start_ms}_{end_ms}
    
    Args:
        frame_dir: The frame_dir from pyskl annotation.
    
    Returns:
        window_id string.
    """
    # frame_dir might have a prefix path; extract just the identifier
    if "/" in frame_dir:
        frame_dir = frame_dir.split("/")[-1]
    if "\\" in frame_dir:
        frame_dir = frame_dir.split("\\")[-1]
    
    return frame_dir


def load_window_metadata(window_csv: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load window metadata from TAL window split CSV.
    
    Args:
        window_csv: Path to window CSV file.
    
    Returns:
        Dict mapping window_id -> {video_key, start_sec, end_sec, ...}.
    """
    df = pd.read_csv(window_csv)
    
    metadata = {}
    for _, row in df.iterrows():
        window_id = row.get("window_id", "")
        if not window_id:
            continue
        
        metadata[window_id] = {
            "video_key": row.get("video_key", ""),
            "start_sec": float(row.get("start_sec", 0)),
            "end_sec": float(row.get("end_sec", 0)),
            "is_background": int(row.get("is_background", 0)),
            "primary_label": int(row.get("primary_label", -1)),
            "mask_severity": row.get("mask_severity", ""),
            "pose_severity": row.get("pose_severity", ""),
        }
    
    return metadata


def export_window_preds(
    ann_pkl: Path,
    scores_pkl: Path,
    out_csv: Path,
    window_csv: Optional[Path] = None,
    apply_softmax: bool = False,
) -> pd.DataFrame:
    """
    Export pyskl test scores to window prediction CSV.
    
    Args:
        ann_pkl: Path to annotation pickle.
        scores_pkl: Path to scores pickle.
        out_csv: Output CSV path.
        window_csv: Optional TAL window CSV for metadata.
        apply_softmax: Whether to apply softmax to raw logits.
    
    Returns:
        DataFrame with window predictions.
    """
    # Load data
    print(f"Loading annotations from: {ann_pkl}")
    annotations = load_pyskl_annotations(ann_pkl)
    print(f"  Found {len(annotations)} validation samples")
    
    print(f"Loading scores from: {scores_pkl}")
    scores = load_pyskl_scores(scores_pkl)
    print(f"  Scores shape: {scores.shape}")
    
    if len(annotations) != len(scores):
        raise ValueError(
            f"Annotation count ({len(annotations)}) != score count ({len(scores)}). "
            "Ensure you tested on the val split only."
        )
    
    # Apply softmax if needed
    if apply_softmax:
        print("  Applying softmax to scores...")
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    num_classes = scores.shape[1]
    print(f"  Number of classes: {num_classes}")
    
    # Load window metadata if provided
    window_meta = {}
    if window_csv:
        print(f"Loading window metadata from: {window_csv}")
        window_meta = load_window_metadata(window_csv)
        print(f"  Found metadata for {len(window_meta)} windows")
    
    # Build output records
    records = []
    missing_meta = 0
    
    for i, ann in enumerate(annotations):
        frame_dir = ann.get("frame_dir", "")
        window_id = extract_window_id(frame_dir)
        label = ann.get("label", -1)
        
        # Get metadata
        meta = window_meta.get(window_id, {})
        if not meta and window_csv:
            missing_meta += 1
        
        record = {
            "window_id": window_id,
            "video_key": meta.get("video_key", ""),
            "start_sec": meta.get("start_sec", 0.0),
            "end_sec": meta.get("end_sec", 0.0),
            "label_id": label,
            "is_background": meta.get("is_background", 0),
            "mask_severity": meta.get("mask_severity", ""),
            "pose_severity": meta.get("pose_severity", ""),
        }
        
        # Add score columns
        for c in range(num_classes):
            record[f"score_class{c}"] = float(scores[i, c])
        
        # Add predicted class
        record["pred_top1"] = int(np.argmax(scores[i]))
        record["pred_conf"] = float(np.max(scores[i]))
        
        records.append(record)
    
    if missing_meta > 0:
        print(f"  Warning: {missing_meta} windows missing metadata")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nExported {len(df)} predictions to: {out_csv}")
    
    # Summary
    if num_classes >= 5:
        # TAL with background
        bg_count = (df["pred_top1"] == num_classes - 1).sum()
        rmm_count = len(df) - bg_count
        print(f"  RMM predictions: {rmm_count}, Background predictions: {bg_count}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Export pyskl test scores to window prediction CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # PoseC3D
    python export_pyskl_window_preds.py \\
        --ann-pkl data/sails/tal/cv_4class/5class_windows_conf04/fold0.pkl \\
        --scores-pkl work_dirs/posec3d/tal/fold0/result.pkl \\
        --window-csv ../splits_cv_4class/fold_0_val_windows.csv \\
        --out-csv fold0_window_preds.csv

    # STGCN++
    python export_pyskl_window_preds.py \\
        --ann-pkl data/sails/tal/cv_4class/5class_windows_conf04/fold0.pkl \\
        --scores-pkl work_dirs/stgcnpp/tal/fold0/result.pkl \\
        --window-csv ../splits_cv_4class/fold_0_val_windows.csv \\
        --out-csv stgcn_fold0_window_preds.csv
        """,
    )
    
    parser.add_argument(
        "--ann-pkl",
        type=Path,
        required=True,
        help="Path to pyskl annotation pickle file.",
    )
    parser.add_argument(
        "--scores-pkl",
        type=Path,
        required=True,
        help="Path to pyskl test.py --out scores pickle file.",
    )
    parser.add_argument(
        "--window-csv",
        type=Path,
        default=None,
        help="Path to TAL window CSV for metadata (video_key, start_sec, end_sec).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output path for window predictions CSV.",
    )
    parser.add_argument(
        "--apply-softmax",
        action="store_true",
        help="Apply softmax to raw logits (use if scores are not already probabilities).",
    )
    
    args = parser.parse_args()
    
    export_window_preds(
        ann_pkl=args.ann_pkl,
        scores_pkl=args.scores_pkl,
        out_csv=args.out_csv,
        window_csv=args.window_csv,
        apply_softmax=args.apply_softmax,
    )


if __name__ == "__main__":
    main()

