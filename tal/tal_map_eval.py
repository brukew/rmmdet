#!/usr/bin/env python3
"""
Segment-level mAP evaluation for Temporal Action Localization.

This module implements ActivityNet-style mean Average Precision (mAP) evaluation
at multiple tIoU thresholds for TAL predictions.

Key features:
- Half-open interval semantics [start, end) consistent with window generation
- Per-video greedy matching (sorted by score descending)
- Per-class AP with proper handling of missing classes
- mAP at tIoU thresholds {0.3, 0.5, 0.7}

Example usage:
    from tal_map_eval import compute_map, load_gt_segments
    
    gt_segments = load_gt_segments(splits_root, task="4class", mode="cv")
    metrics = compute_map(pred_segments_df, gt_segments, tiou_thresholds=[0.3, 0.5, 0.7])
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# Label maps (must match make_tal_window_splits.py)
LABEL_MAP_4CLASS = {
    "hands flapping": 0,
    "jumping": 1,
    "rocking": 2,
    "spinning": 3,
}

LABEL_MAP_5CLASS = {
    "hands flapping": 0,
    "jumping": 1,
    "one hand flap": 2,
    "rocking": 3,
    "spinning": 4,
}

ID2LABEL_4CLASS = {v: k for k, v in LABEL_MAP_4CLASS.items()}
ID2LABEL_5CLASS = {v: k for k, v in LABEL_MAP_5CLASS.items()}


@dataclass
class GTSegment:
    """A ground-truth segment annotation."""
    segment_id: str
    video_key: str
    label_idx: int
    label_name: str
    start_sec: float  # Half-open start
    end_sec: float    # Half-open end (already converted: original_end + 1.0)


@dataclass
class PredSegment:
    """A predicted segment."""
    video_key: str
    class_id: int
    start_sec: float
    end_sec: float
    score: float


def normalize_video_file(video_file: str) -> str:
    """Normalize video_file path for consistent matching."""
    normalized = video_file.replace("\\", "/").lstrip("/")
    prefixes = [
        "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/",
        "Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/",
    ]
    for prefix in prefixes:
        if normalized.lower().startswith(prefix.lower()):
            normalized = normalized[len(prefix):]
    return normalized


def compute_tiou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """
    Compute temporal Intersection-over-Union between two segments.
    
    Uses half-open interval semantics: [start, end).
    
    Args:
        pred_start: Predicted segment start (inclusive).
        pred_end: Predicted segment end (exclusive).
        gt_start: Ground truth segment start (inclusive).
        gt_end: Ground truth segment end (exclusive).
    
    Returns:
        tIoU value in [0, 1].
    """
    # Intersection
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    
    # Union
    pred_len = pred_end - pred_start
    gt_len = gt_end - gt_start
    union = pred_len + gt_len - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def load_gt_segments(
    splits_root: Path,
    task: str = "4class",
    mode: str = "cv",
) -> Dict[str, List[GTSegment]]:
    """
    Load ground-truth segments from split CSVs.
    
    Args:
        splits_root: Root directory containing split subdirectories.
        task: "4class" or "5class".
        mode: "cv" or "single".
    
    Returns:
        Dict mapping video_key -> list of GTSegment.
    """
    if task == "4class":
        label_map = LABEL_MAP_4CLASS
        split_dir = "cv_splits_4class" if mode == "cv" else "single_split_4class"
    else:
        label_map = LABEL_MAP_5CLASS
        split_dir = "cv_splits" if mode == "cv" else "single_split"
    
    split_path = splits_root / split_dir
    
    # Find all CSVs
    if mode == "cv":
        csv_files = sorted(split_path.glob("fold_*_*.csv"))
    else:
        csv_files = [split_path / f"{s}.csv" for s in ("train", "val", "test")]
        csv_files = [f for f in csv_files if f.exists()]
    
    segments_by_video: Dict[str, List[GTSegment]] = defaultdict(list)
    seen_segment_ids: Set[str] = set()
    
    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                segment_id = row.get("segment_id", "")
                
                # Skip duplicates (same segment appears in multiple fold CSVs)
                if segment_id in seen_segment_ids:
                    continue
                seen_segment_ids.add(segment_id)
                
                video_file = row.get("video_file", "")
                video_key = normalize_video_file(video_file)
                label_name = row.get("rmm_type", "")
                
                if label_name not in label_map:
                    continue  # Skip labels not in this task's label set
                
                try:
                    start_sec = float(row.get("start_sec", 0))
                    end_sec = float(row.get("end_sec", 0))
                except ValueError:
                    continue
                
                # Convert to half-open interval
                # Original annotation "2-4 means 2-5", so end_exclusive = end_sec + 1.0
                start_exclusive = start_sec
                end_exclusive = end_sec + 1.0
                
                gt = GTSegment(
                    segment_id=segment_id,
                    video_key=video_key,
                    label_idx=label_map[label_name],
                    label_name=label_name,
                    start_sec=start_exclusive,
                    end_sec=end_exclusive,
                )
                segments_by_video[video_key].append(gt)
    
    return dict(segments_by_video)


def compute_ap_for_class(
    preds: List[PredSegment],
    gt_by_video: Dict[str, List[GTSegment]],
    class_id: int,
    tiou_threshold: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Average Precision for a single class at a given tIoU threshold.
    
    Uses greedy matching: predictions sorted by score (descending), each GT
    can only be matched once.
    
    Args:
        preds: All predicted segments (will be filtered to class_id).
        gt_by_video: GT segments indexed by video_key.
        class_id: Which class to evaluate.
        tiou_threshold: Minimum tIoU for a match to count as TP.
    
    Returns:
        Tuple of (AP, details_dict).
    """
    # Filter predictions for this class
    class_preds = [p for p in preds if p.class_id == class_id]
    
    # Count GT segments for this class
    n_gt = 0
    gt_matched = {}  # {(video_key, segment_idx): bool}
    for video_key, gt_list in gt_by_video.items():
        for i, gt in enumerate(gt_list):
            if gt.label_idx == class_id:
                n_gt += 1
                gt_matched[(video_key, i)] = False
    
    if n_gt == 0:
        # No GT segments for this class
        return float("nan"), {"n_gt": 0, "n_pred": len(class_preds), "note": "no_gt"}
    
    if len(class_preds) == 0:
        # No predictions
        return 0.0, {"n_gt": n_gt, "n_pred": 0, "note": "no_pred"}
    
    # Sort predictions by score (descending)
    class_preds = sorted(class_preds, key=lambda p: -p.score)
    
    # Greedy matching
    tp = np.zeros(len(class_preds))
    fp = np.zeros(len(class_preds))
    
    for pred_idx, pred in enumerate(class_preds):
        video_key = pred.video_key
        
        if video_key not in gt_by_video:
            # No GT for this video -> FP
            fp[pred_idx] = 1
            continue
        
        gt_list = gt_by_video[video_key]
        
        # Find best matching GT segment (highest tIoU)
        best_tiou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_list):
            if gt.label_idx != class_id:
                continue
            if gt_matched.get((video_key, gt_idx), False):
                continue  # Already matched
            
            tiou = compute_tiou(pred.start_sec, pred.end_sec, gt.start_sec, gt.end_sec)
            if tiou > best_tiou:
                best_tiou = tiou
                best_gt_idx = gt_idx
        
        if best_tiou >= tiou_threshold:
            # TP
            tp[pred_idx] = 1
            gt_matched[(video_key, best_gt_idx)] = True
        else:
            # FP
            fp[pred_idx] = 1
    
    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / n_gt
    
    # Compute AP using interpolation (11-point or all-point)
    # We use all-point interpolation for more accurate AP
    ap = compute_ap_from_pr(precision, recall)
    
    details = {
        "n_gt": n_gt,
        "n_pred": len(class_preds),
        "n_tp": int(tp.sum()),
        "n_fp": int(fp.sum()),
        "precision_at_threshold": float(precision[-1]) if len(precision) > 0 else 0.0,
        "recall_at_threshold": float(recall[-1]) if len(recall) > 0 else 0.0,
    }
    
    return ap, details


def compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Compute Average Precision from precision-recall curve.
    
    Uses the all-point interpolation method (PASCAL VOC 2010+).
    
    Args:
        precision: Precision values at each operating point.
        recall: Recall values at each operating point.
    
    Returns:
        AP value.
    """
    if len(precision) == 0 or len(recall) == 0:
        return 0.0
    
    # Append sentinel values
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum areas under the curve
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return float(ap)


def compute_map(
    pred_df: pd.DataFrame,
    gt_by_video: Dict[str, List[GTSegment]],
    class_ids: List[int] = [0, 1, 2, 3],
    tiou_thresholds: List[float] = [0.3, 0.5, 0.7],
) -> Dict[str, Any]:
    """
    Compute mAP and Recall at multiple tIoU thresholds.
    
    Args:
        pred_df: Predicted segments DataFrame with columns:
                 video_key, class_id, start_sec, end_sec, score
        gt_by_video: GT segments indexed by video_key.
        class_ids: Which classes to evaluate.
        tiou_thresholds: tIoU thresholds for evaluation.
    
    Returns:
        Dictionary with:
            - mAP@{threshold} for each threshold
            - Recall@{threshold} for each threshold (average recall across classes)
            - avg_mAP (mean across thresholds)
            - avg_Recall (mean across thresholds)
            - per_class_ap[threshold][class_id] = AP
            - details[threshold][class_id] = details dict
    """
    # Convert DataFrame to list of PredSegment
    preds = []
    for _, row in pred_df.iterrows():
        preds.append(PredSegment(
            video_key=row["video_key"],
            class_id=int(row["class_id"]),
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            score=float(row["score"]),
        ))
    
    results = {
        "per_class_ap": {},
        "details": {},
    }
    
    all_maps = []
    all_recalls = []
    
    for tiou_thr in tiou_thresholds:
        thr_key = f"tIoU={tiou_thr}"
        results["per_class_ap"][thr_key] = {}
        results["details"][thr_key] = {}
        
        aps = []
        recalls = []
        for class_id in class_ids:
            ap, details = compute_ap_for_class(preds, gt_by_video, class_id, tiou_thr)
            results["per_class_ap"][thr_key][class_id] = ap
            results["details"][thr_key][class_id] = details
            
            if not np.isnan(ap):
                aps.append(ap)
        
            # Collect recall for classes that have GT segments
            if details.get("n_gt", 0) > 0:
                recalls.append(details.get("recall_at_threshold", 0.0))
        
        # Compute mAP for this threshold
        if aps:
            map_val = np.mean(aps)
        else:
            map_val = float("nan")
        
        results[f"mAP@{tiou_thr}"] = map_val
        if not np.isnan(map_val):
            all_maps.append(map_val)
        
        # Compute average recall for this threshold
        if recalls:
            recall_val = np.mean(recalls)
        else:
            recall_val = float("nan")
        
        results[f"Recall@{tiou_thr}"] = recall_val
        if not np.isnan(recall_val):
            all_recalls.append(recall_val)
    
    # Average mAP across thresholds
    if all_maps:
        results["avg_mAP"] = np.mean(all_maps)
    else:
        results["avg_mAP"] = float("nan")
    
    # Average Recall across thresholds
    if all_recalls:
        results["avg_Recall"] = np.mean(all_recalls)
    else:
        results["avg_Recall"] = float("nan")
    
    return results


def format_metrics_report(
    metrics: Dict[str, Any],
    class_ids: List[int] = [0, 1, 2, 3],
    id2label: Dict[int, str] = None,
    detail_tiou: float = 0.5,
) -> str:
    """
    Format metrics as a human-readable report.
    
    Args:
        metrics: Output from compute_map().
        class_ids: Class IDs to include.
        id2label: Optional mapping from class ID to label name.
        detail_tiou: tIoU threshold to use for detailed per-class table.
    
    Returns:
        Formatted string report.
    """
    if id2label is None:
        id2label = ID2LABEL_4CLASS
    
    lines = ["=" * 60, "TAL Evaluation Results", "=" * 60, ""]
    
    # Summary mAP and Recall side by side
    lines.append("Summary:")
    
    # Collect mAP and Recall keys
    tiou_thresholds = sorted([float(k.replace("mAP@", "")) for k in metrics.keys() if k.startswith("mAP@")])
    
    for tiou_thr in tiou_thresholds:
        map_val = metrics.get(f"mAP@{tiou_thr}", float("nan"))
        recall_val = metrics.get(f"Recall@{tiou_thr}", float("nan"))
        
        map_str = f"{map_val:.4f}" if not np.isnan(map_val) else "N/A"
        recall_str = f"{recall_val:.4f}" if not np.isnan(recall_val) else "N/A"
        
        lines.append(f"  mAP@{tiou_thr}: {map_str:>8}    Recall@{tiou_thr}: {recall_str:>8}")
    
    avg_map = metrics.get("avg_mAP", float("nan"))
    avg_recall = metrics.get("avg_Recall", float("nan"))
    avg_map_str = f"{avg_map:.4f}" if not np.isnan(avg_map) else "N/A"
    avg_recall_str = f"{avg_recall:.4f}" if not np.isnan(avg_recall) else "N/A"
    lines.append(f"  avg_mAP:  {avg_map_str:>8}    avg_Recall:  {avg_recall_str:>8}")
    lines.append("")
    
    # Per-class AP breakdown
    lines.append("Per-class AP:")
    per_class_ap = metrics.get("per_class_ap", {})
    
    for thr_key in sorted(per_class_ap.keys()):
        lines.append(f"\n  {thr_key}:")
        for class_id in class_ids:
            ap = per_class_ap[thr_key].get(class_id, float("nan"))
            label = id2label.get(class_id, f"class_{class_id}")
            if np.isnan(ap):
                lines.append(f"    {label}: N/A")
            else:
                lines.append(f"    {label}: {ap:.4f}")
    
    lines.append("")
    
    # Per-class Detection Metrics table (at the specified detail_tiou)
    details = metrics.get("details", {})
    detail_key = f"tIoU={detail_tiou}"
    
    if detail_key in details:
        lines.append(f"Per-class Detection Metrics ({detail_key}):")
        lines.append("")
        
        # Find max label length for alignment
        max_label_len = max(len(id2label.get(cid, f"class_{cid}")) for cid in class_ids)
        max_label_len = max(max_label_len, 5)  # Minimum for "TOTAL"
        
        # Header
        header = f"  {'Class':<{max_label_len}}  {'n_GT':>6}  {'n_Pred':>7}  {'n_TP':>6}  {'n_FP':>6}  {'Prec':>7}  {'Recall':>7}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        
        # Per-class rows
        total_n_gt = 0
        total_n_pred = 0
        total_n_tp = 0
        total_n_fp = 0
        
        for class_id in class_ids:
            d = details[detail_key].get(class_id, {})
            label = id2label.get(class_id, f"class_{class_id}")
            
            n_gt = d.get("n_gt", 0)
            n_pred = d.get("n_pred", 0)
            n_tp = d.get("n_tp", 0)
            n_fp = d.get("n_fp", 0)
            precision = d.get("precision_at_threshold", 0.0)
            recall = d.get("recall_at_threshold", 0.0)
            
            total_n_gt += n_gt
            total_n_pred += n_pred
            total_n_tp += n_tp
            total_n_fp += n_fp
            
            lines.append(
                f"  {label:<{max_label_len}}  {n_gt:>6}  {n_pred:>7}  {n_tp:>6}  {n_fp:>6}  {precision:>7.3f}  {recall:>7.3f}"
            )
        
        # Total row
        lines.append("  " + "-" * (len(header) - 2))
        total_precision = total_n_tp / total_n_pred if total_n_pred > 0 else 0.0
        total_recall = total_n_tp / total_n_gt if total_n_gt > 0 else 0.0
        lines.append(
            f"  {'TOTAL':<{max_label_len}}  {total_n_gt:>6}  {total_n_pred:>7}  {total_n_tp:>6}  {total_n_fp:>6}  {total_precision:>7.3f}  {total_recall:>7.3f}"
        )
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def save_metrics(
    metrics: Dict[str, Any],
    out_dir: Path,
    class_ids: List[int] = [0, 1, 2, 3],
    id2label: Dict[int, str] = None,
) -> None:
    """
    Save metrics to files.
    
    Args:
        metrics: Output from compute_map().
        out_dir: Output directory.
        class_ids: Class IDs.
        id2label: Optional mapping from class ID to label name.
    
    Outputs:
        - metrics.json: Full metrics dictionary
        - per_class_metrics.csv: Comprehensive per-class metrics (AP, n_gt, n_pred, n_tp, n_fp, precision, recall)
        - report.txt: Human-readable report
    """
    if id2label is None:
        id2label = ID2LABEL_4CLASS
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = out_dir / "metrics.json"
    
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(convert_for_json(metrics), f, indent=2)
    
    # Save comprehensive per-class metrics as CSV
    per_class_ap = metrics.get("per_class_ap", {})
    details = metrics.get("details", {})
    rows = []
    
    for thr_key in sorted(per_class_ap.keys()):
        for class_id in class_ids:
            ap = per_class_ap[thr_key].get(class_id, float("nan"))
            label = id2label.get(class_id, f"class_{class_id}")
            
            # Get detailed metrics from the details dict
            d = details.get(thr_key, {}).get(class_id, {})
            n_gt = d.get("n_gt", 0)
            n_pred = d.get("n_pred", 0)
            n_tp = d.get("n_tp", 0)
            n_fp = d.get("n_fp", 0)
            precision = d.get("precision_at_threshold", 0.0)
            recall = d.get("recall_at_threshold", 0.0)
            
            rows.append({
                "tiou_threshold": thr_key,
                "class_id": class_id,
                "class_name": label,
                "AP": ap if not np.isnan(ap) else None,
                "n_gt": n_gt,
                "n_pred": n_pred,
                "n_tp": n_tp,
                "n_fp": n_fp,
                "precision": precision,
                "recall": recall,
            })
    
    if rows:
        csv_path = out_dir / "per_class_metrics.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    
    # Save human-readable report
    report = format_metrics_report(metrics, class_ids, id2label)
    report_path = out_dir / "report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)


# ============================================================================
# Convenience functions for quick evaluation
# ============================================================================

def evaluate_predictions(
    pred_segments_df: pd.DataFrame,
    splits_root: Path,
    task: str = "4class",
    mode: str = "cv",
    tiou_thresholds: List[float] = [0.3, 0.5, 0.7],
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate predicted segments against GT.
    
    Args:
        pred_segments_df: DataFrame with columns:
                          video_key, class_id, start_sec, end_sec, score
        splits_root: Root directory containing split CSVs.
        task: "4class" or "5class".
        mode: "cv" or "single".
        tiou_thresholds: tIoU thresholds.
        out_dir: Optional output directory for saving results.
    
    Returns:
        Metrics dictionary.
    """
    # Load GT
    gt_by_video = load_gt_segments(splits_root, task, mode)
    
    # Determine class IDs
    if task == "4class":
        class_ids = [0, 1, 2, 3]
        id2label = ID2LABEL_4CLASS
    else:
        class_ids = [0, 1, 2, 3, 4]
        id2label = ID2LABEL_5CLASS
    
    # Compute metrics
    metrics = compute_map(pred_segments_df, gt_by_video, class_ids, tiou_thresholds)
    
    # Print report
    report = format_metrics_report(metrics, class_ids, id2label)
    print(report)
    
    # Save if requested
    if out_dir:
        save_metrics(metrics, out_dir, class_ids, id2label)
        print(f"\nMetrics saved to: {out_dir}")
    
    return metrics

