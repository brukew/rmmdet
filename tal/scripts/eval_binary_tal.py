#!/usr/bin/env python3
"""
Evaluate binary (RMM vs Background) TAL models.

Binary models output 2 classes:
- score_class0: RMM probability (any RMM type)
- score_class1: Background probability

For evaluation:
- Predictions: Use score_class0 to detect "RMM" segments
- GT: Merge all 4 RMM types into single "RMM" class

Reports:
- Binary mAP@{0.3, 0.5, 0.7}
- Recall@{0.3, 0.5, 0.7}
- Precision, n_GT, n_TP, n_FP

Usage:
    python eval_binary_tal.py \
        --predictions /path/to/tal_format_preds.csv \
        --splits-root /path/to/splits \
        --out-dir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class GTSegment:
    """A ground-truth segment annotation."""
    segment_id: str
    video_key: str
    label_name: str  # Original RMM type
    start_sec: float
    end_sec: float


@dataclass
class PredSegment:
    """A predicted segment."""
    video_key: str
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
    """Compute temporal IoU between two segments."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    
    pred_len = pred_end - pred_start
    gt_len = gt_end - gt_start
    union = pred_len + gt_len - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def load_binary_gt_segments(
    splits_root: Path,
    mode: str = "cv",
) -> Dict[str, List[GTSegment]]:
    """
    Load GT segments for binary evaluation (all RMM types merged).
    
    Args:
        splits_root: Root directory containing split CSVs.
        mode: "cv" or "single".
    
    Returns:
        Dict mapping video_key -> list of GTSegment.
    """
    # Use 4-class splits (has all RMM types)
    split_dir = "cv_splits_4class" if mode == "cv" else "single_split_4class"
    split_path = splits_root / split_dir
    
    # Find all CSVs
    if mode == "cv":
        csv_files = sorted(split_path.glob("fold_*_*.csv"))
    else:
        csv_files = [split_path / f"{s}.csv" for s in ("train", "val", "test")]
        csv_files = [f for f in csv_files if f.exists()]
    
    segments_by_video: Dict[str, List[GTSegment]] = defaultdict(list)
    seen_segment_ids: Set[str] = set()
    
    # RMM types to include (all 4 classes)
    rmm_types = {"hands flapping", "jumping", "rocking", "spinning"}
    
    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                segment_id = row.get("segment_id", "")
                
                # Skip duplicates
                if segment_id in seen_segment_ids:
                    continue
                seen_segment_ids.add(segment_id)
                
                video_file = row.get("video_file", "")
                video_key = normalize_video_file(video_file)
                label_name = row.get("rmm_type", "")
                
                if label_name not in rmm_types:
                    continue
                
                try:
                    start_sec = float(row.get("start_sec", 0))
                    end_sec = float(row.get("end_sec", 0))
                except ValueError:
                    continue
                
                # Convert to half-open interval
                start_exclusive = start_sec
                end_exclusive = end_sec + 1.0
                
                gt = GTSegment(
                    segment_id=segment_id,
                    video_key=video_key,
                    label_name=label_name,
                    start_sec=start_exclusive,
                    end_sec=end_exclusive,
                )
                segments_by_video[video_key].append(gt)
    
    return dict(segments_by_video)


def compute_binary_ap(
    preds: List[PredSegment],
    gt_by_video: Dict[str, List[GTSegment]],
    tiou_threshold: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute AP for binary RMM detection at a given tIoU threshold.
    
    Args:
        preds: All predicted RMM segments.
        gt_by_video: GT segments indexed by video_key.
        tiou_threshold: Minimum tIoU for a match to count as TP.
    
    Returns:
        Tuple of (AP, details_dict).
    """
    # Count GT segments
    n_gt = 0
    gt_matched = {}  # {(video_key, segment_idx): bool}
    for video_key, gt_list in gt_by_video.items():
        for i, gt in enumerate(gt_list):
            n_gt += 1
            gt_matched[(video_key, i)] = False
    
    if n_gt == 0:
        return float("nan"), {"n_gt": 0, "n_pred": len(preds), "note": "no_gt"}
    
    if len(preds) == 0:
        return 0.0, {
            "n_gt": n_gt, "n_pred": 0, "n_tp": 0, "n_fp": 0,
            "precision": 0.0, "recall": 0.0, "note": "no_pred"
        }
    
    # Sort predictions by score (descending)
    preds_sorted = sorted(preds, key=lambda p: -p.score)
    
    # Greedy matching
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))
    
    for pred_idx, pred in enumerate(preds_sorted):
        video_key = pred.video_key
        
        if video_key not in gt_by_video:
            fp[pred_idx] = 1
            continue
        
        gt_list = gt_by_video[video_key]
        
        # Find best matching GT segment
        best_tiou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_list):
            if gt_matched.get((video_key, gt_idx), False):
                continue
            
            tiou = compute_tiou(pred.start_sec, pred.end_sec, gt.start_sec, gt.end_sec)
            if tiou > best_tiou:
                best_tiou = tiou
                best_gt_idx = gt_idx
        
        if best_tiou >= tiou_threshold:
            tp[pred_idx] = 1
            gt_matched[(video_key, best_gt_idx)] = True
        else:
            fp[pred_idx] = 1
    
    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall_curve = tp_cumsum / n_gt
    
    # Compute AP
    ap = compute_ap_from_pr(precision_curve, recall_curve)
    
    # Final metrics
    n_tp = int(tp.sum())
    n_fp = int(fp.sum())
    final_precision = n_tp / len(preds_sorted) if len(preds_sorted) > 0 else 0.0
    final_recall = n_tp / n_gt if n_gt > 0 else 0.0
    
    details = {
        "n_gt": n_gt,
        "n_pred": len(preds_sorted),
        "n_tp": n_tp,
        "n_fp": n_fp,
        "precision": final_precision,
        "recall": final_recall,
    }
    
    return ap, details


def compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute AP from precision-recall curve using all-point interpolation."""
    if len(precision) == 0 or len(recall) == 0:
        return 0.0
    
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return float(ap)


def postprocess_binary_to_segments(
    df: pd.DataFrame,
    threshold: float = 0.5,
    smooth_k: int = 3,
    merge_gap_sec: float = 1.0,
    min_duration_sec: float = 0.0,
) -> List[PredSegment]:
    """
    Convert binary window predictions to segments.
    
    Args:
        df: TAL-format predictions with score_class0 (RMM probability).
        threshold: Detection threshold.
        smooth_k: Smoothing window size.
        merge_gap_sec: Max gap to merge adjacent detections.
        min_duration_sec: Min segment duration.
    
    Returns:
        List of PredSegment.
    """
    all_segments = []
    
    for video_key, video_df in df.groupby("video_key"):
        video_df = video_df.sort_values("start_sec").reset_index(drop=True)
        
        if len(video_df) == 0:
            continue
        
        scores = video_df["score_class0"].values.astype(float)
        starts = video_df["start_sec"].values.astype(float)
        ends = video_df["end_sec"].values.astype(float)
        
        # Smooth scores
        if smooth_k > 1 and len(scores) >= smooth_k:
            kernel = np.ones(smooth_k) / smooth_k
            scores = np.convolve(scores, kernel, mode="same")
        
        # Threshold
        active_mask = scores >= threshold
        
        # Find contiguous runs (run_end is exclusive)
        runs = []
        in_run = False
        run_start = 0
        
        for i, active in enumerate(active_mask):
            if active and not in_run:
                run_start = i
                in_run = True
            elif not active and in_run:
                runs.append((run_start, i))
                in_run = False
        
        if in_run:
            runs.append((run_start, len(active_mask)))
        
        # Convert runs to segments
        raw_segments = []
        for run_start_idx, run_end_idx in runs:
            if run_start_idx >= len(starts) or run_end_idx <= 0:
                continue
            
            # run_end_idx is exclusive, so the last window is at run_end_idx - 1
            last_idx = min(run_end_idx - 1, len(ends) - 1)
            seg_start = starts[run_start_idx]
            seg_end = ends[last_idx]
            seg_score = float(np.max(scores[run_start_idx:run_end_idx]))
            raw_segments.append((seg_start, seg_end, seg_score))
        
        # Merge close segments
        merged = []
        for seg_start, seg_end, seg_score in raw_segments:
            if merged and seg_start - merged[-1][1] <= merge_gap_sec:
                # Merge with previous
                prev_start, prev_end, prev_score = merged[-1]
                merged[-1] = (prev_start, seg_end, max(prev_score, seg_score))
            else:
                merged.append((seg_start, seg_end, seg_score))
        
        # Filter by duration
        for seg_start, seg_end, seg_score in merged:
            if seg_end - seg_start >= min_duration_sec:
                all_segments.append(PredSegment(
                    video_key=video_key,
                    start_sec=seg_start,
                    end_sec=seg_end,
                    score=seg_score,
                ))
    
    return all_segments


def evaluate_binary_tal(
    pred_df: pd.DataFrame,
    gt_by_video: Dict[str, List[GTSegment]],
    pp_params: Dict[str, Any],
    tiou_thresholds: List[float] = [0.3, 0.5, 0.7],
) -> Dict[str, Any]:
    """
    Evaluate binary TAL predictions.
    
    Args:
        pred_df: TAL-format predictions with score_class0.
        gt_by_video: GT segments by video key.
        pp_params: Postprocessing parameters.
        tiou_thresholds: tIoU thresholds for evaluation.
    
    Returns:
        Metrics dictionary.
    """
    # Postprocess to segments
    preds = postprocess_binary_to_segments(
        pred_df,
        threshold=pp_params.get("thr", 0.5),
        smooth_k=pp_params.get("smooth_k", 3),
        merge_gap_sec=pp_params.get("merge_gap_sec", 1.0),
        min_duration_sec=pp_params.get("min_duration_sec", 0.0),
    )
    
    results = {
        "n_windows": len(pred_df),
        "n_segments": len(preds),
        "n_videos": pred_df["video_key"].nunique(),
        "postprocess_params": pp_params,
        "details": {},
    }
    
    all_maps = []
    all_recalls = []
    
    for tiou_thr in tiou_thresholds:
        thr_key = f"tIoU={tiou_thr}"
        ap, details = compute_binary_ap(preds, gt_by_video, tiou_thr)
        
        results["details"][thr_key] = details
        results[f"mAP@{tiou_thr}"] = ap
        results[f"Recall@{tiou_thr}"] = details.get("recall", 0.0)
        results[f"Precision@{tiou_thr}"] = details.get("precision", 0.0)
        
        if not np.isnan(ap):
            all_maps.append(ap)
        if details.get("n_gt", 0) > 0:
            all_recalls.append(details.get("recall", 0.0))
    
    # Averages
    results["avg_mAP"] = np.mean(all_maps) if all_maps else float("nan")
    results["avg_Recall"] = np.mean(all_recalls) if all_recalls else float("nan")
    
    return results


def format_binary_report(metrics: Dict[str, Any]) -> str:
    """Format binary TAL metrics as human-readable report."""
    lines = ["=" * 60, "Binary TAL Evaluation Results (RMM vs Background)", "=" * 60, ""]
    
    # Summary
    lines.append("Summary:")
    tiou_thresholds = [0.3, 0.5, 0.7]
    
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
    
    # Detection metrics at tIoU=0.5
    details = metrics.get("details", {}).get("tIoU=0.5", {})
    n_gt = details.get("n_gt", 0)
    n_pred = details.get("n_pred", 0)
    n_tp = details.get("n_tp", 0)
    n_fp = details.get("n_fp", 0)
    precision = details.get("precision", 0.0)
    recall = details.get("recall", 0.0)
    
    lines.append("Detection Metrics (tIoU=0.5):")
    lines.append(f"  n_GT:       {n_gt:>6}")
    lines.append(f"  n_Pred:     {n_pred:>6}")
    lines.append(f"  n_TP:       {n_tp:>6}")
    lines.append(f"  n_FP:       {n_fp:>6}")
    lines.append(f"  Precision:  {precision:>6.3f}")
    lines.append(f"  Recall:     {recall:>6.3f}")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def save_binary_metrics(
    metrics: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Save binary TAL metrics to files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
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
    
    json_path = out_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(convert_for_json(metrics), f, indent=2)
    
    # Save report
    report = format_binary_report(metrics)
    report_path = out_dir / "report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)
    
    # Save per-threshold metrics CSV
    rows = []
    for tiou_thr in [0.3, 0.5, 0.7]:
        thr_key = f"tIoU={tiou_thr}"
        d = metrics.get("details", {}).get(thr_key, {})
        rows.append({
            "tiou_threshold": thr_key,
            "mAP": metrics.get(f"mAP@{tiou_thr}"),
            "n_gt": d.get("n_gt", 0),
            "n_pred": d.get("n_pred", 0),
            "n_tp": d.get("n_tp", 0),
            "n_fp": d.get("n_fp", 0),
            "precision": d.get("precision", 0.0),
            "recall": d.get("recall", 0.0),
        })
    
    csv_path = out_dir / "per_tiou_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate binary (RMM vs BG) TAL models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--predictions", "-p",
        type=Path,
        required=True,
        help="Path to TAL-format predictions CSV.",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("/orcd/data/satra/001/users/brukew/actreg/dataprep/splits"),
        help="Root directory for GT segment CSVs.",
    )
    parser.add_argument(
        "--out-dir", "-o",
        type=Path,
        required=True,
        help="Output directory for evaluation results.",
    )
    
    # Postprocessing params
    parser.add_argument("--thr", type=float, default=0.5, help="Detection threshold.")
    parser.add_argument("--smooth-k", type=int, default=3, help="Smoothing window.")
    parser.add_argument("--merge-gap-sec", type=float, default=1.0, help="Merge gap.")
    
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output.")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 60)
        print("Binary TAL Evaluation")
        print("=" * 60)
        print(f"Predictions: {args.predictions}")
        print(f"Output: {args.out_dir}")
    
    # Load predictions
    if verbose:
        print("\nLoading predictions...")
    pred_df = pd.read_csv(args.predictions)
    if verbose:
        print(f"  Loaded {len(pred_df)} windows from {pred_df['video_key'].nunique()} videos")
    
    # Load GT
    if verbose:
        print("\nLoading GT segments (all RMM types merged)...")
    gt_by_video = load_binary_gt_segments(args.splits_root, mode="cv")
    n_gt = sum(len(v) for v in gt_by_video.values())
    if verbose:
        print(f"  Loaded {n_gt} GT segments from {len(gt_by_video)} videos")
    
    # Evaluate
    pp_params = {
        "thr": args.thr,
        "smooth_k": args.smooth_k,
        "merge_gap_sec": args.merge_gap_sec,
    }
    
    if verbose:
        print(f"\nPostprocess params: {pp_params}")
        print("\nEvaluating...")
    
    metrics = evaluate_binary_tal(pred_df, gt_by_video, pp_params)
    
    # Save
    save_binary_metrics(metrics, args.out_dir)
    
    # Print report
    if verbose:
        print("\n" + format_binary_report(metrics))
        print(f"\nResults saved to: {args.out_dir}")
    
    return metrics


if __name__ == "__main__":
    main()

