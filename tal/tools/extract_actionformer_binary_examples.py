#!/usr/bin/env python3
"""
Generate ActionFormer binary TAL example clips (TP/FP/FN/TN) at segment level.

Uses ActionFormer binary segment predictions and ground-truth RMM segments to
select 4 examples for each category (2 near-miss + 2 confident). True negatives
are defined using background windows from TAL split CSVs (primary_label=-1).

Outputs short mp4 clips with web-friendly encoding (H.264 + AAC + faststart).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import shutil

# Resolve paths relative to the repo root so this runs from any clone location.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "insights/tal_binary_examples"
DEFAULT_SEGMENTS_CSV = _REPO_ROOT / "dataprep/rmm_segments.csv"
DEFAULT_TAL_SPLITS_DIR = _REPO_ROOT / "dataprep/tal/splits_cv_4class"
DEFAULT_ACTIONFORMER_EXPS = _REPO_ROOT / "OpenTAD/exps/sails_rmm"

FOLD_PRED_PATH = "actionformer_vjepa_binary_fold{fold}/gpu1_id99/result_detection.json"

IOU_THRESHOLD = 0.3
SCORE_THRESHOLD = 0.1

NEAR_MISS_RANGE = (0.40, 0.60)
CONFIDENT_MIN = 0.80

RMM_SHORT = {
    "hands flapping": "HF",
    "jumping": "J",
    "rocking": "R",
    "spinning": "S",
}


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class ExampleItem:
    """Selected example for export."""
    category: str  # TP/FP/FN/TN
    confidence: float
    fold: int
    video_key: str
    start_sec: float
    end_sec: float
    source_video: Path
    rmm_type: Optional[str] = None
    pred_score: Optional[float] = None
    bg_conf: Optional[float] = None
    iou: Optional[float] = None
    near_or_conf: Optional[str] = None  # near or confident


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def normalize_rmm_type(rmm_type: str) -> str:
    """Normalize RMM labels to the 4-class naming."""
    if rmm_type == "one hand flap":
        return "hands flapping"
    return rmm_type


def normalize_video_key(video_file: str) -> Optional[str]:
    """
    Create ActionFormer-style video_key: <child_id>_<filename_stem>.
    Mirrors actionformer_failure_analysis.py.
    """
    if not video_file:
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


def window_key_from_window_id(window_id: str) -> str:
    """Extract video_key-like prefix from window_id."""
    return window_id.rsplit("__t", 1)[0] if "__t" in window_id else window_id


def compute_segment_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Compute temporal IoU between predicted and ground truth segments."""
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 0.0
    return intersection / union


def load_actionformer_predictions(pred_path: Path) -> Dict[str, List[Dict]]:
    """Load ActionFormer detection results from JSON."""
    with pred_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", data)


def load_val_windows(split_path: Path) -> pd.DataFrame:
    """Load validation windows CSV for a fold."""
    df = pd.read_csv(split_path)
    df["window_key"] = df["window_id"].astype(str).apply(window_key_from_window_id)
    return df


def load_gt_segments(segments_csv: Path) -> pd.DataFrame:
    """Load and normalize ground-truth segments."""
    df = pd.read_csv(segments_csv)
    df["rmm_type_norm"] = df["rmm_type"].apply(normalize_rmm_type)
    df["video_key"] = df["video_file"].apply(normalize_video_key)
    return df


def match_predictions_to_gt(
    predictions: List[Dict],
    gt_segments: List[Dict],
    iou_threshold: float = IOU_THRESHOLD,
) -> Dict[str, List]:
    """
    Match predictions to GT segments (binary: all GT are RMM).

    Returns dict with true_positives, false_positives, false_negatives.
    """
    matched_gt = set()
    true_positives = []
    false_positives = []

    sorted_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)
    for pred in sorted_preds:
        pred_start, pred_end = pred["segment"]
        best_iou = 0.0
        best_gt_idx = None

        for i, gt in enumerate(gt_segments):
            if i in matched_gt:
                continue
            iou = compute_segment_iou(pred_start, pred_end, gt["start_sec"], gt["end_sec"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_gt_idx is not None:
            matched_gt.add(best_gt_idx)
            true_positives.append({
                "pred": pred,
                "gt": gt_segments[best_gt_idx],
                "iou": best_iou,
            })
        else:
            false_positives.append(pred)

    false_negatives = [gt for i, gt in enumerate(gt_segments) if i not in matched_gt]
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def max_pred_score_overlap(
    predictions: List[Dict],
    start_sec: float,
    end_sec: float,
) -> float:
    """
    Return max RMM score for any prediction overlapping [start_sec, end_sec].
    Overlap is defined by any intersection > 0.
    """
    max_score = 0.0
    for pred in predictions:
        ps, pe = pred["segment"]
        if min(pe, end_sec) > max(ps, start_sec):
            max_score = max(max_score, pred["score"])
    return max_score


def select_near_and_confident(items: List[ExampleItem], total: int = 4) -> List[ExampleItem]:
    """
    Select near-miss and confident items from a category.

    Args:
        items: Candidate items.
        total: Total number to return. Defaults to 4 (2 near + 2 confident).
    """
    if not items:
        return []

    near_min, near_max = NEAR_MISS_RANGE
    near = [i for i in items if near_min <= i.confidence <= near_max]
    confident = [i for i in items if i.confidence >= CONFIDENT_MIN]

    # Prefer highest confidence for "confident"
    target_conf = max(1, total // 2)
    target_near = max(1, total - target_conf)
    confident = sorted(confident, key=lambda x: x.confidence, reverse=True)[:target_conf]
    for item in confident:
        item.near_or_conf = "confident"

    # Prefer closest to 0.5 for "near miss"
    near = sorted(near, key=lambda x: abs(x.confidence - 0.5))[:target_near]
    for item in near:
        item.near_or_conf = "near"

    chosen = {id(x) for x in confident + near}
    remaining = [i for i in items if id(i) not in chosen]

    # Fill near misses if we didn't get enough
    while len(near) < target_near and remaining:
        candidate = min(remaining, key=lambda x: abs(x.confidence - 0.5))
        candidate.near_or_conf = "near"
        near.append(candidate)
        remaining.remove(candidate)

    # Fill confident if we didn't get enough
    while len(confident) < target_conf and remaining:
        candidate = max(remaining, key=lambda x: x.confidence)
        candidate.near_or_conf = "confident"
        confident.append(candidate)
        remaining.remove(candidate)

    return (near + confident)[:total]


def ensure_valid_range(start_sec: float, end_sec: float) -> Tuple[float, float]:
    """Ensure end_sec > start_sec, fixing zero-length segments."""
    if end_sec <= start_sec:
        return start_sec, start_sec + 1.0
    return start_sec, end_sec


def format_name(item: ExampleItem) -> str:
    """Build output filename with required fields plus a uniqueness suffix."""
    score_int = int(round(item.confidence * 100))
    start_ms = int(round(item.start_sec * 1000))
    end_ms = int(round(item.end_sec * 1000))
    suffix = f"{item.video_key}_t{start_ms}_{end_ms}"
    if item.rmm_type:
        short = RMM_SHORT.get(item.rmm_type, item.rmm_type[:2].upper())
        return f"{item.category}_{score_int}_{short}_{suffix}.mp4"
    return f"{item.category}_{score_int}_{suffix}.mp4"


def run_ffmpeg(source: Path, start_sec: float, end_sec: float, output_path: Path) -> None:
    """Cut a clip with web-friendly encoding."""
    ffmpeg_bin = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise FileNotFoundError(
            "ffmpeg not found in PATH. Load the ffmpeg module or set FFMPEG_BIN."
        )
    encoders = _detect_encoders(ffmpeg_bin)
    duration = max(0.01, end_sec - start_sec)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-avoid_negative_ts",
        "make_zero",
    ]

    if encoders.get("libx264"):
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
    elif encoders.get("h264_v4l2m2m"):
        # Hardware H.264 encoder (if available). Requires bitrate-based settings.
        cmd += [
            "-c:v",
            "h264_v4l2m2m",
            "-b:v",
            "5M",
            "-maxrate",
            "5M",
            "-bufsize",
            "10M",
            "-pix_fmt",
            "nv12",
        ]
    else:
        cmd += [
            "-c:v",
            "mpeg4",
            "-q:v",
            "4",
            "-pix_fmt",
            "yuv420p",
        ]

    if encoders.get("aac"):
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]

    cmd += [
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


_ENCODER_CACHE: Dict[str, Dict[str, bool]] = {}


def _detect_encoders(ffmpeg_bin: str) -> Dict[str, bool]:
    """Detect available encoders once per ffmpeg binary."""
    if ffmpeg_bin in _ENCODER_CACHE:
        return _ENCODER_CACHE[ffmpeg_bin]
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    encoders = {
        "libx264": "libx264" in output,
        "h264_v4l2m2m": "h264_v4l2m2m" in output,
        "aac": "aac" in output,
    }
    _ENCODER_CACHE[ffmpeg_bin] = encoders
    return encoders


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ActionFormer binary TAL examples (TP/FP/FN/TN) as clips."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--segments-csv", type=Path, default=DEFAULT_SEGMENTS_CSV)
    parser.add_argument("--tal-splits-dir", type=Path, default=DEFAULT_TAL_SPLITS_DIR)
    parser.add_argument("--actionformer-exps", type=Path, default=DEFAULT_ACTIONFORMER_EXPS)
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    parser.add_argument(
        "--extra-tp-classes",
        nargs="*",
        default=[],
        help="RMM classes to add extra TP examples for (e.g., jumping rocking spinning).",
    )
    parser.add_argument(
        "--extra-tp-per-class",
        type=int,
        default=0,
        help="Additional TP examples to add per class in --extra-tp-classes.",
    )
    parser.add_argument(
        "--extra-fn-class",
        type=str,
        default=None,
        help="Add one extra FN example for this RMM class (e.g., rocking).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing manifest/outputs instead of replacing.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_manifest = output_dir / "manifest.csv"
    existing_outputs = set()
    if args.append and existing_manifest.exists():
        existing_df = pd.read_csv(existing_manifest)
        existing_outputs = set(existing_df["output_path"].astype(str).tolist())

    gt_df = load_gt_segments(args.segments_csv)

    # Per-fold processing
    tp_items: List[ExampleItem] = []
    fp_items: List[ExampleItem] = []
    fn_items: List[ExampleItem] = []
    tn_items: List[ExampleItem] = []

    for fold in range(3):
        split_path = args.tal_splits_dir / f"fold_{fold}_val_windows.csv"
        if not split_path.exists():
            continue

        val_windows = load_val_windows(split_path)
        fold_video_keys = sorted(val_windows["window_key"].unique().tolist())

        # Map video_key -> video_path
        video_path_map = (
            val_windows[["window_key", "video_path"]]
            .drop_duplicates("window_key")
            .set_index("window_key")["video_path"]
            .to_dict()
        )

        # Load fold predictions
        pred_path = args.actionformer_exps / FOLD_PRED_PATH.format(fold=fold)
        if not pred_path.exists():
            continue
        pred_by_video = load_actionformer_predictions(pred_path)

        # Filter predictions to fold validation videos only
        pred_by_video = {k: v for k, v in pred_by_video.items() if k in fold_video_keys}

        # Prepare GT for this fold
        gt_fold = gt_df[gt_df["video_key"].isin(fold_video_keys)].copy()

        # Build per-video GT lookup
        gt_by_video: Dict[str, List[Dict]] = {}
        for _, row in gt_fold.iterrows():
            start, end = ensure_valid_range(float(row["start_sec"]), float(row["end_sec"]))
            gt_by_video.setdefault(row["video_key"], []).append({
                "start_sec": start,
                "end_sec": end,
                "rmm_type_norm": row["rmm_type_norm"],
            })

        # TP/FP/FN from segment-level matching
        for video_key, preds in pred_by_video.items():
            filtered_preds = [p for p in preds if p["score"] >= args.score_threshold]
            gt_segments = gt_by_video.get(video_key, [])

            match_result = match_predictions_to_gt(
                filtered_preds, gt_segments, iou_threshold=args.iou_threshold
            )

            src_video = Path(video_path_map.get(video_key, ""))
            if not src_video.exists():
                continue

            for tp in match_result["true_positives"]:
                pred = tp["pred"]
                gt = tp["gt"]
                tp_items.append(ExampleItem(
                    category="TP",
                    confidence=float(pred["score"]),
                    fold=fold,
                    video_key=video_key,
                    start_sec=float(pred["segment"][0]),
                    end_sec=float(pred["segment"][1]),
                    source_video=src_video,
                    rmm_type=gt["rmm_type_norm"],
                    pred_score=float(pred["score"]),
                    iou=float(tp["iou"]),
                ))

            for pred in match_result["false_positives"]:
                fp_items.append(ExampleItem(
                    category="FP",
                    confidence=float(pred["score"]),
                    fold=fold,
                    video_key=video_key,
                    start_sec=float(pred["segment"][0]),
                    end_sec=float(pred["segment"][1]),
                    source_video=src_video,
                    pred_score=float(pred["score"]),
                ))

            for gt in match_result["false_negatives"]:
                max_score = max_pred_score_overlap(filtered_preds, gt["start_sec"], gt["end_sec"])
                bg_conf = 1.0 - max_score
                fn_items.append(ExampleItem(
                    category="FN",
                    confidence=float(bg_conf),
                    fold=fold,
                    video_key=video_key,
                    start_sec=float(gt["start_sec"]),
                    end_sec=float(gt["end_sec"]),
                    source_video=src_video,
                    rmm_type=gt["rmm_type_norm"],
                    bg_conf=float(bg_conf),
                ))

        # TNs from background windows
        bg_windows = val_windows[val_windows["primary_label"] == -1]
        for _, row in bg_windows.iterrows():
            video_key = row["window_key"]
            src_video = Path(video_path_map.get(video_key, ""))
            if not src_video.exists():
                continue

            preds = pred_by_video.get(video_key, [])
            filtered_preds = [p for p in preds if p["score"] >= args.score_threshold]
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
            max_score = max_pred_score_overlap(filtered_preds, start_sec, end_sec)
            if max_score < args.score_threshold:
                bg_conf = 1.0 - max_score
                tn_items.append(ExampleItem(
                    category="TN",
                    confidence=float(bg_conf),
                    fold=fold,
                    video_key=video_key,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    source_video=src_video,
                    bg_conf=float(bg_conf),
                ))

    # Select examples
    selected = []
    for cat, items in [("TP", tp_items), ("FP", fp_items), ("FN", fn_items), ("TN", tn_items)]:
        selected.extend(select_near_and_confident(items))

    def _filter_new(items: List[ExampleItem]) -> List[ExampleItem]:
        """Filter out items that would duplicate existing outputs."""
        fresh = []
        for item in items:
            out_name = format_name(item)
            out_path = output_dir / out_name
            if str(out_path) not in existing_outputs:
                fresh.append(item)
        return fresh

    # Extra TP selections by class
    if args.extra_tp_classes and args.extra_tp_per_class > 0:
        for cls in args.extra_tp_classes:
            cls_lower = cls.lower()
            cls_items = [i for i in tp_items if (i.rmm_type or "").lower() == cls_lower]
            cls_items = _filter_new(cls_items)
            extra = select_near_and_confident(cls_items, total=args.extra_tp_per_class)
            for item in extra:
                item.near_or_conf = item.near_or_conf or "extra"
            selected.extend(extra)

    # Extra FN selection by class
    if args.extra_fn_class:
        cls_lower = args.extra_fn_class.lower()
        fn_cls_items = [i for i in fn_items if (i.rmm_type or "").lower() == cls_lower]
        fn_cls_items = _filter_new(fn_cls_items)
        extra_fn = select_near_and_confident(fn_cls_items, total=1)
        for item in extra_fn:
            item.near_or_conf = item.near_or_conf or "extra"
        selected.extend(extra_fn)

    # Write clips and manifest
    manifest_rows = []
    for item in selected:
        start_sec, end_sec = ensure_valid_range(item.start_sec, item.end_sec)
        item.start_sec, item.end_sec = start_sec, end_sec

        out_name = format_name(item)
        out_path = output_dir / out_name
        if str(out_path) in existing_outputs and args.append:
            continue

        run_ffmpeg(item.source_video, start_sec, end_sec, out_path)

        manifest_rows.append({
            "type": item.category,
            "near_or_conf": item.near_or_conf,
            "confidence": item.confidence,
            "fold": item.fold,
            "video_key": item.video_key,
            "start_sec": item.start_sec,
            "end_sec": item.end_sec,
            "source_video": str(item.source_video),
            "rmm_type": item.rmm_type or "",
            "pred_score": item.pred_score if item.pred_score is not None else "",
            "bg_conf": item.bg_conf if item.bg_conf is not None else "",
            "iou": item.iou if item.iou is not None else "",
            "output_path": str(out_path),
        })

    manifest_df = pd.DataFrame(manifest_rows)
    if args.append and existing_manifest.exists():
        combined = pd.concat([existing_df, manifest_df], ignore_index=True)
        combined.to_csv(existing_manifest, index=False)
    else:
        manifest_df.to_csv(output_dir / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
