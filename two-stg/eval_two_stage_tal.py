#!/usr/bin/env python3
"""
Two-stage TAL evaluation: Binary ActionFormer detection + Stage-2 classification.

Stage 1: Load precomputed ActionFormer binary detections (result_detection.json).

Stage 2 backends (--classifier-backend):
  - vjepa2: Hugging Face V-JEPA2 on RGB clips (4- or 5-class).
  - three_way: Live V-JEPA2 (4-class) + 3-way MLP fusion with PoseC3D/STGCN++ scores.
    Default: skeleton_index.csv (nearest clip by tIoU). Optional:
    --posec3d-proposal-scores + --stgcn-proposal-scores (live inference CSVs; see
    two-stg/build_proposal_poses.py and run_proposal_skeleton_inference.sh).

Final score: det_score * max(class_probs after stage-2). Evaluate mAP on val fold.

5-class is only supported for the vjepa2 backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor
from opentad_eval import save_opentad_results_and_metrics, sanitize_for_json
from stage2_three_way import ThreeWayStage2, default_bundle_dir, format_proposal_id

# Actreg root for tal_map_eval import
ACTREG_ROOT = Path(__file__).resolve().parent.parent
if str(ACTREG_ROOT) not in sys.path:
    sys.path.insert(0, str(ACTREG_ROOT))

from tal.tal_map_eval import (
    ID2LABEL_4CLASS,
    LABEL_MAP_4CLASS,
    GTSegment,
    compute_map,
    normalize_video_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Default paths (relative to actreg root)
DEFAULT_WINDOW_SPLITS_DIR = ACTREG_ROOT / "dataprep/tal/splits_cv_4class"
DEFAULT_SPLITS_ROOT = ACTREG_ROOT / "dataprep/splits"


def load_video_mapping(
    window_splits_dir: Path,
    fold: int,
) -> tuple[dict[str, Path], dict[str, str]]:
    """
    Load mapping from ActionFormer video_key to full video path and to GT video_key.

    Uses the val window CSVs which contain the actual ``video_path`` column pointing
    to the real video files on disk.

    Returns:
        af_key_to_path: ActionFormer key (child_id_filename_stem) -> full Path
        af_key_to_gt_key: ActionFormer key -> GT video_key (normalized path for compute_map)
    """
    val_csv = window_splits_dir / f"fold_{fold}_val_windows.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Val windows CSV not found: {val_csv}")

    df = pd.read_csv(val_csv)
    df["filename_stem"] = df["filename"].apply(lambda x: Path(x).stem)
    df["af_key"] = df["child_id"] + "_" + df["filename_stem"]

    # Deduplicate: one row per video
    video_df = df.groupby("af_key").first().reset_index()

    af_key_to_path = {}
    af_key_to_gt_key = {}
    for _, row in video_df.iterrows():
        af_key = row["af_key"]
        video_file = row["video_file"].replace("\\", "/").strip()
        video_path = row.get("video_path", "")
        if pd.isna(video_path) or not video_path:
            logger.warning("No video_path for %s, skipping", af_key)
            continue
        af_key_to_path[af_key] = Path(str(video_path).strip())
        af_key_to_gt_key[af_key] = normalize_video_file(video_file)

    logger.info("Loaded mapping: %d val videos for fold %d", len(af_key_to_path), fold)
    return af_key_to_path, af_key_to_gt_key


def load_gt_segments_val_only(splits_root: Path, fold: int) -> dict[str, list]:
    """Load ground-truth segments from fold_X_val.csv only. Returns gt_by_video with GT video_key."""
    val_csv = splits_root / "cv_splits_4class" / f"fold_{fold}_val.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Val CSV not found: {val_csv}")

    label_map = LABEL_MAP_4CLASS
    segments_by_video = {}
    with open(val_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_file = row.get("video_file", "")
            video_key = normalize_video_file(video_file)
            label_name = row.get("rmm_type", "")
            if label_name not in label_map:
                continue
            try:
                start_sec = float(row.get("start_sec", 0))
                end_sec = float(row.get("end_sec", 0))
            except ValueError:
                continue
            start_exclusive = start_sec
            end_exclusive = end_sec + 1.0
            gt = GTSegment(
                segment_id=row.get("segment_id", ""),
                video_key=video_key,
                label_idx=label_map[label_name],
                label_name=label_name,
                start_sec=start_exclusive,
                end_sec=end_exclusive,
            )
            segments_by_video.setdefault(video_key, []).append(gt)

    logger.info("Loaded GT: %d val segments in %d videos (fold %d)",
                sum(len(v) for v in segments_by_video.values()), len(segments_by_video), fold)
    return segments_by_video


def _extract_frames_from_reader(
    vr: VideoReader,
    fps: float,
    total_frames: int,
    start_sec: float,
    end_sec: float,
    n_frames: int = 64,
) -> np.ndarray | None:
    """Extract n_frames from an already-open VideoReader. Returns (n_frames, H, W, C) or None."""
    try:
        start_frame = max(0, int(start_sec * fps))
        end_frame = min(total_frames, int(end_sec * fps))
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        frame_indices = np.linspace(start_frame, end_frame - 1, n_frames).astype(np.int64)
        frame_indices = np.clip(frame_indices, 0, total_frames - 1)
        frames = vr.get_batch(frame_indices).asnumpy()
        return frames
    except Exception as e:
        logger.warning("Failed to extract [%.1f-%.1f]: %s", start_sec, end_sec, e)
        return None


def _classify_batch(
    batch_frames: list[np.ndarray],
    batch_det_scores: list[float],
    batch_segments: list[tuple[float, float]],
    gt_key: str,
    opentad_video_key: str,
    model,
    processor,
    device: str,
    num_classes: int = 4,
) -> list[dict]:
    """Classify a batch of frame arrays and return prediction rows.
    
    For 5-class models (4 RMM + background), we compute scores using only RMM
    class probabilities (indices 0-3), ignoring background (index 4). This
    naturally downweights proposals where the classifier thinks it's background.
    """
    inputs = processor(batch_frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    
    # For 5-class (4 RMM + BG), only consider RMM classes (0-3) for scoring
    if num_classes == 5:
        rmm_probs = probs[:, :4]  # Indices 0-3 are RMM classes
        pred_classes = rmm_probs.argmax(-1)
        # Use max RMM prob as confidence (ignores background)
        cls_confs = rmm_probs.max(-1).values
    else:
        pred_classes = probs.argmax(-1)
        cls_confs = probs.max(-1).values

    rows = []
    for i in range(len(batch_frames)):
        cls_id = pred_classes[i].item()
        cls_conf = cls_confs[i].item()
        rows.append({
            "video_key": gt_key,
            "opentad_video_key": opentad_video_key,
            "class_id": cls_id,
            "label": ID2LABEL_4CLASS[cls_id],
            "start_sec": batch_segments[i][0],
            "end_sec": batch_segments[i][1],
            "score": batch_det_scores[i] * cls_conf,
            # Store raw background prob for analysis (5-class only)
            "bg_prob": probs[i, 4].item() if num_classes == 5 else None,
        })
    return rows


def _classify_batch_three_way(
    batch_frames: list[np.ndarray],
    batch_det_scores: list[float],
    batch_segments: list[tuple[float, float]],
    batch_proposal_ids: list[str],
    gt_key: str,
    opentad_video_key: str,
    vjepa_model,
    processor,
    three_way: ThreeWayStage2,
    device: str,
) -> list[dict]:
    """4-class only: live V-JEPA2 probs + 3-way MLP (proposal CSVs and/or skeleton_index)."""
    inputs = processor(batch_frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = vjepa_model(**inputs).logits
    v_probs = F.softmax(logits, dim=-1)
    if v_probs.shape[-1] != 4:
        raise ValueError(
            "three_way backend requires a 4-class V-JEPA2 checkpoint "
            f"(got {v_probs.shape[-1]} classes)."
        )
    video_keys = [gt_key] * len(batch_frames)
    fused_probs, fused_conf = three_way.fuse_batch(
        v_probs.cpu(), video_keys, batch_segments, proposal_ids=batch_proposal_ids
    )
    fused_probs = fused_probs.cpu()
    fused_conf = fused_conf.cpu()
    pred_classes = fused_probs.argmax(dim=-1)
    rows = []
    for i in range(len(batch_frames)):
        cls_id = pred_classes[i].item()
        cls_conf = fused_conf[i].item()
        rows.append({
            "video_key": gt_key,
            "opentad_video_key": opentad_video_key,
            "class_id": cls_id,
            "label": ID2LABEL_4CLASS[cls_id],
            "start_sec": batch_segments[i][0],
            "end_sec": batch_segments[i][1],
            "score": batch_det_scores[i] * cls_conf,
            "bg_prob": None,
        })
    return rows


def run_two_stage_eval(
    detection_json: Path,
    checkpoint_dir: Path,
    af_key_to_path: dict[str, Path],
    af_key_to_gt_key: dict[str, str],
    gt_by_video: dict[str, list],
    *,
    min_det_score: float = 0.0,
    n_frames: int = 64,
    batch_size: int = 8,
    device: str = "cuda",
    num_classes: int | None = None,
    classifier_backend: str = "vjepa2",
    fusion_bundle_dir: Path | None = None,
    pose_proposal_scores_csv: Path | None = None,
    stgcn_proposal_scores_csv: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run Stage 2 on detections and compute mAP. Returns (pred_df, metrics)."""
    logger.info("Loading detections from %s", detection_json)
    with open(detection_json, "r") as f:
        data = json.load(f)
    results = data.get("results", {})
    logger.info("Loaded detections for %d videos", len(results))

    logger.info("Classifier backend: %s", classifier_backend)
    logger.info("Loading V-JEPA2 encoder/classifier from %s", checkpoint_dir)
    model = VJEPA2ForVideoClassification.from_pretrained(str(checkpoint_dir))
    processor = VJEPA2VideoProcessor.from_pretrained(str(checkpoint_dir))
    model.to(device)
    model.eval()

    three_way: ThreeWayStage2 | None = None
    if classifier_backend == "three_way":
        bundle = fusion_bundle_dir
        if bundle is None:
            raise ValueError("three_way backend requires fusion_bundle_dir")
        kw: dict = {"device": device, "num_classes": 4}
        if pose_proposal_scores_csv is not None and stgcn_proposal_scores_csv is not None:
            kw["pose_proposal_scores_csv"] = pose_proposal_scores_csv
            kw["stgcn_proposal_scores_csv"] = stgcn_proposal_scores_csv
            logger.info("Using proposal skeleton CSVs: %s , %s", pose_proposal_scores_csv, stgcn_proposal_scores_csv)
        elif pose_proposal_scores_csv is not None or stgcn_proposal_scores_csv is not None:
            raise ValueError(
                "three_way: provide both --posec3d-proposal-scores and --stgcn-proposal-scores, or neither."
            )
        three_way = ThreeWayStage2(bundle, **kw)
        num_classes = 4
        logger.info("3-way fusion bundle: %s", bundle)
    else:
        # Auto-detect num_classes from model config if not specified
        if num_classes is None:
            num_classes = len(model.config.id2label)
            logger.info("Auto-detected %d classes from model config", num_classes)
        else:
            logger.info("Using %d classes (user-specified)", num_classes)

        if num_classes == 5:
            logger.info("5-class mode: scores computed from RMM probs only (bg ignored)")

    logger.info("V-JEPA2 loaded on %s", device)

    rows = []
    total_proposals = 0
    skipped_video_fail = 0
    skipped_extract_fail = 0

    val_videos = [k for k in results if k in af_key_to_path and k in af_key_to_gt_key]
    val_proposal_count = sum(len(results[k]) for k in val_videos)
    logger.info("Will classify %d proposals across %d val videos", val_proposal_count, len(val_videos))

    classified = 0
    for vid_idx, af_key in enumerate(val_videos):
        video_path = af_key_to_path[af_key]
        gt_key = af_key_to_gt_key[af_key]
        proposals = results[af_key]

        if not video_path.exists():
            logger.warning("Video not found: %s", video_path)
            skipped_video_fail += len(proposals)
            classified += len(proposals)
            continue

        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            fps = vr.get_avg_fps()
            if fps is None or fps <= 0:
                fps = 30.0
            total_frames = len(vr)
        except Exception as e:
            logger.warning("Failed to open %s: %s", video_path, e)
            skipped_video_fail += len(proposals)
            classified += len(proposals)
            continue

        batch_frames = []
        batch_det_scores = []
        batch_segments = []
        batch_proposal_ids: list[str] = []

        for prop in proposals:
            det_score = float(prop["score"])
            if det_score < min_det_score:
                continue
            total_proposals += 1
            start_sec, end_sec = prop["segment"]

            frames = _extract_frames_from_reader(vr, fps, total_frames, start_sec, end_sec, n_frames)
            if frames is None:
                skipped_extract_fail += 1
                classified += 1
                continue

            batch_frames.append(frames)
            batch_det_scores.append(det_score)
            batch_segments.append((float(start_sec), float(end_sec)))
            batch_proposal_ids.append(
                format_proposal_id(af_key, float(start_sec), float(end_sec))
            )

            if len(batch_frames) >= batch_size:
                if three_way is not None:
                    rows.extend(_classify_batch_three_way(
                        batch_frames, batch_det_scores, batch_segments,
                        batch_proposal_ids,
                        gt_key, af_key, model, processor, three_way, device,
                    ))
                else:
                    rows.extend(_classify_batch(
                        batch_frames, batch_det_scores, batch_segments,
                        gt_key, af_key, model, processor, device,
                        num_classes=num_classes,
                    ))
                classified += len(batch_frames)
                batch_frames, batch_det_scores, batch_segments = [], [], []
                batch_proposal_ids = []

                if classified % 100 < batch_size:
                    logger.info("Progress: %d / %d proposals (%.0f%%)", classified, val_proposal_count,
                                100.0 * classified / val_proposal_count)

        if batch_frames:
            if three_way is not None:
                rows.extend(_classify_batch_three_way(
                    batch_frames, batch_det_scores, batch_segments,
                    batch_proposal_ids,
                    gt_key, af_key, model, processor, three_way, device,
                ))
            else:
                rows.extend(_classify_batch(
                    batch_frames, batch_det_scores, batch_segments,
                    gt_key, af_key, model, processor, device,
                    num_classes=num_classes,
                ))
            classified += len(batch_frames)
            batch_frames, batch_det_scores, batch_segments = [], [], []

        del vr

        if (vid_idx + 1) % 10 == 0:
            logger.info("Videos: %d / %d  |  Proposals: %d / %d (%.0f%%)",
                        vid_idx + 1, len(val_videos), classified, val_proposal_count,
                        100.0 * classified / val_proposal_count)

    logger.info("Classified %d segments (skipped video: %d, extract: %d)",
                len(rows), skipped_video_fail, skipped_extract_fail)

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        metrics = {
            "mAP@0.3": float("nan"),
            "mAP@0.5": float("nan"),
            "mAP@0.7": float("nan"),
            "avg_mAP": float("nan"),
        }
        return pred_df, metrics

    metrics = compute_map(
        pred_df,
        gt_by_video,
        class_ids=[0, 1, 2, 3],
        tiou_thresholds=[0.3, 0.5, 0.7],
    )
    return pred_df, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage TAL: Binary det + V-JEPA2 or 3-way fusion stage-2"
    )
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2], help="CV fold to evaluate")
    parser.add_argument("--detection-json", type=Path, required=True, help="ActionFormer result_detection.json")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="V-JEPA2 fold checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save predictions and metrics")
    parser.add_argument("--min-det-score", type=float, default=0.0, help="Min detection score to classify")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for classifier (currently 1 per segment)")
    parser.add_argument("--n-frames", type=int, default=64, help="Frames per clip for V-JEPA2")
    parser.add_argument("--window-splits-dir", type=Path, default=DEFAULT_WINDOW_SPLITS_DIR,
                        help="Directory with fold_*_val_windows.csv (has video_path column)")
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for V-JEPA2 (cuda or cpu). Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--skip-opentad-eval",
        action="store_true",
        help="Only save predictions.csv and metrics_custom.json. Skip native OpenTAD evaluation.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        choices=[4, 5],
        help="Number of classifier classes (4 or 5). Auto-detected from model if not specified. "
             "For 5-class (4 RMM + BG), scores are computed using RMM probs only. "
             "Ignored when --classifier-backend three_way (always 4-class).",
    )
    parser.add_argument(
        "--classifier-backend",
        type=str,
        choices=["vjepa2", "three_way"],
        default="vjepa2",
        help="vjepa2: V-JEPA2 only. three_way: V-JEPA2 + 3-way MLP (needs --fusion-bundle-dir).",
    )
    parser.add_argument(
        "--fusion-bundle-dir",
        type=Path,
        default=None,
        help="Directory with mlp_state.pt, config.json, skeleton_index.csv "
             "(default: two-stg/fusion_checkpoints/three_way/fold_{fold}).",
    )
    parser.add_argument(
        "--posec3d-proposal-scores",
        type=Path,
        default=None,
        help="Optional predictions_clip.csv from PoseC3D on proposal pickle (segment_id = proposal_id).",
    )
    parser.add_argument(
        "--stgcn-proposal-scores",
        type=Path,
        default=None,
        help="Optional fused STGCN++ predictions_clip.csv for proposals (same segment_ids as PoseC3D).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using device: %s", args.device)

    af_key_to_path, af_key_to_gt_key = load_video_mapping(
        args.window_splits_dir, args.fold
    )
    gt_by_video = load_gt_segments_val_only(args.splits_root, args.fold)

    fusion_bundle = args.fusion_bundle_dir
    if args.classifier_backend == "three_way":
        fusion_bundle = fusion_bundle or default_bundle_dir(ACTREG_ROOT, args.fold)
        if not (fusion_bundle / "mlp_state.pt").exists():
            raise FileNotFoundError(
                f"Missing 3-way fusion bundle at {fusion_bundle}. "
                "Run: python fusion/export_three_way_deploy_checkpoint.py --all-folds"
            )
    pose_csv = args.posec3d_proposal_scores
    stgcn_csv = args.stgcn_proposal_scores
    if (pose_csv is None) ^ (stgcn_csv is None):
        parser.error("Provide both --posec3d-proposal-scores and --stgcn-proposal-scores, or neither.")
    if pose_csv is not None and args.classifier_backend != "three_way":
        parser.error("Proposal score CSVs require --classifier-backend three_way.")

    pred_df, metrics = run_two_stage_eval(
        args.detection_json,
        args.checkpoint_dir,
        af_key_to_path,
        af_key_to_gt_key,
        gt_by_video,
        min_det_score=args.min_det_score,
        n_frames=args.n_frames,
        batch_size=args.batch_size,
        device=args.device,
        num_classes=args.num_classes,
        classifier_backend=args.classifier_backend,
        fusion_bundle_dir=fusion_bundle,
        pose_proposal_scores_csv=pose_csv,
        stgcn_proposal_scores_csv=stgcn_csv,
    )

    pred_df.to_csv(args.output_dir / "predictions.csv", index=False)
    with open(args.output_dir / "metrics_custom.json", "w") as f:
        json.dump(sanitize_for_json(metrics), f, indent=2)

    logger.info("Fold %d custom avg_mAP@{0.3,0.5,0.7}: %.4f", args.fold, metrics.get("avg_mAP", float("nan")))
    if args.skip_opentad_eval:
        logger.info("Skipped OpenTAD evaluation; wrapper script should run two-stg/opentad_eval.py next.")
    else:
        _, opentad_metrics = save_opentad_results_and_metrics(
            pred_df,
            fold=args.fold,
            output_dir=args.output_dir,
            window_splits_dir=args.window_splits_dir,
        )
        logger.info("Fold %d OpenTAD avg_mAP@{0.3:0.7}: %.4f", args.fold, opentad_metrics.get("avg_mAP", float("nan")))
    logger.info("Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
