#!/usr/bin/env python3
"""
Write per-fold metadata for two-stage TAL result folders.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACTREG_ROOT = Path(__file__).resolve().parent.parent
OPENTAD_EXPS = ACTREG_ROOT / "OpenTAD" / "exps" / "sails_rmm"
OPENTAD_ANN_DIR = ACTREG_ROOT / "OpenTAD" / "data" / "sails_rmm" / "annotations"
VJEPA_CKPT_ROOT = (
    ACTREG_ROOT
    / "v-jepa"
    / "runs"
    / "vjepa2_rmm_cv"
    / "f64_lr1e-5_bs1_acc8_ep20_crop_4cls"
)
WINDOW_SPLITS_DIR = ACTREG_ROOT / "dataprep" / "tal" / "splits_cv_4class"
SPLITS_ROOT = ACTREG_ROOT / "dataprep" / "splits"


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
        }

    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_info(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    fold = args.fold
    stage2_backend = getattr(args, "stage2_backend", "vjepa2")

    detection_json = (
        OPENTAD_EXPS
        / f"actionformer_vjepa_binary_fold{fold}"
        / "gpu1_id99"
        / "result_detection.json"
    )
    checkpoint_dir = VJEPA_CKPT_ROOT / f"fold_{fold}"
    annotation_json = OPENTAD_ANN_DIR / f"fold{fold}_anno.json"
    fusion_bundle_dir = getattr(args, "fusion_bundle_dir", None)
    if fusion_bundle_dir is None:
        fusion_bundle_dir = (
            ACTREG_ROOT
            / "two-stg"
            / "fusion_checkpoints"
            / "three_way"
            / f"fold_{fold}"
        )
    fusion_bundle_dir = Path(fusion_bundle_dir).resolve()

    pose_prop = getattr(args, "posec3d_proposal_scores", None)
    stgcn_prop = getattr(args, "stgcn_proposal_scores", None)

    if stage2_backend == "three_way":
        stage2_desc = (
            "Live V-JEPA2 4-class softmax vectors + 3-way MLP fusion with PoseC3D and STGCN++ "
            "score vectors from exported skeleton_index.csv (nearest training clip by half-open tIoU; "
            "uniform fallback if no overlap)."
        )
        stage2_model = "V-JEPA2 4-class + ThreeWayMLPFusion (PoseC3D/STGCN++ via skeleton index)"
        stage2_extra: dict[str, Any] = {
            "classifier_backend": "three_way",
            "vjepa_checkpoint_dir": str(checkpoint_dir),
            "fusion_bundle_dir": str(fusion_bundle_dir),
            "fusion_artifacts": {
                "mlp_state": str(fusion_bundle_dir / "mlp_state.pt"),
                "config": str(fusion_bundle_dir / "config.json"),
                "skeleton_index": str(fusion_bundle_dir / "skeleton_index.csv"),
            },
        }
        if pose_prop and stgcn_prop:
            pp = Path(pose_prop).resolve()
            sp = Path(stgcn_prop).resolve()
            stage2_desc += (
                " Optional per-proposal PoseC3D/STGCN++ scores from live inference CSVs "
                "(see two-stg/build_proposal_poses.py); missing proposal_ids fall back to skeleton index."
            )
            stage2_model += " + live proposal skeleton CSVs"
            stage2_extra["proposal_posec3d_scores_csv"] = str(pp)
            stage2_extra["proposal_stgcn_scores_csv"] = str(sp)
            stage2_extra["fusion_artifacts"]["posec3d_proposal_scores"] = file_info(pp)
            stage2_extra["fusion_artifacts"]["stgcn_proposal_scores"] = file_info(sp)
    else:
        stage2_desc = "V-JEPA2 4-class (or 5-class) clip classification."
        stage2_model = "V-JEPA2 classifier (vjepa2 backend)"
        stage2_extra = {
            "classifier_backend": "vjepa2",
            "checkpoint_dir": str(checkpoint_dir),
        }

    info = {
        "generated_at_utc": iso_utc_now(),
        "fold": fold,
        "output_dir": str(output_dir),
        "pipeline": {
            "name": "two-stage-tal",
            "description": (
                "Binary ActionFormer proposal generation followed by stage-2 classification "
                f"({stage2_backend})."
            ),
            "stage1": {
                "model_name": "ActionFormer + V-JEPA Binary",
                "task": "binary TAL proposal generation (rmm vs background)",
                "detection_json": str(detection_json),
            },
            "stage2": {
                "model_name": stage2_model,
                "task": (
                    "clip classification into hands flapping / jumping / rocking / spinning; "
                    + stage2_desc
                ),
                "checkpoint_dir": str(checkpoint_dir),
                "frames_per_clip": 64,
                "score_combination": (
                    "final_score = detection_score * max(stage2_class_probs) "
                    "(stage2 is fused softmax for three_way; V-JEPA-only softmax for vjepa2)."
                ),
                **stage2_extra,
            },
            "evaluation": {
                "primary_metrics_file": "metrics.json",
                "primary_evaluator": "OpenTAD mAP",
                "primary_tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
                "custom_crosscheck_metrics_file": "metrics_custom.json",
                "custom_crosscheck_tiou_thresholds": [0.3, 0.5, 0.7],
                "annotation_json": str(annotation_json),
            },
        },
        "environments": {
            "classification_env": "vjepa2",
            "evaluation_env": "opentad",
            "stage2_backend": stage2_backend,
        },
        "invocation": {
            "window_splits_dir": str(WINDOW_SPLITS_DIR),
            "splits_root": str(SPLITS_ROOT),
            "source_job_ids": args.source_job_ids,
            "postprocess_job_id": args.postprocess_job_id,
            "postprocess_dependency": args.postprocess_dependency,
            "hostname": os.uname().nodename,
        },
        "artifacts": {
            "predictions_csv": file_info(output_dir / "predictions.csv"),
            "result_detection_json": file_info(output_dir / "result_detection.json"),
            "metrics_json": file_info(output_dir / "metrics.json"),
            "metrics_opentad_json": file_info(output_dir / "metrics_opentad.json"),
            "metrics_custom_json": file_info(output_dir / "metrics_custom.json"),
        },
    }

    metrics = load_json_if_exists(output_dir / "metrics.json")
    if metrics is not None:
        info["metrics_summary"] = {
            key: metrics.get(key)
            for key in ("average_mAP", "avg_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7")
            if key in metrics
        }

    custom_metrics = load_json_if_exists(output_dir / "metrics_custom.json")
    if custom_metrics is not None:
        info["custom_metrics_summary"] = {
            key: custom_metrics.get(key)
            for key in ("avg_mAP", "mAP@0.3", "mAP@0.5", "mAP@0.7")
            if key in custom_metrics
        }

    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Write two-stage TAL info.json metadata.")
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage2-backend",
        type=str,
        choices=["vjepa2", "three_way"],
        default="vjepa2",
        help="Must match how eval_two_stage_tal.py was run.",
    )
    parser.add_argument(
        "--fusion-bundle-dir",
        type=Path,
        default=None,
        help="For three_way: bundle with mlp_state.pt (default: two-stg/fusion_checkpoints/three_way/fold_N).",
    )
    parser.add_argument(
        "--posec3d-proposal-scores",
        type=Path,
        default=None,
        help="Optional: PoseC3D predictions_clip.csv for ActionFormer proposals.",
    )
    parser.add_argument(
        "--stgcn-proposal-scores",
        type=Path,
        default=None,
        help="Optional: fused STGCN++ predictions_clip.csv for proposals (same segment_ids).",
    )
    parser.add_argument("--source-job-ids", type=str, default=None, help="Optional upstream classification job ids.")
    parser.add_argument("--postprocess-job-id", type=str, default=None, help="Optional OpenTAD postprocess job id.")
    parser.add_argument("--postprocess-dependency", type=str, default=None, help="Optional dependency string.")
    args = parser.parse_args()

    info = build_info(args)
    output_path = args.output_dir / "info.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(output_path)


if __name__ == "__main__":
    main()
