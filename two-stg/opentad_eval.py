#!/usr/bin/env python3
"""
Helpers for evaluating two-stage TAL predictions with the native OpenTAD evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ACTREG_ROOT = Path(__file__).resolve().parent.parent
OPENTAD_ROOT = ACTREG_ROOT / "OpenTAD"
OPENTAD_ANN_DIR = OPENTAD_ROOT / "data" / "sails_rmm" / "annotations"
DEFAULT_WINDOW_SPLITS_DIR = ACTREG_ROOT / "dataprep" / "tal" / "splits_cv_4class"
DEFAULT_TIOU_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

if str(ACTREG_ROOT) not in sys.path:
    sys.path.insert(0, str(ACTREG_ROOT))

from tal.tal_map_eval import ID2LABEL_4CLASS, normalize_video_file


def sanitize_for_json(obj: Any) -> Any:
    """Convert numpy / NaN values to JSON-safe Python values."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def ensure_mmengine_registry_shim() -> None:
    """
    Provide the tiny subset of `mmengine.registry` that OpenTAD's evaluator imports.
    """
    try:
        import mmengine.registry  # type: ignore  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    registry_module = types.ModuleType("mmengine.registry")

    class Registry:
        def __init__(self, name: str):
            self.name = name
            self._modules: dict[str, type] = {}

        def register_module(self):
            def decorator(cls):
                self._modules[cls.__name__] = cls
                return cls

            return decorator

        def build(self, cfg):
            cfg = dict(cfg)
            module_type = cfg.pop("type")
            if module_type not in self._modules:
                raise KeyError(f"{module_type} is not registered in {self.name}")
            return self._modules[module_type](**cfg)

    registry_module.Registry = Registry

    mmengine_module = types.ModuleType("mmengine")
    mmengine_module.registry = registry_module

    sys.modules["mmengine"] = mmengine_module
    sys.modules["mmengine.registry"] = registry_module


def load_gt_to_opentad_video_mapping(
    window_splits_dir: Path,
    fold: int,
) -> dict[str, str]:
    """
    Map normalized `video_file` paths to OpenTAD video ids for a validation fold.
    """
    val_csv = window_splits_dir / f"fold_{fold}_val_windows.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Val windows CSV not found: {val_csv}")

    df = pd.read_csv(val_csv)
    df["filename_stem"] = df["filename"].apply(lambda x: Path(x).stem)
    df["opentad_video_key"] = df["child_id"] + "_" + df["filename_stem"]
    df["gt_video_key"] = df["video_file"].apply(normalize_video_file)

    dedup_df = df.groupby("gt_video_key").first().reset_index()
    return dict(zip(dedup_df["gt_video_key"], dedup_df["opentad_video_key"]))


def prediction_df_to_opentad_results(
    pred_df: pd.DataFrame,
    *,
    gt_to_opentad_video: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Convert two-stage predictions to OpenTAD `result_detection.json` format.

    Supported inputs:
    - `opentad_video_key` column already present
    - `video_key` column with normalized `video_file` paths plus `gt_to_opentad_video`
    """
    required_cols = {"start_sec", "end_sec", "score"}
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        raise ValueError(f"Prediction DataFrame missing columns: {sorted(missing_cols)}")

    if "opentad_video_key" in pred_df.columns:
        video_ids = pred_df["opentad_video_key"].astype(str)
    else:
        if "video_key" not in pred_df.columns:
            raise ValueError("Need either `opentad_video_key` or `video_key` column in predictions.")
        if gt_to_opentad_video is None:
            raise ValueError("`gt_to_opentad_video` mapping is required when predictions only have `video_key`.")
        video_ids = pred_df["video_key"].map(gt_to_opentad_video)
        missing_video_ids = pred_df.loc[video_ids.isna(), "video_key"].drop_duplicates().tolist()
        if missing_video_ids:
            preview = ", ".join(missing_video_ids[:5])
            raise ValueError(
                f"Could not map {len(missing_video_ids)} prediction video_key values to OpenTAD ids. "
                f"Examples: {preview}"
            )

    if "label" in pred_df.columns:
        labels = pred_df["label"].astype(str)
    elif "label_name" in pred_df.columns:
        labels = pred_df["label_name"].astype(str)
    elif "class_id" in pred_df.columns:
        labels = pred_df["class_id"].map(ID2LABEL_4CLASS)
        unknown = pred_df.loc[labels.isna(), "class_id"].drop_duplicates().tolist()
        if unknown:
            raise ValueError(f"Unknown class ids in predictions: {unknown}")
    else:
        raise ValueError("Need one of `label`, `label_name`, or `class_id` in predictions.")

    result_rows = pred_df.copy()
    result_rows["opentad_video_key"] = video_ids
    result_rows["label_name"] = labels

    results: dict[str, list[dict[str, Any]]] = {}
    for video_id, group in result_rows.groupby("opentad_video_key", sort=False):
        ordered_group = group.sort_values("score", ascending=False)
        results[str(video_id)] = [
            {
                "segment": [float(row.start_sec), float(row.end_sec)],
                "label": str(row.label_name),
                "score": float(row.score),
            }
            for row in ordered_group.itertuples(index=False)
        ]

    return {"results": results}


def normalize_opentad_metrics(
    raw_metrics: dict[str, Any],
    *,
    per_class_ap: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """
    Keep OpenTAD's metrics, but also expose `avg_mAP` for local consistency.
    """
    metrics = dict(raw_metrics)
    if "average_mAP" in metrics:
        metrics["avg_mAP"] = metrics["average_mAP"]
    if per_class_ap is not None:
        metrics["per_class_ap"] = per_class_ap
    return metrics


def evaluate_with_opentad(
    prediction_data: dict[str, Any] | str | Path,
    *,
    fold: int,
    tiou_thresholds: list[float] | None = None,
    threads: int = 4,
) -> dict[str, Any]:
    """
    Evaluate predictions with OpenTAD's native mAP evaluator.
    """
    if tiou_thresholds is None:
        tiou_thresholds = DEFAULT_TIOU_THRESHOLDS

    if str(OPENTAD_ROOT) not in sys.path:
        sys.path.insert(0, str(OPENTAD_ROOT))

    ensure_mmengine_registry_shim()

    from opentad.evaluations.mAP import mAP as OpenTADmAP

    annotation_path = OPENTAD_ANN_DIR / f"fold{fold}_anno.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"OpenTAD annotation not found: {annotation_path}")

    prediction_arg: dict[str, Any] | str
    if isinstance(prediction_data, Path):
        prediction_arg = str(prediction_data)
    else:
        prediction_arg = prediction_data

    evaluator = OpenTADmAP(
        ground_truth_filename=str(annotation_path),
        prediction_filename=prediction_arg,
        subset="validation",
        tiou_thresholds=tiou_thresholds,
        thread=threads,
    )
    raw_metrics = evaluator.evaluate()

    inv_activity_index = {idx: label for label, idx in evaluator.activity_index.items()}
    per_class_ap = {}
    for thr_idx, tiou in enumerate(tiou_thresholds):
        thr_key = f"tIoU={tiou}"
        per_class_ap[thr_key] = {
            inv_activity_index[class_idx]: float(evaluator.ap[thr_idx, class_idx])
            for class_idx in sorted(inv_activity_index)
        }

    return normalize_opentad_metrics(raw_metrics, per_class_ap=per_class_ap)


def save_opentad_results_and_metrics(
    pred_df: pd.DataFrame,
    *,
    fold: int,
    output_dir: Path,
    window_splits_dir: Path = DEFAULT_WINDOW_SPLITS_DIR,
    tiou_thresholds: list[float] | None = None,
    threads: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Convert predictions, save OpenTAD-format outputs, and evaluate them.
    """
    gt_to_opentad_video = None
    if "opentad_video_key" not in pred_df.columns:
        gt_to_opentad_video = load_gt_to_opentad_video_mapping(window_splits_dir, fold)

    result_json = prediction_df_to_opentad_results(
        pred_df,
        gt_to_opentad_video=gt_to_opentad_video,
    )
    metrics = evaluate_with_opentad(
        result_json,
        fold=fold,
        tiou_thresholds=tiou_thresholds,
        threads=threads,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result_detection.json"
    metrics_path = output_dir / "metrics.json"
    metrics_opentad_path = output_dir / "metrics_opentad.json"

    result_path.write_text(json.dumps(sanitize_for_json(result_json), indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(sanitize_for_json(metrics), indent=2), encoding="utf-8")
    metrics_opentad_path.write_text(json.dumps(sanitize_for_json(metrics), indent=2), encoding="utf-8")

    return result_json, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two-stage TAL predictions with OpenTAD.")
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--predictions-csv", type=Path, required=True, help="Path to predictions.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for result_detection.json and metrics.json")
    parser.add_argument(
        "--window-splits-dir",
        type=Path,
        default=DEFAULT_WINDOW_SPLITS_DIR,
        help="Directory with fold_*_val_windows.csv files.",
    )
    parser.add_argument(
        "--tiou-thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_TIOU_THRESHOLDS,
        help="OpenTAD tIoU thresholds.",
    )
    parser.add_argument("--threads", type=int, default=4, help="OpenTAD evaluator worker processes.")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.predictions_csv)
    _, metrics = save_opentad_results_and_metrics(
        pred_df,
        fold=args.fold,
        output_dir=args.output_dir,
        window_splits_dir=args.window_splits_dir,
        tiou_thresholds=args.tiou_thresholds,
        threads=args.threads,
    )

    print(json.dumps(sanitize_for_json(metrics), indent=2))


if __name__ == "__main__":
    main()
