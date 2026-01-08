#!/usr/bin/env python
"""
Summarize cross-validation results for STGCN++ and PoseC3D TAL/RMM experiments.

This script computes:
1. Per-modality CV summaries (mean ± std across folds)
2. 4-stream fusion summary for STGCN++ (aggregates all modalities)
3. Single-model summary for PoseC3D

Usage:
    # STGCN++ 4-stream summary
    python tools/summarize_cv_results.py \
        --work-dir work_dirs/stgcnpp/tal/cv_4class_5class_ce_balanced \
        --model stgcnpp \
        --task tal \
        --loss ce \
        --modalities j b jm bm

    # PoseC3D single-model summary
    python tools/summarize_cv_results.py \
        --work-dir work_dirs/posec3d/tal/cv_4class_5class_ce_balanced \
        --model posec3d \
        --task tal \
        --loss ce

    # Add metadata like class_prob or bg_subsample
    python tools/summarize_cv_results.py \
        --work-dir work_dirs/stgcnpp/tal/cv_4class_5class_ce_balanced \
        --model stgcnpp \
        --task tal \
        --loss ce \
        --metadata "class_prob=[1.0, 1.0, 1.93, 9.12, 0.1]" "patience=5"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


def load_fold_metrics(fold_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metrics.json from a fold's eval directory."""
    metrics_file = fold_dir / "eval_val" / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


def compute_cv_summary(
    base_dir: Path,
    modality: Optional[str] = None,
    num_folds: int = 3,
    metadata: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute CV summary (mean ± std) for a single model/modality.

    Args:
        base_dir: Directory containing fold0, fold1, fold2 subdirectories
        modality: Modality name (for STGCN++), None for PoseC3D
        num_folds: Number of CV folds
        metadata: Additional metadata to include in summary

    Returns:
        Summary dict with per-fold results and aggregated stats
    """
    folds = []

    for fold in range(num_folds):
        fold_dir = base_dir / f"fold{fold}"
        metrics = load_fold_metrics(fold_dir)

        if metrics:
            metrics["fold"] = fold
            folds.append(metrics)
            status = "loaded"
        else:
            status = "not found"

        if modality:
            print(f"  {modality} Fold {fold}: {status}")
        else:
            print(f"  Fold {fold}: {status}")

    if not folds:
        return None

    # Build summary
    summary: Dict[str, Any] = {"per_fold": folds}

    if modality:
        summary["modality"] = modality

    if metadata:
        summary.update(metadata)

    # Compute mean/std for numeric metrics
    metric_keys = [
        k for k in folds[0].keys() if k not in ("fold", "split", "task", "checkpoint")
    ]

    for key in metric_keys:
        vals = [f[key] for f in folds if key in f]
        if vals and isinstance(vals[0], (int, float)):
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))

    return summary


def compute_stgcnpp_summaries(
    work_dir: Path,
    modalities: List[str],
    task: str,
    loss: str,
    num_folds: int = 3,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Compute per-modality and fusion summaries for STGCN++.

    Args:
        work_dir: Base work directory (e.g., work_dirs/stgcnpp/tal/cv_4class_5class_ce_balanced)
        modalities: List of modalities (e.g., ['j', 'b', 'jm', 'bm'])
        task: Task type ('tal' or 'rmm')
        loss: Loss type ('ce' or 'focal')
        num_folds: Number of CV folds
        metadata: Additional metadata

    Returns:
        Dict with all modality summaries and fusion summary
    """
    all_summaries = {}

    # Per-modality summaries
    print("\n" + "=" * 60)
    print("Computing Per-Modality CV Summaries")
    print("=" * 60)

    for modality in modalities:
        print(f"\nModality: {modality}")
        modality_dir = work_dir / modality
        mod_metadata = {"task": task, "loss": loss}
        if metadata:
            mod_metadata.update(metadata)

        summary = compute_cv_summary(modality_dir, modality, num_folds, mod_metadata)

        if summary:
            # Save per-modality summary
            summary_file = modality_dir / "cv_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved: {summary_file}")

            # Print key metrics
            top1 = summary.get("clip_top1_acc_mean", 0)
            macro_f1 = summary.get("clip_macro_f1_mean", 0)
            print(f"  CV Results: top1={top1:.2f}% macro_f1={macro_f1:.2f}%")

            all_summaries[modality] = summary

    # 4-Stream Fusion Summary
    print("\n" + "=" * 60)
    print("Computing 4-Stream Fusion Summary")
    print("=" * 60)

    if all_summaries:
        fusion_summary: Dict[str, Any] = {
            "model": "STGCN++",
            "task": task,
            "loss": loss,
            "modalities": modalities,
            "per_modality": {},
        }

        if metadata:
            fusion_summary.update(metadata)

        # Extract key metrics per modality
        for mod, data in all_summaries.items():
            fusion_summary["per_modality"][mod] = {
                "top1_acc_mean": data.get("clip_top1_acc_mean", 0),
                "top1_acc_std": data.get("clip_top1_acc_std", 0),
                "macro_f1_mean": data.get("clip_macro_f1_mean", 0),
                "macro_f1_std": data.get("clip_macro_f1_std", 0),
                "macro_precision_mean": data.get("clip_macro_precision_mean", 0),
                "macro_recall_mean": data.get("clip_macro_recall_mean", 0),
                "rmm_vs_bg_f1_mean": data.get("clip_rmm_vs_bg_f1_mean", 0),
                "rmm_vs_bg_auc_mean": data.get("clip_rmm_vs_bg_auc_mean", 0),
                "rmm_only_macro_f1_mean": data.get("clip_rmm_only_macro_f1_mean", 0),
            }

        # Compute ensemble averages
        top1_means = [
            v["top1_acc_mean"] for v in fusion_summary["per_modality"].values()
        ]
        macro_f1_means = [
            v["macro_f1_mean"] for v in fusion_summary["per_modality"].values()
        ]
        rmm_bg_f1_means = [
            v["rmm_vs_bg_f1_mean"]
            for v in fusion_summary["per_modality"].values()
            if v["rmm_vs_bg_f1_mean"] > 0
        ]

        fusion_summary["ensemble_avg_top1"] = float(np.mean(top1_means))
        fusion_summary["ensemble_avg_macro_f1"] = float(np.mean(macro_f1_means))
        if rmm_bg_f1_means:
            fusion_summary["ensemble_avg_rmm_vs_bg_f1"] = float(np.mean(rmm_bg_f1_means))

        # Save fusion summary
        fusion_file = work_dir / "fusion_summary.json"
        with open(fusion_file, "w") as f:
            json.dump(fusion_summary, f, indent=2)
        print(f"\nSaved: {fusion_file}")

        # Print results
        print("\n4-Stream Results:")
        for mod, v in fusion_summary["per_modality"].items():
            rmm_bg = v.get("rmm_vs_bg_f1_mean", 0)
            rmm_bg_str = f" rmm_vs_bg_f1={rmm_bg:.2f}%" if rmm_bg > 0 else ""
            print(
                f"  {mod}: top1={v['top1_acc_mean']:.2f}% "
                f"macro_f1={v['macro_f1_mean']:.2f}%{rmm_bg_str}"
            )
        print(f"\n  Ensemble Avg Top1: {fusion_summary['ensemble_avg_top1']:.2f}%")
        print(f"  Ensemble Avg Macro-F1: {fusion_summary['ensemble_avg_macro_f1']:.2f}%")
        if "ensemble_avg_rmm_vs_bg_f1" in fusion_summary:
            print(
                f"  Ensemble Avg RMM vs BG F1: {fusion_summary['ensemble_avg_rmm_vs_bg_f1']:.2f}%"
            )

        all_summaries["_fusion"] = fusion_summary

    return all_summaries


def compute_posec3d_summary(
    work_dir: Path,
    task: str,
    loss: str,
    num_folds: int = 3,
    metadata: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute CV summary for PoseC3D (single model, no modalities).

    Args:
        work_dir: Base work directory (e.g., work_dirs/posec3d/tal/cv_4class_5class_ce_balanced)
        task: Task type ('tal' or 'rmm')
        loss: Loss type ('ce' or 'focal')
        num_folds: Number of CV folds
        metadata: Additional metadata

    Returns:
        Summary dict
    """
    print("\n" + "=" * 60)
    print("Computing PoseC3D CV Summary")
    print("=" * 60)

    full_metadata = {"model": "PoseC3D", "task": task, "loss": loss}
    if metadata:
        full_metadata.update(metadata)

    summary = compute_cv_summary(work_dir, None, num_folds, full_metadata)

    if summary:
        # Save summary
        summary_file = work_dir / "cv_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {summary_file}")

        # Print key metrics
        print("\nCV Results:")
        top1 = summary.get("clip_top1_acc_mean", 0)
        top1_std = summary.get("clip_top1_acc_std", 0)
        macro_f1 = summary.get("clip_macro_f1_mean", 0)
        macro_f1_std = summary.get("clip_macro_f1_std", 0)
        macro_prec = summary.get("clip_macro_precision_mean", 0)
        macro_rec = summary.get("clip_macro_recall_mean", 0)

        print(f"  Clip Top-1: {top1:.2f} ± {top1_std:.2f}%")
        print(f"  Macro-F1: {macro_f1:.2f} ± {macro_f1_std:.2f}%")
        print(f"  Macro-Precision: {macro_prec:.2f}%  Macro-Recall: {macro_rec:.2f}%")

        # TAL-specific metrics
        rmm_bg_f1 = summary.get("clip_rmm_vs_bg_f1_mean", 0)
        if rmm_bg_f1 > 0:
            rmm_bg_auc = summary.get("clip_rmm_vs_bg_auc_mean", 0)
            rmm_only_f1 = summary.get("clip_rmm_only_macro_f1_mean", 0)
            print(f"  RMM vs BG F1: {rmm_bg_f1:.2f}%  AUC: {rmm_bg_auc:.2f}%")
            print(f"  RMM-Only Macro-F1: {rmm_only_f1:.2f}%")

    return summary


def parse_metadata(metadata_args: Optional[List[str]]) -> Dict[str, str]:
    """Parse metadata key=value pairs from command line."""
    if not metadata_args:
        return {}

    result = {}
    for item in metadata_args:
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
        else:
            print(f"Warning: Ignoring invalid metadata format: {item}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Summarize cross-validation results for TAL/RMM experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Base work directory containing fold subdirectories",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["stgcnpp", "posec3d"],
        help="Model type",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["tal", "rmm"],
        help="Task type",
    )
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        choices=["ce", "focal"],
        help="Loss function used",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["j", "b", "jm", "bm"],
        help="Modalities for STGCN++ (default: j b jm bm)",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=3,
        help="Number of CV folds (default: 3)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        nargs="*",
        help="Additional metadata as key=value pairs (e.g., 'class_prob=[1.0,1.0]' 'patience=5')",
    )

    args = parser.parse_args()

    # Validate work directory
    if not args.work_dir.exists():
        print(f"Error: Work directory does not exist: {args.work_dir}")
        sys.exit(1)

    # Parse metadata
    metadata = parse_metadata(args.metadata)

    print("=" * 60)
    print(f"CV Results Summary")
    print("=" * 60)
    print(f"Model: {args.model.upper()}")
    print(f"Task: {args.task.upper()}")
    print(f"Loss: {args.loss.upper()}")
    print(f"Work Dir: {args.work_dir}")
    if metadata:
        print(f"Metadata: {metadata}")

    # Compute summaries based on model type
    if args.model == "stgcnpp":
        compute_stgcnpp_summaries(
            args.work_dir,
            args.modalities,
            args.task,
            args.loss,
            args.num_folds,
            metadata,
        )
    else:  # posec3d
        compute_posec3d_summary(
            args.work_dir,
            args.task,
            args.loss,
            args.num_folds,
            metadata,
        )

    print("\n" + "=" * 60)
    print("Summary complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()










