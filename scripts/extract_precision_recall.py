#!/usr/bin/env python3
"""
Extract precision/recall metrics from all model types for MODEL_COMPARISON.md.

Handles 4 different data formats:
- PySKL: Direct from cv_summary.json
- Fusion: Aggregate from per-fold data in cv_summary.json
- V-JEPA: Compute from per_class_clip.csv files
- Qwen: From metrics.json and predictions CSV

Usage:
    python scripts/extract_precision_recall.py
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass


# Base paths
# Resolve paths relative to the repo root so this runs from any clone location.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())

ACTREG_ROOT = _REPO_ROOT
# Qwen-VL baseline outputs (metrics/predictions only, ~1.3 MB) are vendored into
# the repo so this comparison reproduces without the original scratch run dir.
QWEN_ROOT = _REPO_ROOT / "insights/vlm/qwen_outputs"


@dataclass
class ModelMetrics:
    """Container for model metrics."""
    name: str
    task: str  # "4class" or "5class"
    clip_top1: float
    clip_top1_std: float
    clip_top2: float
    clip_top2_std: float
    clip_macro_f1: float
    clip_macro_f1_std: float
    clip_macro_prec: float
    clip_macro_prec_std: float
    clip_macro_rec: float
    clip_macro_rec_std: float
    cohen_kappa: float
    cohen_kappa_std: float
    per_class: Optional[Dict[str, Dict[str, float]]] = None  # {class: {prec, rec, f1}}


def extract_pyskl_metrics(cv_summary_path: Path, name: str, task: str) -> ModelMetrics:
    """Extract metrics from PySKL cv_summary.json (PoseC3D, STGCN++)."""
    with open(cv_summary_path) as f:
        data = json.load(f)
    
    # Check if it's new format (with _mean suffix) or old format
    if "clip_top1_acc_mean" in data:
        # New format with direct mean/std
        return ModelMetrics(
            name=name,
            task=task,
            clip_top1=data.get("clip_top1_acc_mean", 0) / 100 if data.get("clip_top1_acc_mean", 0) > 1 else data.get("clip_top1_acc_mean", 0),
            clip_top1_std=data.get("clip_top1_acc_std", 0) / 100 if data.get("clip_top1_acc_std", 0) > 1 else data.get("clip_top1_acc_std", 0),
            clip_top2=data.get("clip_top2_acc_mean", 0) / 100 if data.get("clip_top2_acc_mean", 0) > 1 else data.get("clip_top2_acc_mean", 0),
            clip_top2_std=data.get("clip_top2_acc_std", 0) / 100 if data.get("clip_top2_acc_std", 0) > 1 else data.get("clip_top2_acc_std", 0),
            clip_macro_f1=data.get("clip_macro_f1_mean", 0) / 100 if data.get("clip_macro_f1_mean", 0) > 1 else data.get("clip_macro_f1_mean", 0),
            clip_macro_f1_std=data.get("clip_macro_f1_std", 0) / 100 if data.get("clip_macro_f1_std", 0) > 1 else data.get("clip_macro_f1_std", 0),
            clip_macro_prec=data.get("clip_macro_precision_mean", 0) / 100 if data.get("clip_macro_precision_mean", 0) > 1 else data.get("clip_macro_precision_mean", 0),
            clip_macro_prec_std=data.get("clip_macro_precision_std", 0) / 100 if data.get("clip_macro_precision_std", 0) > 1 else data.get("clip_macro_precision_std", 0),
            clip_macro_rec=data.get("clip_macro_recall_mean", 0) / 100 if data.get("clip_macro_recall_mean", 0) > 1 else data.get("clip_macro_recall_mean", 0),
            clip_macro_rec_std=data.get("clip_macro_recall_std", 0) / 100 if data.get("clip_macro_recall_std", 0) > 1 else data.get("clip_macro_recall_std", 0),
            cohen_kappa=data.get("clip_cohens_kappa_mean", 0),
            cohen_kappa_std=data.get("clip_cohens_kappa_std", 0),
            per_class=_extract_pyskl_per_class(data),
        )
    elif "average" in data:
        # Old format with average/std structure
        avg = data["average"]
        std = data.get("std", {})
        return ModelMetrics(
            name=name,
            task=task,
            clip_top1=avg.get("clip_top1_acc", 0) / 100 if avg.get("clip_top1_acc", 0) > 1 else avg.get("clip_top1_acc", 0),
            clip_top1_std=std.get("clip_top1_acc", 0) / 100 if std.get("clip_top1_acc", 0) > 1 else std.get("clip_top1_acc", 0),
            clip_top2=avg.get("clip_top2_acc", 0) / 100 if avg.get("clip_top2_acc", 0) > 1 else avg.get("clip_top2_acc", 0),
            clip_top2_std=std.get("clip_top2_acc", 0) / 100 if std.get("clip_top2_acc", 0) > 1 else std.get("clip_top2_acc", 0),
            clip_macro_f1=avg.get("clip_macro_f1", 0) / 100 if avg.get("clip_macro_f1", 0) > 1 else avg.get("clip_macro_f1", 0),
            clip_macro_f1_std=std.get("clip_macro_f1", 0) / 100 if std.get("clip_macro_f1", 0) > 1 else std.get("clip_macro_f1", 0),
            clip_macro_prec=avg.get("clip_macro_precision", 0) / 100 if avg.get("clip_macro_precision", 0) > 1 else avg.get("clip_macro_precision", 0),
            clip_macro_prec_std=std.get("clip_macro_precision", 0) / 100 if std.get("clip_macro_precision", 0) > 1 else std.get("clip_macro_precision", 0),
            clip_macro_rec=avg.get("clip_macro_recall", 0) / 100 if avg.get("clip_macro_recall", 0) > 1 else avg.get("clip_macro_recall", 0),
            clip_macro_rec_std=std.get("clip_macro_recall", 0) / 100 if std.get("clip_macro_recall", 0) > 1 else std.get("clip_macro_recall", 0),
            cohen_kappa=avg.get("clip_cohens_kappa", 0),
            cohen_kappa_std=std.get("clip_cohens_kappa", 0),
            per_class=_extract_pyskl_per_class_old_format(avg, std),
        )
    else:
        raise ValueError(f"Unknown PySKL cv_summary format in {cv_summary_path}")


def _extract_pyskl_per_class(data: dict) -> Dict[str, Dict[str, float]]:
    """Extract per-class metrics from new format PySKL cv_summary."""
    per_class = {}
    # Look for keys like clip_prec_hands flapping_mean
    for key in data:
        if key.startswith("clip_prec_") and key.endswith("_mean"):
            class_name = key[len("clip_prec_"):-len("_mean")]
            if class_name not in per_class:
                per_class[class_name] = {}
            val = data[key]
            per_class[class_name]["prec"] = val / 100 if val > 1 else val
            std_key = key.replace("_mean", "_std")
            std_val = data.get(std_key, 0)
            per_class[class_name]["prec_std"] = std_val / 100 if std_val > 1 else std_val
        elif key.startswith("clip_rec_") and key.endswith("_mean"):
            class_name = key[len("clip_rec_"):-len("_mean")]
            if class_name not in per_class:
                per_class[class_name] = {}
            val = data[key]
            per_class[class_name]["rec"] = val / 100 if val > 1 else val
            std_key = key.replace("_mean", "_std")
            std_val = data.get(std_key, 0)
            per_class[class_name]["rec_std"] = std_val / 100 if std_val > 1 else std_val
        elif key.startswith("clip_f1_") and key.endswith("_mean"):
            class_name = key[len("clip_f1_"):-len("_mean")]
            if class_name not in per_class:
                per_class[class_name] = {}
            val = data[key]
            per_class[class_name]["f1"] = val / 100 if val > 1 else val
            std_key = key.replace("_mean", "_std")
            std_val = data.get(std_key, 0)
            per_class[class_name]["f1_std"] = std_val / 100 if std_val > 1 else std_val
    return per_class if per_class else None


def _extract_pyskl_per_class_old_format(avg: dict, std: dict) -> Dict[str, Dict[str, float]]:
    """Extract per-class metrics from old format PySKL cv_summary."""
    per_class = {}
    for key in avg:
        if key.startswith("clip_prec_"):
            class_name = key[len("clip_prec_"):]
            if class_name not in per_class:
                per_class[class_name] = {}
            val = avg[key]
            per_class[class_name]["prec"] = val / 100 if val > 1 else val
            std_val = std.get(key, 0)
            per_class[class_name]["prec_std"] = std_val / 100 if std_val > 1 else std_val
        elif key.startswith("clip_rec_"):
            class_name = key[len("clip_rec_"):]
            if class_name not in per_class:
                per_class[class_name] = {}
            val = avg[key]
            per_class[class_name]["rec"] = val / 100 if val > 1 else val
            std_val = std.get(key, 0)
            per_class[class_name]["rec_std"] = std_val / 100 if std_val > 1 else std_val
        elif key.startswith("clip_f1_"):
            class_name = key[len("clip_f1_"):]
            if class_name not in per_class:
                per_class[class_name] = {}
            val = avg[key]
            per_class[class_name]["f1"] = val / 100 if val > 1 else val
            std_val = std.get(key, 0)
            per_class[class_name]["f1_std"] = std_val / 100 if std_val > 1 else std_val
    return per_class if per_class else None


def extract_fusion_metrics(cv_summary_path: Path, name: str, task: str) -> ModelMetrics:
    """Extract metrics from Fusion cv_summary.json (aggregating from per-fold)."""
    with open(cv_summary_path) as f:
        data = json.load(f)
    
    # Get aggregated clip metrics
    clip = data.get("clip", {})
    
    # For precision/recall, need to aggregate from per-fold data
    folds = data.get("folds", [])
    prec_vals = []
    rec_vals = []
    for fold in folds:
        fold_clip = fold.get("clip", {})
        if "macro_precision" in fold_clip:
            prec_vals.append(fold_clip["macro_precision"])
        if "macro_recall" in fold_clip:
            rec_vals.append(fold_clip["macro_recall"])
    
    prec_mean = np.mean(prec_vals) if prec_vals else 0
    prec_std = np.std(prec_vals) if prec_vals else 0
    rec_mean = np.mean(rec_vals) if rec_vals else 0
    rec_std = np.std(rec_vals) if rec_vals else 0
    
    return ModelMetrics(
        name=name,
        task=task,
        clip_top1=clip.get("top1_acc_mean", 0),
        clip_top1_std=clip.get("top1_acc_std", 0),
        clip_top2=clip.get("top2_acc_mean", 0),
        clip_top2_std=clip.get("top2_acc_std", 0),
        clip_macro_f1=clip.get("macro_f1_mean", 0),
        clip_macro_f1_std=clip.get("macro_f1_std", 0),
        clip_macro_prec=prec_mean,
        clip_macro_prec_std=prec_std,
        clip_macro_rec=rec_mean,
        clip_macro_rec_std=rec_std,
        cohen_kappa=clip.get("cohen_kappa_mean", 0),
        cohen_kappa_std=clip.get("cohen_kappa_std", 0),
        per_class=_extract_fusion_per_class(folds),
    )


def _extract_fusion_per_class(folds: List[dict]) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Extract per-class metrics from fusion folds.
    Fusion doesn't store per-class in cv_summary, so return None.
    Would need to compute from predictions_clip.csv if needed.
    """
    # Fusion models don't have per-class in cv_summary
    # Would need to load predictions_clip.csv to compute
    return None


def extract_vjepa_metrics(run_dir: Path, name: str, task: str) -> ModelMetrics:
    """Extract metrics from V-JEPA run directory (per_class_clip.csv per fold)."""
    # Load cv_summary.json for overall metrics
    cv_summary_path = run_dir / "cv_summary.json"
    with open(cv_summary_path) as f:
        cv_data = json.load(f)
    
    # cv_data is a list of fold results
    if isinstance(cv_data, list):
        fold_data = cv_data
    else:
        fold_data = cv_data.get("folds", cv_data)
    
    # Aggregate per-fold metrics
    top1_vals, top2_vals, f1_vals, kappa_vals = [], [], [], []
    per_class_all = {}  # {class: {prec: [], rec: [], f1: []}}
    
    for fold_idx, fold in enumerate(fold_data):
        if isinstance(fold, dict):
            top1_vals.append(fold.get("top1_acc", 0))
            top2_vals.append(fold.get("top2_acc", 0))
            f1_vals.append(fold.get("macro_f1", 0))
            kappa_vals.append(fold.get("video_kappa", 0))
        
        # Load per-class from CSV
        per_class_csv = run_dir / f"fold_{fold_idx}" / "per_class_clip.csv"
        if per_class_csv.exists():
            df = pd.read_csv(per_class_csv)
            for _, row in df.iterrows():
                cls = row["class"]
                if cls not in per_class_all:
                    per_class_all[cls] = {"prec": [], "rec": [], "f1": []}
                per_class_all[cls]["prec"].append(row["precision"])
                per_class_all[cls]["rec"].append(row["recall"])
                per_class_all[cls]["f1"].append(row["f1"])
    
    # Compute macro precision/recall from per-class averages
    all_prec, all_rec = [], []
    per_class_final = {}
    for cls, vals in per_class_all.items():
        prec_mean = np.mean(vals["prec"])
        rec_mean = np.mean(vals["rec"])
        f1_mean = np.mean(vals["f1"])
        all_prec.append(prec_mean)
        all_rec.append(rec_mean)
        per_class_final[cls] = {
            "prec": prec_mean,
            "prec_std": np.std(vals["prec"]),
            "rec": rec_mean,
            "rec_std": np.std(vals["rec"]),
            "f1": f1_mean,
            "f1_std": np.std(vals["f1"]),
        }
    
    macro_prec = np.mean(all_prec) if all_prec else 0
    macro_rec = np.mean(all_rec) if all_rec else 0
    
    # For std of macro precision/recall, compute per-fold macro then std
    fold_macro_prec, fold_macro_rec = [], []
    for fold_idx in range(len(fold_data)):
        per_class_csv = run_dir / f"fold_{fold_idx}" / "per_class_clip.csv"
        if per_class_csv.exists():
            df = pd.read_csv(per_class_csv)
            fold_macro_prec.append(df["precision"].mean())
            fold_macro_rec.append(df["recall"].mean())
    
    return ModelMetrics(
        name=name,
        task=task,
        clip_top1=np.mean(top1_vals) if top1_vals else 0,
        clip_top1_std=np.std(top1_vals) if top1_vals else 0,
        clip_top2=np.mean(top2_vals) if top2_vals else 0,
        clip_top2_std=np.std(top2_vals) if top2_vals else 0,
        clip_macro_f1=np.mean(f1_vals) if f1_vals else 0,
        clip_macro_f1_std=np.std(f1_vals) if f1_vals else 0,
        clip_macro_prec=macro_prec,
        clip_macro_prec_std=np.std(fold_macro_prec) if fold_macro_prec else 0,
        clip_macro_rec=macro_rec,
        clip_macro_rec_std=np.std(fold_macro_rec) if fold_macro_rec else 0,
        cohen_kappa=np.mean(kappa_vals) if kappa_vals else 0,
        cohen_kappa_std=np.std(kappa_vals) if kappa_vals else 0,
        per_class=per_class_final if per_class_final else None,
    )


def extract_qwen_metrics(metrics_path: Path, preds_path: Path, name: str, task: str) -> ModelMetrics:
    """Extract metrics from Qwen metrics.json and predictions CSV."""
    with open(metrics_path) as f:
        data = json.load(f)
    
    agg = data.get("aggregate_clip_level", {})
    fold_mean = data.get("per_fold_mean", {})
    fold_std = data.get("per_fold_std", {})
    
    # Compute per-class from predictions
    per_class = None
    if preds_path.exists():
        df = pd.read_csv(preds_path)
        per_class = _compute_per_class_from_preds(df)
    
    return ModelMetrics(
        name=name,
        task=task,
        clip_top1=agg.get("top1_accuracy", 0),
        clip_top1_std=fold_std.get("top1_accuracy_std", 0),
        clip_top2=agg.get("top2_accuracy", 0),
        clip_top2_std=fold_std.get("top2_accuracy_std", 0),
        clip_macro_f1=agg.get("macro_f1", 0),
        clip_macro_f1_std=fold_std.get("macro_f1_std", 0),
        clip_macro_prec=agg.get("macro_precision", 0),
        clip_macro_prec_std=fold_std.get("macro_precision_std", 0),
        clip_macro_rec=agg.get("macro_recall", 0),
        clip_macro_rec_std=fold_std.get("macro_recall_std", 0),
        cohen_kappa=agg.get("cohen_kappa", 0),
        cohen_kappa_std=fold_std.get("cohen_kappa_std", 0),
        per_class=per_class,
    )


def _compute_per_class_from_preds(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision/recall from predictions DataFrame."""
    from sklearn.metrics import precision_recall_fscore_support
    
    # Filter out rows with missing predictions
    df_valid = df.dropna(subset=["label", "prediction"])
    
    labels = df_valid["label"].astype(str).values
    preds = df_valid["prediction"].astype(str).values
    classes = sorted(df_valid["label"].unique())
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=classes, average=None, zero_division=0
    )
    
    per_class = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            "prec": prec[i],
            "rec": rec[i],
            "f1": f1[i],
            # No std available from aggregate predictions
            "prec_std": 0,
            "rec_std": 0,
            "f1_std": 0,
        }
    return per_class


def compute_fusion_per_class(pred_csv_paths: List[Path]) -> Dict[str, Dict[str, float]]:
    """Compute per-class metrics from fusion prediction CSVs across folds."""
    from sklearn.metrics import precision_recall_fscore_support
    
    all_labels, all_preds = [], []
    for path in pred_csv_paths:
        if path.exists():
            df = pd.read_csv(path)
            all_labels.extend(df["label_id"].values)
            all_preds.extend(df["pred_id"].values)
    
    if not all_labels:
        return {}
    
    # Get class names from first file
    df = pd.read_csv(pred_csv_paths[0])
    class_names = sorted(df["label_name"].unique())
    classes = list(range(len(class_names)))
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=classes, average=None, zero_division=0
    )
    
    per_class = {}
    for i, cls_name in enumerate(class_names):
        per_class[cls_name] = {
            "prec": prec[i],
            "rec": rec[i],
            "f1": f1[i],
        }
    return per_class


def format_pct(val: float, std: float = 0, decimals: int = 1) -> str:
    """Format as percentage with optional std."""
    pct = val * 100 if val <= 1 else val
    std_pct = std * 100 if std <= 1 else std
    if std > 0:
        return f"{pct:.{decimals}f}% ± {std_pct:.{decimals}f}%"
    return f"{pct:.{decimals}f}%"


def format_kappa(val: float, std: float = 0) -> str:
    """Format Cohen's kappa."""
    if std > 0:
        return f"{val:.3f} ± {std:.2f}"
    return f"{val:.3f}"


def print_summary_table(metrics_list: List[ModelMetrics], task: str):
    """Print summary table for a task."""
    print(f"\n{'='*80}")
    print(f"{task.upper()} TASK - CLIP-LEVEL METRICS")
    print(f"{'='*80}")
    
    # Header
    print(f"| {'Method':<45} | {'Top-1':<15} | {'Top-2':<15} | {'Macro F1':<15} | {'Macro Prec':<15} | {'Macro Rec':<15} | {'κ':<12} |")
    print(f"|{'-'*47}|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*14}|")
    
    for m in metrics_list:
        print(f"| {m.name:<45} | {format_pct(m.clip_top1, m.clip_top1_std):<15} | {format_pct(m.clip_top2, m.clip_top2_std):<15} | {format_pct(m.clip_macro_f1, m.clip_macro_f1_std):<15} | {format_pct(m.clip_macro_prec, m.clip_macro_prec_std):<15} | {format_pct(m.clip_macro_rec, m.clip_macro_rec_std):<15} | {format_kappa(m.cohen_kappa, m.cohen_kappa_std):<12} |")


def print_per_class_table(metrics_list: List[ModelMetrics], task: str, classes: List[str]):
    """Print per-class precision/recall table."""
    print(f"\n{'='*80}")
    print(f"{task.upper()} TASK - PER-CLASS PRECISION/RECALL (Best Models)")
    print(f"{'='*80}")
    
    # Header
    header = f"| {'Model':<30} |"
    for cls in classes:
        header += f" {cls[:15]:<15} |"
    print(header)
    
    sep = f"|{'-'*32}|"
    for _ in classes:
        sep += f"{'-'*17}|"
    print(sep)
    
    for m in metrics_list:
        if m.per_class is None:
            continue
        row = f"| {m.name:<30} |"
        for cls in classes:
            if cls in m.per_class:
                p = m.per_class[cls].get("prec", 0) * 100
                r = m.per_class[cls].get("rec", 0) * 100
                row += f" {p:.1f}/{r:.1f}%{' ':<6} |"
            else:
                row += f" {'—':<15} |"
        print(row)


def main():
    """Main function to extract and display all metrics."""
    
    # =========================================================================
    # 4-CLASS MODELS
    # =========================================================================
    print("\n" + "="*80)
    print("EXTRACTING 4-CLASS METRICS")
    print("="*80)
    
    metrics_4class = []
    
    # Fusion 3-Way MLP
    fusion_3way_4class = ACTREG_ROOT / "fusion/runs/4class_cv_3way/cv_summary.json"
    if fusion_3way_4class.exists():
        m = extract_fusion_metrics(fusion_3way_4class, "3-Way MLP (V-JEPA2 + PoseC3D + STGCN++)", "4class")
        # Compute per-class from predictions
        pred_paths = [ACTREG_ROOT / f"fusion/runs/4class_cv_3way/fold_{i}/predictions_clip.csv" for i in range(3)]
        m.per_class = compute_fusion_per_class(pred_paths)
        metrics_4class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # MLP Fusion (V-JEPA2 + STGCN++)
    fusion_stgcn_4class = ACTREG_ROOT / "fusion/runs/4class_cv_stgcn_mlp/cv_summary.json"
    if fusion_stgcn_4class.exists():
        m = extract_fusion_metrics(fusion_stgcn_4class, "MLP Fusion (V-JEPA2 + STGCN++)", "4class")
        metrics_4class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # MLP Fusion (V-JEPA2 + PoseC3D)
    fusion_posec3d_4class = ACTREG_ROOT / "fusion/runs/4class_cv_mlp/cv_summary.json"
    if fusion_posec3d_4class.exists():
        m = extract_fusion_metrics(fusion_posec3d_4class, "MLP Fusion (V-JEPA2 + PoseC3D)", "4class")
        metrics_4class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # STGCN++ 4-stream
    stgcn_4class = ACTREG_ROOT / "pyskl/work_dirs/stgcnpp/cv/4class_conf04_4stream/cv_summary.json"
    if stgcn_4class.exists():
        m = extract_pyskl_metrics(stgcn_4class, "STGCN++ (4-stream)", "4class")
        metrics_4class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # V-JEPA2 + SAM3 crop
    vjepa_4class = ACTREG_ROOT / "v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls"
    if vjepa_4class.exists():
        m = extract_vjepa_metrics(vjepa_4class, "V-JEPA2 + SAM3 crop", "4class")
        metrics_4class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # PoseC3D (non-weighted)
    posec3d_4class = ACTREG_ROOT / "pyskl/work_dirs/posec3d/cv/4class_conf04/cv_summary.json"
    if posec3d_4class.exists():
        m = extract_pyskl_metrics(posec3d_4class, "PoseC3D (non-weighted)", "4class")
        metrics_4class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # Qwen 4-class
    qwen_4class_metrics = QWEN_ROOT / "cv_4class_7489124/frames16_samples4/cv_cv_4class/metrics.json"
    qwen_4class_preds = QWEN_ROOT / "cv_4class_7489124/frames16_samples4/cv_cv_4class/predictions_clip_all_folds.csv"
    if qwen_4class_metrics.exists():
        m = extract_qwen_metrics(qwen_4class_metrics, qwen_4class_preds, "Qwen2.5-VL (zero-shot)", "4class")
        metrics_4class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    print_summary_table(metrics_4class, "4-class")
    
    # Per-class for best models
    classes_4 = ["hands flapping", "jumping", "rocking", "spinning"]
    best_4class = [m for m in metrics_4class if m.per_class is not None]
    if best_4class:
        print_per_class_table(best_4class, "4-class", classes_4)
    
    # =========================================================================
    # 5-CLASS MODELS
    # =========================================================================
    print("\n" + "="*80)
    print("EXTRACTING 5-CLASS METRICS")
    print("="*80)
    
    metrics_5class = []
    
    # Fusion 3-Way MLP
    fusion_3way_5class = ACTREG_ROOT / "fusion/runs/5class_cv_3way/cv_summary.json"
    if fusion_3way_5class.exists():
        m = extract_fusion_metrics(fusion_3way_5class, "3-Way MLP (V-JEPA2 + PoseC3D + STGCN++)", "5class")
        # Compute per-class from predictions
        pred_paths = [ACTREG_ROOT / f"fusion/runs/5class_cv_3way/fold_{i}/predictions_clip.csv" for i in range(3)]
        m.per_class = compute_fusion_per_class(pred_paths)
        metrics_5class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # MLP Fusion (V-JEPA2 + STGCN++)
    fusion_stgcn_5class = ACTREG_ROOT / "fusion/runs/5class_cv_stgcn_mlp/cv_summary.json"
    if fusion_stgcn_5class.exists():
        m = extract_fusion_metrics(fusion_stgcn_5class, "MLP Fusion (V-JEPA2 + STGCN++)", "5class")
        metrics_5class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # MLP Fusion (V-JEPA2 + PoseC3D)
    fusion_posec3d_5class = ACTREG_ROOT / "fusion/runs/5class_cv_posec3d_mlp/cv_summary.json"
    if fusion_posec3d_5class.exists():
        m = extract_fusion_metrics(fusion_posec3d_5class, "MLP Fusion (V-JEPA2 + PoseC3D)", "5class")
        metrics_5class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # PoseC3D + Weighted
    posec3d_5class = ACTREG_ROOT / "pyskl/work_dirs/posec3d/cv/5class_conf04_weighted/cv_summary.json"
    if posec3d_5class.exists():
        m = extract_pyskl_metrics(posec3d_5class, "PoseC3D + Weighted", "5class")
        metrics_5class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # STGCN++ 4-stream
    stgcn_5class = ACTREG_ROOT / "pyskl/work_dirs/stgcnpp/cv/5class_conf04_4stream/cv_summary.json"
    if stgcn_5class.exists():
        m = extract_pyskl_metrics(stgcn_5class, "STGCN++ 4-stream", "5class")
        metrics_5class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # STGCN++ 4-stream + Weighted
    stgcn_weighted_5class = ACTREG_ROOT / "pyskl/work_dirs/stgcnpp/cv/5class_conf04_weighted_4stream/cv_summary.json"
    if stgcn_weighted_5class.exists():
        m = extract_pyskl_metrics(stgcn_weighted_5class, "STGCN++ 4-stream + Weighted", "5class")
        metrics_5class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # PoseC3D + Weighted Sqrt
    posec3d_sqrt_5class = ACTREG_ROOT / "pyskl/work_dirs/posec3d/cv/5class_conf04_weighted_sqrt/cv_summary.json"
    if posec3d_sqrt_5class.exists():
        m = extract_pyskl_metrics(posec3d_sqrt_5class, "PoseC3D + Weighted Sqrt", "5class")
        metrics_5class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # PoseC3D + Focal Loss
    posec3d_focal_5class = ACTREG_ROOT / "pyskl/work_dirs/posec3d/cv/5class_conf04_focal/cv_summary.json"
    if posec3d_focal_5class.exists():
        m = extract_pyskl_metrics(posec3d_focal_5class, "PoseC3D + Focal Loss", "5class")
        metrics_5class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # STGCN++ 4-stream + Focal
    stgcn_focal_5class = ACTREG_ROOT / "pyskl/work_dirs/stgcnpp/cv/5class_conf04_focal_4stream/cv_summary.json"
    if stgcn_focal_5class.exists():
        m = extract_pyskl_metrics(stgcn_focal_5class, "STGCN++ 4-stream + Focal", "5class")
        metrics_5class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # V-JEPA2 + SAM3 crop
    vjepa_5class = ACTREG_ROOT / "v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop"
    if vjepa_5class.exists():
        m = extract_vjepa_metrics(vjepa_5class, "V-JEPA2 + SAM3 crop", "5class")
        metrics_5class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    # Qwen 5-class
    qwen_5class_metrics = QWEN_ROOT / "cv_5class_7489123/frames16_samples4/cv_cv/metrics.json"
    qwen_5class_preds = QWEN_ROOT / "cv_5class_7489123/frames16_samples4/cv_cv/predictions_clip_all_folds.csv"
    if qwen_5class_metrics.exists():
        m = extract_qwen_metrics(qwen_5class_metrics, qwen_5class_preds, "Qwen2.5-VL (zero-shot)", "5class")
        metrics_5class.append(m)
        print(f"✓ Loaded: {m.name}")
    
    print_summary_table(metrics_5class, "5-class")
    
    # Per-class for best models
    classes_5 = ["hands flapping", "jumping", "one hand flap", "rocking", "spinning"]
    best_5class = [m for m in metrics_5class if m.per_class is not None]
    if best_5class:
        print_per_class_table(best_5class, "5-class", classes_5)
    
    # =========================================================================
    # MARKDOWN OUTPUT
    # =========================================================================
    print("\n" + "="*80)
    print("MARKDOWN TABLES FOR MODEL_COMPARISON.md")
    print("="*80)
    
    # 4-class summary table (markdown)
    print("\n### 4-Class Task (Clip-Level Metrics)\n")
    print("| Rank | Method | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Clip Macro Prec | Clip Macro Rec | Cohen's κ |")
    print("|------|--------|------------|------------|---------------|-----------------|----------------|-----------|")
    for i, m in enumerate(metrics_4class, 1):
        rank = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else str(i)))
        name = f"**{m.name}**" if i == 1 else m.name
        print(f"| {rank} | {name} | {format_pct(m.clip_top1, m.clip_top1_std)} | {format_pct(m.clip_top2, m.clip_top2_std)} | {format_pct(m.clip_macro_f1, m.clip_macro_f1_std)} | {format_pct(m.clip_macro_prec, m.clip_macro_prec_std)} | {format_pct(m.clip_macro_rec, m.clip_macro_rec_std)} | {format_kappa(m.cohen_kappa, m.cohen_kappa_std)} |")
    
    # 5-class summary table (markdown)
    print("\n### 5-Class Task (Clip-Level Metrics)\n")
    print("| Rank | Method | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Clip Macro Prec | Clip Macro Rec | Cohen's κ |")
    print("|------|--------|------------|------------|---------------|-----------------|----------------|-----------|")
    for i, m in enumerate(metrics_5class, 1):
        rank = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else str(i)))
        name = f"**{m.name}**" if i == 1 else m.name
        print(f"| {rank} | {name} | {format_pct(m.clip_top1, m.clip_top1_std)} | {format_pct(m.clip_top2, m.clip_top2_std)} | {format_pct(m.clip_macro_f1, m.clip_macro_f1_std)} | {format_pct(m.clip_macro_prec, m.clip_macro_prec_std)} | {format_pct(m.clip_macro_rec, m.clip_macro_rec_std)} | {format_kappa(m.cohen_kappa, m.cohen_kappa_std)} |")
    
    # 4-class per-class table
    print("\n### 4-Class Per-Class Precision/Recall (Best Models)\n")
    print("| Model | hands_flapping | jumping | rocking | spinning |")
    print("|-------|----------------|---------|---------|----------|")
    for m in best_4class:
        row = f"| {m.name} |"
        for cls in classes_4:
            if cls in m.per_class:
                p = m.per_class[cls].get("prec", 0) * 100
                r = m.per_class[cls].get("rec", 0) * 100
                row += f" {p:.1f}% / {r:.1f}% |"
            else:
                row += " — |"
        print(row)
    
    # 5-class per-class table
    print("\n### 5-Class Per-Class Precision/Recall (Best Models)\n")
    print("| Model | hands_flapping | jumping | one_hand_flap | rocking | spinning |")
    print("|-------|----------------|---------|---------------|---------|----------|")
    for m in best_5class:
        row = f"| {m.name} |"
        for cls in classes_5:
            if cls in m.per_class:
                p = m.per_class[cls].get("prec", 0) * 100
                r = m.per_class[cls].get("rec", 0) * 100
                row += f" {p:.1f}% / {r:.1f}% |"
            else:
                row += " — |"
        print(row)


if __name__ == "__main__":
    main()
