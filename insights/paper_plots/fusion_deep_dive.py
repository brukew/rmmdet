#!/usr/bin/env python3
"""
Fusion Deep Dive: Clip-level vs Window-level vs TAL (segment-level).

This script supports the Chapter 6.1 narrative by verifying and quantifying
claims about fusion and metric mismatch across tasks:

1) Clip-level classification (4-class and 5-class):
   - Compare per-class precision/recall for Fusion vs single-modality models.
   - Summarize where fusion helps/hurts, highlighting class-specific trade-offs.

2) Window-level detection (5-class: 4 RMM + background):
   - Compute predicted background rate (argmax==background).
   - Compute per-class precision/recall/F1.
   - Compute macro-F1 over RMM classes only (matches Table 5.5 definition).
   - Show how fusion shifts the prediction distribution away from background,
     boosting recall but often crushing precision for rare classes.

3) Segment-level TAL (mAP):
   - Compare per-class AP for fusion backbones vs V-JEPA to reveal where the
     mAP gains are coming from (often tail classes).

Outputs:
  - insights/tables/fusion_deep_dive.json

Usage:
  python fusion_deep_dive.py
  python fusion_deep_dive.py --output-json /path/to/output.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# -----------------------------
# Paths / constants
# -----------------------------

ACTREG_ROOT = Path(__file__).parent.parent.parent

INSIGHTS_TABLES = ACTREG_ROOT / "insights" / "tables"
TAL_EVAL_RESULTS = ACTREG_ROOT / "tal" / "eval_results"
TAL_SPLITS_DIR = ACTREG_ROOT / "dataprep" / "tal" / "splits_cv_4class"

N_WD_CLASSES = 5
RMM_LABELS = [0, 1, 2, 3]
BG_LABEL = 4

WD_CLASS_NAMES = ["hands_flapping", "jumping", "rocking", "spinning", "background"]
SCORE_COLS = [f"score_class{i}" for i in range(N_WD_CLASSES)]


@dataclass(frozen=True)
class WindowModelSpec:
    """Where to find window-level predictions for a model."""

    name: str
    model_dir: Path
    preds_filename_candidates: Tuple[str, ...] = ("tal_format_preds.csv", "fused_tal_format_preds.csv")
    folds_expected: Tuple[int, ...] = (0, 1, 2)

    def resolve_preds_path(self, fold: int) -> Optional[Path]:
        """Return the first existing prediction file for fold, else None."""
        fold_dir = self.model_dir / f"fold{fold}"
        for fname in self.preds_filename_candidates:
            p = fold_dir / fname
            if p.exists():
                return p
        return None


def safe_read_json(path: Path) -> dict:
    """Read JSON with a helpful error."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path) as f:
        return json.load(f)


# -----------------------------
# Clip-level helpers
# -----------------------------

def summarize_clip_pr_tables() -> dict:
    """
    Summarize clip-level per-class precision/recall differences.

    Uses:
      - insights/tables/table_5_3_4class_pr.json
      - insights/tables/table_5_4_5class_pr.json
    """
    table_4c = safe_read_json(INSIGHTS_TABLES / "table_5_3_4class_pr.json")
    table_5c = safe_read_json(INSIGHTS_TABLES / "table_5_4_5class_pr.json")

    def comp(table: dict, classes: List[str]) -> dict:
        out = {}
        for cls in classes:
            out[cls] = {}
            for metric in ["precision", "recall"]:
                fusion = float(table["Fusion"]["per_class"][cls][metric]["mean"])
                vjepa = float(table["V-JEPA"]["per_class"][cls][metric]["mean"])
                pose = float(table["PoseC3D"]["per_class"][cls][metric]["mean"])
                stg = float(table["STGCN++"]["per_class"][cls][metric]["mean"])
                out[cls][metric] = {
                    "fusion": fusion,
                    "vjepa": vjepa,
                    "posec3d": pose,
                    "stgcnpp": stg,
                    "fusion_minus_vjepa": fusion - vjepa,
                    "fusion_minus_posec3d": fusion - pose,
                    "fusion_minus_stgcnpp": fusion - stg,
                }
        return out

    return {
        "clip_level": {
            "4class_per_class_pr": comp(table_4c, ["hands flapping", "jumping", "rocking", "spinning"]),
            "5class_per_class_pr": comp(
                table_5c,
                ["hands flapping", "jumping", "one hand flap", "rocking", "spinning"],
            ),
        }
    }


# -----------------------------
# Window-level helpers
# -----------------------------

def load_gt_windows(fold: int) -> pd.DataFrame:
    """Load validation windows for a fold and map primary_label -> 5-class id."""
    gt_path = TAL_SPLITS_DIR / f"fold_{fold}_val_windows.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT windows: {gt_path}")
    df = pd.read_csv(gt_path)
    df["y_true"] = df["primary_label"].apply(lambda x: BG_LABEL if int(x) == -1 else int(x))
    return df[["window_id", "y_true"]]


def compute_window_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute window-level metrics, including two macro-F1 variants:

    - macro_f1_rmm: macro-F1 over the 4 RMM classes only (0..3). This matches
      the definition used in insights/tables/table_5_5_window_detection.json.
    - macro_f1_all: macro-F1 over all 5 classes (0..4). This is included as
      extra context (background can inflate this value).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_WD_CLASSES)))

    per_class = {}
    for i, cls_name in enumerate(WD_CLASS_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_rmm": float(f1_score(y_true, y_pred, labels=RMM_LABELS, average="macro")),
        "macro_f1_all": float(f1_score(y_true, y_pred, labels=list(range(5)), average="macro")),
        "bg_pred_rate": float((y_pred == BG_LABEL).mean()),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def evaluate_window_models(model_specs: List[WindowModelSpec]) -> dict:
    """Compute fold-wise and aggregated window-level metrics for selected models."""
    out: Dict[str, dict] = {}

    for spec in model_specs:
        fold_metrics: List[dict] = []
        for fold in spec.folds_expected:
            preds_path = spec.resolve_preds_path(fold)
            if preds_path is None:
                continue

            gt_df = load_gt_windows(fold)
            pred_df = pd.read_csv(preds_path)
            merged = gt_df.merge(pred_df, on="window_id", how="inner")
            if len(merged) == 0:
                continue

            y_true = merged["y_true"].astype(int).to_numpy()
            y_pred = merged[SCORE_COLS].to_numpy().argmax(axis=1)

            m = compute_window_metrics(y_true, y_pred)
            m["fold"] = fold
            m["n"] = int(len(merged))
            fold_metrics.append(m)

        if not fold_metrics:
            continue

        # Aggregate scalar metrics as mean±std across folds
        scalar_keys = ["accuracy", "macro_f1_rmm", "macro_f1_all", "bg_pred_rate"]
        scalars = {}
        for k in scalar_keys:
            vals = np.array([fm[k] for fm in fold_metrics], dtype=float)
            scalars[k] = {"mean": float(vals.mean()), "std": float(vals.std())}

        # Aggregate per-class precision/recall/f1
        per_class_agg: Dict[str, dict] = {}
        for cls in WD_CLASS_NAMES:
            per_class_agg[cls] = {}
            for met in ["precision", "recall", "f1"]:
                vals = np.array([fm["per_class"][cls][met] for fm in fold_metrics], dtype=float)
                per_class_agg[cls][met] = {"mean": float(vals.mean()), "std": float(vals.std())}

        out[spec.name] = {
            "model_dir": str(spec.model_dir),
            "folds": [fm["fold"] for fm in fold_metrics],
            "aggregate": {"scalars": scalars, "per_class": per_class_agg},
        }

    return out


# -----------------------------
# Segment-level TAL helpers
# -----------------------------

def summarize_segment_map() -> dict:
    """
    Compare per-class AP of the fusion TAL backbone vs V-JEPA.

    Uses:
      - insights/tables/table_5_7_segment_tal_map.json
    """
    table = safe_read_json(INSIGHTS_TABLES / "table_5_7_segment_tal_map.json")
    models = {m["model"]: m for m in table["models"]}

    fusion_key = "V-JEPA + PoseC3D (MLP)"
    base_key = "V-JEPA"

    if fusion_key not in models or base_key not in models:
        return {"segment_level": {"available": False}}

    fusion = models[fusion_key]
    base = models[base_key]

    deltas = {}
    for tiou in ["tIoU=0.3", "tIoU=0.5", "tIoU=0.7"]:
        deltas[tiou] = {}
        for cls in ["hands flapping", "jumping", "rocking", "spinning"]:
            f = float(fusion["per_class_ap"][tiou][cls]["mean"])
            b = float(base["per_class_ap"][tiou][cls]["mean"])
            deltas[tiou][cls] = {"fusion": f, "vjepa": b, "delta": f - b}

    return {
        "segment_level": {
            "available": True,
            "fusion_model": fusion_key,
            "base_model": base_key,
            "avg_mAP": {
                "fusion": float(fusion["avg_mAP"]["mean"]),
                "vjepa": float(base["avg_mAP"]["mean"]),
                "delta": float(fusion["avg_mAP"]["mean"] - base["avg_mAP"]["mean"]),
            },
            "per_class_ap_delta": deltas,
        }
    }


def summarize_postprocessing_hpo() -> dict:
    """
    Summarize the best post-processing hyperparameters for models that have
    grid-search results available on disk.

    Why this matters:
      Fusion models tend to be more "RMM-happy" at the window level, so the best
      TAL score threshold can differ substantially across backbones.

    Files used (if present):
      - tal/eval_results/vjepa_balanced/grid_search_results.csv
      - tal/eval_results/vjepa_binary/grid_search_results.csv
      - tal/eval_results/grid_search_results.csv (repo-level; may contain a single model)
    """
    out: Dict[str, dict] = {}

    # Per-model grid-search CSVs with columns: thr,smooth_k,merge_gap_sec,avg_mAP
    for key in ["vjepa_balanced", "vjepa_binary"]:
        csv_path = TAL_EVAL_RESULTS / key / "grid_search_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if len(df) == 0 or "avg_mAP" not in df.columns:
            continue
        best = df.loc[df["avg_mAP"].idxmax()].to_dict()
        out[key] = {
            "grid_search_csv": str(csv_path),
            "best_params": {
                "thr": float(best["thr"]),
                "smooth_k": int(best["smooth_k"]),
                "merge_gap_sec": float(best["merge_gap_sec"]),
            },
            "best_avg_mAP": float(best["avg_mAP"]),
        }

    # Repo-level grid-search CSV (may include a 'model' column)
    repo_csv = TAL_EVAL_RESULTS / "grid_search_results.csv"
    if repo_csv.exists():
        df = pd.read_csv(repo_csv)
        if len(df) > 0 and "avg_mAP" in df.columns:
            if "model" in df.columns:
                for model_key, sub in df.groupby("model"):
                    best = sub.loc[sub["avg_mAP"].idxmax()].to_dict()
                    out[str(model_key)] = {
                        "grid_search_csv": str(repo_csv),
                        "model_name": best.get("model_name"),
                        "n_folds": int(best.get("n_folds", 0)) if "n_folds" in best else None,
                        "best_params": {
                            "thr": float(best["thr"]),
                            "smooth_k": int(best["smooth_k"]),
                            "merge_gap_sec": float(best["merge_gap_sec"]),
                        },
                        "best_avg_mAP": float(best["avg_mAP"]),
                    }
            else:
                best = df.loc[df["avg_mAP"].idxmax()].to_dict()
                out["repo_level"] = {
                    "grid_search_csv": str(repo_csv),
                    "best_params": {
                        "thr": float(best["thr"]),
                        "smooth_k": int(best["smooth_k"]),
                        "merge_gap_sec": float(best["merge_gap_sec"]),
                    },
                    "best_avg_mAP": float(best["avg_mAP"]),
                }

    return {"postprocessing_hpo": out}


def summarize_best_eval_postprocess(model_specs: List[WindowModelSpec]) -> dict:
    """
    Read best-eval postprocess configs (if present) from each model directory.

    Many window/TAL backbone eval directories include:
      foldX/best_eval/eval_config.json

    This is often the most direct place to find the chosen post-processing params
    for that fold, even when a consolidated grid-search CSV is missing.
    """
    out: Dict[str, dict] = {}

    for spec in model_specs:
        fold_params = []
        for fold in spec.folds_expected:
            cfg_path = spec.model_dir / f"fold{fold}" / "best_eval" / "eval_config.json"
            if not cfg_path.exists():
                continue
            cfg = safe_read_json(cfg_path)
            post = cfg.get("postprocess", {})
            if not post:
                continue
            fold_params.append(
                {
                    "fold": int(fold),
                    "eval_config": str(cfg_path),
                    "thr": float(post.get("thr")),
                    "smooth_k": int(post.get("smooth_k")),
                    "merge_gap_sec": float(post.get("merge_gap_sec")),
                }
            )

        if not fold_params:
            continue

        thrs = np.array([p["thr"] for p in fold_params], dtype=float)
        smooths = np.array([p["smooth_k"] for p in fold_params], dtype=float)
        gaps = np.array([p["merge_gap_sec"] for p in fold_params], dtype=float)

        out[spec.name] = {
            "model_dir": str(spec.model_dir),
            "fold_params": fold_params,
            "aggregate": {
                "thr": {"mean": float(thrs.mean()), "std": float(thrs.std())},
                "smooth_k": {"mean": float(smooths.mean()), "std": float(smooths.std())},
                "merge_gap_sec": {"mean": float(gaps.mean()), "std": float(gaps.std())},
            },
        }

    return {"tal_best_eval_postprocess": out}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion deep dive analysis")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(INSIGHTS_TABLES / "fusion_deep_dive.json"),
        help="Where to write the analysis JSON.",
    )
    args = parser.parse_args()

    window_models = [
        # single-modality window detectors
        WindowModelSpec("V-JEPA", TAL_EVAL_RESULTS / "vjepa", folds_expected=(0, 1, 2)),
        WindowModelSpec("V-JEPA (Balanced)", TAL_EVAL_RESULTS / "vjepa_balanced", folds_expected=(0, 1, 2)),
        WindowModelSpec("PoseC3D (CE)", TAL_EVAL_RESULTS / "posec3d_ce", folds_expected=(0, 1)),
        WindowModelSpec("PoseC3D (CE Balanced)", TAL_EVAL_RESULTS / "posec3d_ce_balanced", folds_expected=(0, 1, 2)),
        WindowModelSpec("STGCN++ (CE, 4-stream)", TAL_EVAL_RESULTS / "stgcnpp_ce_4stream", folds_expected=(0, 1)),
        # fusion window detectors
        WindowModelSpec(
            "Fusion: V-JEPA + PoseC3D (MLP)",
            TAL_EVAL_RESULTS / "vjepa_posec3d_mlp_logp",
            folds_expected=(0, 1),
        ),
        WindowModelSpec(
            "Fusion: V-JEPA + STGCN++ (MLP)",
            TAL_EVAL_RESULTS / "vjepa_stgcnpp_mlp_logp",
            folds_expected=(0, 1),
        ),
        WindowModelSpec(
            "Fusion: V-JEPA + PoseC3D Bal (MLP)",
            TAL_EVAL_RESULTS / "vjepa_posec3d_bal_mlp_logp",
            folds_expected=(0, 1),
        ),
        WindowModelSpec(
            "Fusion: V-JEPA Bal + PoseC3D (MLP)",
            TAL_EVAL_RESULTS / "vjepa_balanced_posec3d_mlp_logp",
            folds_expected=(0, 1),
        ),
    ]

    output: Dict[str, dict] = {}
    output.update(summarize_clip_pr_tables())
    output["window_level"] = evaluate_window_models(window_models)
    output.update(summarize_segment_map())
    output.update(summarize_postprocessing_hpo())
    output.update(summarize_best_eval_postprocess(window_models))

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

