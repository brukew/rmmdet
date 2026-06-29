#!/usr/bin/env python3
"""
Comprehensive analysis of classifier effect (MLP vs V-JEPA2) and
two-stage vs end-to-end performance across all runs in the leaderboard table.

Outputs: two-stg/analysis_classifier_effect.md

Analyses:
  Part 1 — MLP Effect (same proposals, different classifiers)
    1a. Classification accuracy on GT-overlapping proposals (IoU >= 0.3)
    1b. Head-to-head label agreement (1 vs 3c, 5 vs 5c)
    1c. Per-class precision/recall on GT-overlapping proposals
    1d. AUROC for TP vs FP ranking (binary: does score separate TPs from FPs?)
    1e. Score distribution comparison (TP vs FP, per classifier)

  Part 2 — Two-Stage vs E2E / Error Taxonomy
    2a. Error taxonomy across all runs (no-overlap, low-IoU, wrong-class, boundary, correct)
    2b. Per-class error breakdown
    2c. E2E vs two-stage comparison (ActionFormer Balanced vs Runs 1, 3c)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ACTREG = Path("/orcd/data/satra/001/users/brukew/actreg")
TWO_STG = ACTREG / "two-stg"
OPENTAD = ACTREG / "OpenTAD"
ANNO_DIR = OPENTAD / "data/sails_rmm/annotations"

CLASS_NAMES = ["hands flapping", "jumping", "rocking", "spinning"]
# Must match LABEL_MAP_4CLASS in tal/tal_map_eval.py
ID2LABEL = {0: "hands flapping", 1: "jumping", 2: "rocking", 3: "spinning"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
FOLDS = [0, 1, 2]

# ── Run definitions ──────────────────────────────────────────────────────
# Two-stage runs: predictions.csv (video_key, class_id, label, start_sec, end_sec, score)
TWO_STAGE_RUNS = {
    "AF + V-JEPA2 (Run 1)":   TWO_STG / "eval_results",
    "AF + MLP (Run 3c)":      TWO_STG / "eval_results_3way_live",
    "TD + V-JEPA2 (Run 5)":   TWO_STG / "eval_results_tridet",
    "TD + MLP (Run 5c)":      TWO_STG / "eval_results_tridet_3way_live",
}

# E2E runs: result_detection.json (label, segment, score per proposal)
E2E_RUNS = {
    "AF E2E (baseline)": {
        fold: OPENTAD / f"exps/sails_rmm/actionformer_vjepa_balanced_fold{fold}/gpu1_id99/result_detection.json"
        for fold in FOLDS
    },
    "TD E2E (Run 6)": {
        fold: OPENTAD / f"exps/sails_rmm/tridet_vjepa_balanced_fold{fold}/gpu1_id99/result_detection.json"
        for fold in FOLDS
    },
}

# Head-to-head pairs: (V-JEPA2 run, MLP run) sharing the same proposals
H2H_PAIRS = [
    ("AF + V-JEPA2 (Run 1)", "AF + MLP (Run 3c)", "ActionFormer"),
    ("TD + V-JEPA2 (Run 5)", "TD + MLP (Run 5c)", "TriDet"),
]


def load_annotation(fold: int) -> dict:
    """Load OpenTAD GT annotation for a fold."""
    p = ANNO_DIR / f"fold{fold}_anno.json"
    return json.loads(p.read_text())


def _build_videokey_mapping(fold: int) -> dict[str, str]:
    """
    Build mapping from full video_key paths to OpenTAD-style keys
    using Run 3c predictions (which have both columns).
    """
    ref_path = TWO_STG / "eval_results_3way_live" / f"fold{fold}" / "predictions.csv"
    if not ref_path.exists():
        return {}
    ref = pd.read_csv(ref_path)
    if "video_key" in ref.columns and "opentad_video_key" in ref.columns:
        mapping = dict(zip(ref["video_key"], ref["opentad_video_key"]))
        return mapping
    return {}


_VK_CACHE: dict[int, dict[str, str]] = {}


def _get_videokey_mapping(fold: int) -> dict[str, str]:
    if fold not in _VK_CACHE:
        _VK_CACHE[fold] = _build_videokey_mapping(fold)
    return _VK_CACHE[fold]


def load_two_stage_preds(run_dir: Path, fold: int) -> pd.DataFrame | None:
    """Load predictions.csv for a two-stage run fold."""
    p = run_dir / f"fold{fold}" / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "label" not in df.columns and "class_id" in df.columns:
        df["label"] = df["class_id"].map(ID2LABEL)
    if "opentad_video_key" not in df.columns:
        vk_map = _get_videokey_mapping(fold)
        if vk_map and "video_key" in df.columns:
            df["opentad_video_key"] = df["video_key"].map(vk_map)
        elif "video_key" in df.columns:
            df["opentad_video_key"] = df["video_key"]
    return df


def load_e2e_preds(json_path: Path) -> pd.DataFrame | None:
    """Load result_detection.json and convert to DataFrame matching two-stage format."""
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    results = data.get("results", {})
    rows = []
    for vid, props in results.items():
        for p in props:
            label = p["label"]
            if label not in LABEL2ID:
                continue
            rows.append({
                "opentad_video_key": vid,
                "class_id": LABEL2ID[label],
                "label": label,
                "start_sec": p["segment"][0],
                "end_sec": p["segment"][1],
                "score": p["score"],
            })
    return pd.DataFrame(rows) if rows else None


def load_all_preds(run_name: str, fold: int) -> pd.DataFrame | None:
    """Unified loader for any run."""
    if run_name in TWO_STAGE_RUNS:
        return load_two_stage_preds(TWO_STAGE_RUNS[run_name], fold)
    if run_name in E2E_RUNS:
        return load_e2e_preds(E2E_RUNS[run_name][fold])
    return None


ALL_RUNS = list(TWO_STAGE_RUNS.keys()) + list(E2E_RUNS.keys())


def _merge_pair(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Merge two prediction DataFrames on matching proposals, handling key format differences."""
    merge_cols = ["opentad_video_key", "start_sec", "end_sec"]
    merged = df_a.merge(df_b, on=merge_cols, suffixes=("_vjepa", "_mlp"), how="inner")
    if len(merged) == 0 and "video_key" in df_a.columns and "video_key" in df_b.columns:
        merge_cols = ["video_key", "start_sec", "end_sec"]
        merged = df_a.merge(df_b, on=merge_cols, suffixes=("_vjepa", "_mlp"), how="inner")
    return merged


def tiou(seg1, seg2) -> float:
    """Temporal IoU between two [start, end] segments."""
    inter = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
    return inter / union if union > 0 else 0.0


def get_val_gt(anno: dict) -> dict[str, list[dict]]:
    """Extract val GT segments from OpenTAD annotation."""
    db = anno.get("database", {})
    gt = {}
    for vid, info in db.items():
        if info.get("subset") != "validation":
            continue
        segs = [
            {"label": a["label"], "segment": a["segment"]}
            for a in info.get("annotations", [])
            if a.get("label") in LABEL2ID
        ]
        if segs:
            gt[vid] = segs
    return gt


def match_preds_to_gt(preds: pd.DataFrame, val_gt: dict, tiou_thr: float = 0.3):
    """
    For each prediction, find its best-matching GT segment and classify the error.

    Returns a DataFrame with columns added: best_iou, gt_label, error_type, is_tp.
    """
    rows = []
    for _, row in preds.iterrows():
        vid = row["opentad_video_key"]
        gt_segs = val_gt.get(vid, [])
        pseg = [row["start_sec"], row["end_sec"]]

        best_iou = 0.0
        best_gt_label = None
        for gs in gt_segs:
            t = tiou(pseg, gs["segment"])
            if t > best_iou:
                best_iou = t
                best_gt_label = gs["label"]

        if best_iou == 0:
            error_type = "no_overlap"
        elif best_iou < tiou_thr:
            error_type = "low_iou"
        elif row["label"] != best_gt_label:
            error_type = "wrong_class"
        elif best_iou < 0.5:
            error_type = "boundary_error"
        else:
            error_type = "correct"

        is_tp = best_iou >= tiou_thr and row["label"] == best_gt_label
        rows.append({
            **row.to_dict(),
            "best_iou": best_iou,
            "gt_label": best_gt_label,
            "error_type": error_type,
            "is_tp": is_tp,
        })
    return pd.DataFrame(rows)


def main():
    lines: list[str] = []

    def W(s: str = ""):
        lines.append(s)

    W("# Classifier Effect & Two-Stage Analysis")
    W()
    W("All metrics are 3-fold cross-validation. GT-overlapping = best IoU ≥ 0.3 "
      "with any GT segment. TP = correct class + IoU ≥ 0.3.")
    W()

    # ── Load all data ─────────────────────────────────────────────────────
    # Per run × fold: matched predictions
    matched: dict[str, dict[int, pd.DataFrame]] = {}
    for run_name in ALL_RUNS:
        matched[run_name] = {}
        for fold in FOLDS:
            anno = load_annotation(fold)
            val_gt = get_val_gt(anno)
            preds = load_all_preds(run_name, fold)
            if preds is None or preds.empty:
                continue
            matched[run_name][fold] = match_preds_to_gt(preds, val_gt)

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: MLP EFFECT
    # ══════════════════════════════════════════════════════════════════════
    W("---")
    W()
    W("# Part 1: MLP Effect")
    W()
    W("Pairs share the **same detector proposals** (same start/end times). "
      "Only the Stage-2 classifier differs: V-JEPA2 alone vs 3-way MLP fusion "
      "(V-JEPA2 + PoseC3D + STGCN++).")
    W()

    # ── 1a. Classification accuracy on GT-overlapping proposals ──────────
    W("## 1a. Classification Accuracy on GT-Overlapping Proposals")
    W()
    W("For proposals with best IoU ≥ 0.3 to any GT segment, what fraction "
      "get the **correct class label**?")
    W()

    W("| Run | GT-overlap proposals | Correct label | Accuracy |")
    W("|-----|---------------------|---------------|----------|")
    for run_name in ALL_RUNS:
        all_overlapping = []
        for fold in FOLDS:
            df = matched[run_name].get(fold)
            if df is None:
                continue
            overlapping = df[df["best_iou"] >= 0.3]
            all_overlapping.append(overlapping)
        if not all_overlapping:
            continue
        pool = pd.concat(all_overlapping)
        n_overlap = len(pool)
        n_correct = (pool["label"] == pool["gt_label"]).sum()
        acc = n_correct / n_overlap if n_overlap > 0 else 0
        W(f"| {run_name} | {n_overlap:,} | {n_correct:,} | **{acc*100:.1f}%** |")
    W()

    # Per-class accuracy
    W("### Per-class accuracy (IoU ≥ 0.3, 3-fold pooled)")
    W()
    W("| GT Class | " + " | ".join(ALL_RUNS) + " |")
    W("|----------|" + "|".join(["------|"] * len(ALL_RUNS)))
    for cn in CLASS_NAMES:
        cells = []
        for run_name in ALL_RUNS:
            all_overlapping = []
            for fold in FOLDS:
                df = matched[run_name].get(fold)
                if df is None:
                    continue
                overlapping = df[(df["best_iou"] >= 0.3) & (df["gt_label"] == cn)]
                all_overlapping.append(overlapping)
            if not all_overlapping:
                cells.append("—")
                continue
            pool = pd.concat(all_overlapping)
            if len(pool) == 0:
                cells.append("—")
                continue
            acc = (pool["label"] == pool["gt_label"]).mean()
            cells.append(f"{acc*100:.1f}%")
        W(f"| {cn} | " + " | ".join(cells) + " |")
    W()

    # Balanced accuracy = mean of per-class accuracies
    W("### Balanced accuracy (mean of per-class accuracies)")
    W()
    W("Neutralizes the majority-class effect. "
      "Balanced acc = (1/C) Σ per-class accuracy.")
    W()
    W("| Run | Overall accuracy | Balanced accuracy |")
    W("|-----|:---------------:|:-----------------:|")
    for run_name in ALL_RUNS:
        per_class_accs = []
        all_overlapping_total = []
        for cn in CLASS_NAMES:
            cls_pool = []
            for fold in FOLDS:
                df = matched[run_name].get(fold)
                if df is None:
                    continue
                overlapping = df[(df["best_iou"] >= 0.3) & (df["gt_label"] == cn)]
                cls_pool.append(overlapping)
            if not cls_pool:
                continue
            pool = pd.concat(cls_pool)
            if len(pool) == 0:
                continue
            per_class_accs.append((pool["label"] == pool["gt_label"]).mean())
        # Overall accuracy
        for fold in FOLDS:
            df = matched[run_name].get(fold)
            if df is None:
                continue
            all_overlapping_total.append(df[df["best_iou"] >= 0.3])
        if not all_overlapping_total or not per_class_accs:
            continue
        overall_pool = pd.concat(all_overlapping_total)
        overall_acc = (overall_pool["label"] == overall_pool["gt_label"]).mean()
        balanced_acc = np.mean(per_class_accs)
        W(f"| {run_name} | {overall_acc*100:.1f}% | **{balanced_acc*100:.1f}%** |")
    W()

    # ── 1b. Head-to-head label agreement ─────────────────────────────────
    W("## 1b. Head-to-Head Label Comparison")
    W()
    W("Same proposals, two classifiers. When they **disagree**, which is correct?")
    W()

    for vjepa_name, mlp_name, det_name in H2H_PAIRS:
        W(f"### {det_name} proposals: {vjepa_name} vs {mlp_name}")
        W()

        agree_correct = 0
        agree_wrong = 0
        vjepa_right_mlp_wrong = 0
        mlp_right_vjepa_wrong = 0
        both_wrong_diff = 0
        n_matched_total = 0
        n_gt_overlap_total = 0

        for fold in FOLDS:
            df_v = matched[vjepa_name].get(fold)
            df_m = matched[mlp_name].get(fold)
            if df_v is None or df_m is None:
                continue

            merged = _merge_pair(df_v, df_m)
            n_matched_total += len(merged)
            if merged.empty:
                continue

            iou_col = "best_iou_vjepa" if "best_iou_vjepa" in merged.columns else "best_iou"
            gt_col = "gt_label_vjepa" if "gt_label_vjepa" in merged.columns else "gt_label"
            label_v = "label_vjepa" if "label_vjepa" in merged.columns else "label"
            label_m = "label_mlp" if "label_mlp" in merged.columns else "label"

            gt_overlap = merged[merged[iou_col] >= 0.3]
            n_gt_overlap_total += len(gt_overlap)

            for _, row in gt_overlap.iterrows():
                gt_label = row[gt_col]
                v_label = row[label_v]
                m_label = row[label_m]
                v_correct = v_label == gt_label
                m_correct = m_label == gt_label

                if v_label == m_label:
                    if v_correct:
                        agree_correct += 1
                    else:
                        agree_wrong += 1
                else:
                    if v_correct and not m_correct:
                        vjepa_right_mlp_wrong += 1
                    elif m_correct and not v_correct:
                        mlp_right_vjepa_wrong += 1
                    else:
                        both_wrong_diff += 1

        total_gt = n_gt_overlap_total
        if total_gt == 0:
            W(f"*(No GT-overlapping proposals could be matched for {det_name} — "
              f"possible video key format mismatch.)*")
            W()
            continue
        W(f"| Outcome | Count | % of GT-overlapping |")
        W(f"|---------|------:|--------------------:|")
        W(f"| **Agree, correct** | {agree_correct:,} | {agree_correct/total_gt*100:.1f}% |")
        W(f"| **Agree, wrong** | {agree_wrong:,} | {agree_wrong/total_gt*100:.1f}% |")
        W(f"| **V-JEPA2 right, MLP wrong** | {vjepa_right_mlp_wrong:,} | {vjepa_right_mlp_wrong/total_gt*100:.1f}% |")
        W(f"| **MLP right, V-JEPA2 wrong** | {mlp_right_vjepa_wrong:,} | {mlp_right_vjepa_wrong/total_gt*100:.1f}% |")
        W(f"| **Both wrong, different labels** | {both_wrong_diff:,} | {both_wrong_diff/total_gt*100:.1f}% |")
        W(f"| **Total GT-overlapping** | {total_gt:,} | 100% |")
        W(f"| Total proposals matched | {n_matched_total:,} | |")
        W()
        agree_rate = (agree_correct + agree_wrong) / total_gt if total_gt > 0 else 0
        W(f"**Label agreement rate:** {agree_rate*100:.1f}% "
          f"(agree on {agree_correct + agree_wrong:,} / {total_gt:,} GT-overlapping proposals)")
        W()
        net = mlp_right_vjepa_wrong - vjepa_right_mlp_wrong
        W(f"**Net MLP advantage:** {net:+d} proposals "
          f"(MLP uniquely correct: {mlp_right_vjepa_wrong}, "
          f"V-JEPA2 uniquely correct: {vjepa_right_mlp_wrong})")
        W()

        # Per-class disagreement detail
        W(f"#### Per-class disagreement detail ({det_name})")
        W()
        W("When the two classifiers **disagree**, what class does each pick?")
        W()
        W("| GT class | V-JEPA2 right / MLP wrong | MLP right / V-JEPA2 wrong | Net MLP |")
        W("|----------|:-------------------------:|:-------------------------:|:-------:|")
        per_class_v_right = defaultdict(int)
        per_class_m_right = defaultdict(int)
        for fold in FOLDS:
            df_v = matched[vjepa_name].get(fold)
            df_m = matched[mlp_name].get(fold)
            if df_v is None or df_m is None:
                continue
            merged = _merge_pair(df_v, df_m)
            if merged.empty:
                continue
            iou_col = "best_iou_vjepa" if "best_iou_vjepa" in merged.columns else "best_iou"
            gt_col = "gt_label_vjepa" if "gt_label_vjepa" in merged.columns else "gt_label"
            lv = "label_vjepa" if "label_vjepa" in merged.columns else "label"
            lm = "label_mlp" if "label_mlp" in merged.columns else "label"
            gt_overlap = merged[merged[iou_col] >= 0.3]
            for _, row in gt_overlap.iterrows():
                gt_label = row[gt_col]
                v_label = row[lv]
                m_label = row[lm]
                if v_label != m_label:
                    if v_label == gt_label:
                        per_class_v_right[gt_label] += 1
                    elif m_label == gt_label:
                        per_class_m_right[gt_label] += 1
        for cn in CLASS_NAMES:
            vr = per_class_v_right.get(cn, 0)
            mr = per_class_m_right.get(cn, 0)
            net_c = mr - vr
            W(f"| {cn} | {vr} | {mr} | {net_c:+d} |")
        W()

    # ── 1c. AUROC for TP vs FP ranking ───────────────────────────────────
    W("## 1c. AUROC: Does the Final Score Separate TPs from FPs?")
    W()
    W("Binary classification: is this prediction a TP (correct class + IoU ≥ 0.3)? "
      "Higher AUROC = final score better ranks TPs above FPs.")
    W()

    W("| Run | Total preds | TPs | FPs | AUROC |")
    W("|-----|------------:|----:|----:|------:|")
    for run_name in ALL_RUNS:
        all_labels = []
        all_scores = []
        for fold in FOLDS:
            df = matched[run_name].get(fold)
            if df is None:
                continue
            all_labels.extend(df["is_tp"].astype(int).tolist())
            all_scores.extend(df["score"].tolist())
        if not all_labels or sum(all_labels) == 0:
            continue
        y_true = np.array(all_labels)
        y_score = np.array(all_scores)
        auroc = roc_auc_score(y_true, y_score)
        n_tp = y_true.sum()
        n_fp = len(y_true) - n_tp
        W(f"| {run_name} | {len(y_true):,} | {n_tp:,} | {n_fp:,} | **{auroc:.4f}** |")
    W()

    # Per-class AUROC
    W("### Per-class AUROC")
    W()
    W("| Class | " + " | ".join(ALL_RUNS) + " |")
    W("|-------|" + "|".join(["------|"] * len(ALL_RUNS)))
    for cn in CLASS_NAMES:
        cells = []
        for run_name in ALL_RUNS:
            all_labels = []
            all_scores = []
            for fold in FOLDS:
                df = matched[run_name].get(fold)
                if df is None:
                    continue
                cls_preds = df[df["label"] == cn]
                is_tp = (cls_preds["is_tp"] & (cls_preds["gt_label"] == cn)).astype(int)
                all_labels.extend(is_tp.tolist())
                all_scores.extend(cls_preds["score"].tolist())
            if not all_labels or sum(all_labels) == 0:
                cells.append("—")
                continue
            try:
                auroc = roc_auc_score(np.array(all_labels), np.array(all_scores))
                cells.append(f"{auroc:.3f}")
            except ValueError:
                cells.append("—")
        W(f"| {cn} | " + " | ".join(cells) + " |")
    W()

    # ── 1d. Score distributions: TP vs FP ────────────────────────────────
    W("## 1d. Score Distributions (TP vs FP)")
    W()
    W("| Run | TP mean score | FP mean score | TP/FP ratio | TP median | FP median |")
    W("|-----|:------------:|:------------:|:-----------:|:---------:|:---------:|")
    for run_name in ALL_RUNS:
        all_tp_scores = []
        all_fp_scores = []
        for fold in FOLDS:
            df = matched[run_name].get(fold)
            if df is None:
                continue
            all_tp_scores.extend(df[df["is_tp"]]["score"].tolist())
            all_fp_scores.extend(df[~df["is_tp"]]["score"].tolist())
        if not all_tp_scores:
            continue
        tp_arr = np.array(all_tp_scores)
        fp_arr = np.array(all_fp_scores)
        ratio = tp_arr.mean() / fp_arr.mean() if fp_arr.mean() > 0 else float("inf")
        W(f"| {run_name} | {tp_arr.mean():.4f} | {fp_arr.mean():.4f} | "
          f"{ratio:.1f}× | {np.median(tp_arr):.4f} | {np.median(fp_arr):.4f} |")
    W()

    # ── 1e. Class distribution shift detail ──────────────────────────────
    W("## 1e. What Does the MLP Actually Change? (Label Redistribution)")
    W()
    W("For GT-overlapping proposals (IoU ≥ 0.3), how does each classifier label them?")
    W()

    for vjepa_name, mlp_name, det_name in H2H_PAIRS:
        W(f"### {det_name} proposals")
        W()
        W("| GT Class | n GT-overlap | V-JEPA2 assigns correct | MLP assigns correct | "
          "V-JEPA2 most common wrong | MLP most common wrong |")
        W("|----------|:-----------:|:----------------------:|:-------------------:|"
          ":------------------------:|:---------------------:|")

        for cn in CLASS_NAMES:
            v_correct = 0
            m_correct = 0
            n_gt = 0
            v_wrong_labels = defaultdict(int)
            m_wrong_labels = defaultdict(int)
            for fold in FOLDS:
                df_v = matched[vjepa_name].get(fold)
                df_m = matched[mlp_name].get(fold)
                if df_v is None or df_m is None:
                    continue
                merged = _merge_pair(df_v, df_m)
                if merged.empty:
                    continue
                iou_col = "best_iou_vjepa" if "best_iou_vjepa" in merged.columns else "best_iou"
                gt_col = "gt_label_vjepa" if "gt_label_vjepa" in merged.columns else "gt_label"
                lv = "label_vjepa" if "label_vjepa" in merged.columns else "label"
                lm = "label_mlp" if "label_mlp" in merged.columns else "label"
                gt_cls = merged[(merged[iou_col] >= 0.3) & (merged[gt_col] == cn)]
                n_gt += len(gt_cls)
                v_correct += (gt_cls[lv] == cn).sum()
                m_correct += (gt_cls[lm] == cn).sum()
                for _, row in gt_cls.iterrows():
                    if row[lv] != cn:
                        v_wrong_labels[row[lv]] += 1
                    if row[lm] != cn:
                        m_wrong_labels[row[lm]] += 1

            v_top_wrong = max(v_wrong_labels, key=v_wrong_labels.get) if v_wrong_labels else "—"
            m_top_wrong = max(m_wrong_labels, key=m_wrong_labels.get) if m_wrong_labels else "—"
            v_tw_n = v_wrong_labels.get(v_top_wrong, 0) if v_wrong_labels else 0
            m_tw_n = m_wrong_labels.get(m_top_wrong, 0) if m_wrong_labels else 0

            W(f"| {cn} | {n_gt} | {v_correct} ({v_correct/n_gt*100:.0f}%) | "
              f"{m_correct} ({m_correct/n_gt*100:.0f}%) | "
              f"{v_top_wrong} ({v_tw_n}) | {m_top_wrong} ({m_tw_n}) |")
        W()

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: ERROR TAXONOMY ACROSS ALL RUNS
    # ══════════════════════════════════════════════════════════════════════
    W("---")
    W()
    W("# Part 2: Error Taxonomy Across All Runs")
    W()
    W("Every prediction classified into one of five categories based on its "
      "best-matching GT segment.")
    W()

    # ── 2a. Aggregate error taxonomy ─────────────────────────────────────
    W("## 2a. Error Breakdown (3-fold pooled)")
    W()
    error_order = ["correct", "boundary_error", "wrong_class", "low_iou", "no_overlap"]
    error_labels = {
        "correct": "Correct (IoU≥0.5, right class)",
        "boundary_error": "Boundary (0.3≤IoU<0.5, right class)",
        "wrong_class": "Wrong class (IoU≥0.3)",
        "low_iou": "Low IoU (0<IoU<0.3)",
        "no_overlap": "No overlap (IoU=0)",
    }

    W("| Run | Total | " + " | ".join(error_labels[e] for e in error_order) + " |")
    W("|-----|------:|" + "|".join(["-----:|"] * len(error_order)))
    run_error_counts: dict[str, dict[str, int]] = {}
    for run_name in ALL_RUNS:
        counts = defaultdict(int)
        total = 0
        for fold in FOLDS:
            df = matched[run_name].get(fold)
            if df is None:
                continue
            for et in error_order:
                counts[et] += (df["error_type"] == et).sum()
            total += len(df)
        run_error_counts[run_name] = dict(counts)

        cells = []
        for et in error_order:
            n = counts[et]
            pct = n / total * 100 if total > 0 else 0
            cells.append(f"{n:,} ({pct:.1f}%)")
        W(f"| {run_name} | {total:,} | " + " | ".join(cells) + " |")
    W()

    # ── 2b. Error taxonomy: percentage-only view (cleaner) ───────────────
    W("### Percentage-only view")
    W()
    W("| Run | Correct | Boundary | Wrong class | Low IoU | No overlap |")
    W("|-----|--------:|---------:|------------:|--------:|-----------:|")
    for run_name in ALL_RUNS:
        total = sum(run_error_counts[run_name].values())
        if total == 0:
            continue
        cells = [f"{run_error_counts[run_name].get(et, 0)/total*100:.1f}%"
                 for et in error_order]
        W(f"| {run_name} | " + " | ".join(cells) + " |")
    W()

    # ── 2c. Per-class error breakdown ────────────────────────────────────
    W("## 2b. Per-Class Error Breakdown (IoU ≥ 0.3 proposals only)")
    W()
    W("Among predictions that overlap GT at IoU ≥ 0.3, what fraction are "
      "correct vs wrong class, broken down by GT class?")
    W()

    for run_name in ALL_RUNS:
        W(f"### {run_name}")
        W()
        W("| GT Class | Correct class | Wrong class | Accuracy |")
        W("|----------|:------------:|:-----------:|:--------:|")
        for cn in CLASS_NAMES:
            n_correct = 0
            n_wrong = 0
            for fold in FOLDS:
                df = matched[run_name].get(fold)
                if df is None:
                    continue
                gt_cls = df[(df["best_iou"] >= 0.3) & (df["gt_label"] == cn)]
                n_correct += (gt_cls["label"] == cn).sum()
                n_wrong += (gt_cls["label"] != cn).sum()
            total = n_correct + n_wrong
            acc = n_correct / total * 100 if total > 0 else 0
            W(f"| {cn} | {n_correct} | {n_wrong} | {acc:.1f}% |")
        W()

    # ── 2d. E2E vs Two-Stage comparison ──────────────────────────────────
    W("## 2c. E2E vs Two-Stage: Where Do the Errors Differ?")
    W()
    W("Direct comparison of ActionFormer E2E baseline against its "
      "two-stage counterparts (Runs 1 and 3c), all using ActionFormer proposals.")
    W()

    af_runs = ["AF E2E (baseline)", "AF + V-JEPA2 (Run 1)", "AF + MLP (Run 3c)"]
    W("| Metric | " + " | ".join(af_runs) + " |")
    W("|--------|" + "|".join(["-----:|"] * len(af_runs)))

    for et in error_order:
        cells = []
        for rn in af_runs:
            counts = run_error_counts.get(rn, {})
            total = sum(counts.values())
            n = counts.get(et, 0)
            pct = n / total * 100 if total > 0 else 0
            cells.append(f"{pct:.1f}%")
        W(f"| {error_labels[et]} | " + " | ".join(cells) + " |")

    # Total predictions
    cells = []
    for rn in af_runs:
        counts = run_error_counts.get(rn, {})
        total = sum(counts.values())
        cells.append(f"{total:,}")
    W(f"| **Total predictions** | " + " | ".join(cells) + " |")
    W()

    # Classification accuracy comparison
    W("### Classification accuracy: E2E vs Two-Stage (IoU ≥ 0.3)")
    W()
    W("| GT Class | " + " | ".join(af_runs) + " |")
    W("|----------|" + "|".join(["-----:|"] * len(af_runs)))
    for cn in CLASS_NAMES:
        cells = []
        for rn in af_runs:
            n_correct = 0
            n_total = 0
            for fold in FOLDS:
                df = matched[rn].get(fold)
                if df is None:
                    continue
                gt_cls = df[(df["best_iou"] >= 0.3) & (df["gt_label"] == cn)]
                n_correct += (gt_cls["label"] == cn).sum()
                n_total += len(gt_cls)
            acc = n_correct / n_total * 100 if n_total > 0 else 0
            cells.append(f"{acc:.1f}%")
        W(f"| {cn} | " + " | ".join(cells) + " |")
    W()

    # ── 2e. TriDet two-stage comparison ──────────────────────────────────
    W("### TriDet two-stage: Run 5 vs Run 5c")
    W()
    td_runs = ["TD + V-JEPA2 (Run 5)", "TD + MLP (Run 5c)"]
    W("| Metric | " + " | ".join(td_runs) + " |")
    W("|--------|" + "|".join(["-----:|"] * len(td_runs)))
    for et in error_order:
        cells = []
        for rn in td_runs:
            counts = run_error_counts.get(rn, {})
            total = sum(counts.values())
            n = counts.get(et, 0)
            pct = n / total * 100 if total > 0 else 0
            cells.append(f"{pct:.1f}%")
        W(f"| {error_labels[et]} | " + " | ".join(cells) + " |")
    cells = [f"{sum(run_error_counts.get(rn, {}).values()):,}" for rn in td_runs]
    W(f"| **Total predictions** | " + " | ".join(cells) + " |")
    W()

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    W("---")
    W()
    W("# Summary")
    W()
    W("*(Key findings from this analysis — review and interpret for paper.)*")
    W()

    out_path = TWO_STG / "analysis_classifier_effect.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written to: {out_path}")
    print(f"Total lines: {len(lines)}")


if __name__ == "__main__":
    main()
