#!/usr/bin/env python3
"""
Deep analysis of Run 5 (TriDet + V-JEPA2 4-class) and Run 5c (TriDet + 3-way live fusion)
compared to Run 1 (ActionFormer + V-JEPA2 4-class).

Outputs: analysis_runs_5_5c.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ACTREG = Path("/orcd/data/satra/001/users/brukew/actreg")
TWO_STG = ACTREG / "two-stg"
OPENTAD = ACTREG / "OpenTAD"

RUNS = {
    "Run 1 (AF+VJEPA)": TWO_STG / "eval_results",
    "Run 5 (TD+VJEPA)": TWO_STG / "eval_results_tridet",
    "Run 5c (TD+3way)": TWO_STG / "eval_results_tridet_3way_live",
}

DETECTION_JSONS = {
    "ActionFormer": OPENTAD / "exps/sails_rmm/actionformer_vjepa_binary_fold{fold}/gpu1_id99/result_detection.json",
    "TriDet": OPENTAD / "exps/sails_rmm/tridet_vjepa_binary_fold{fold}/gpu1_id99/result_detection.json",
}

ANNO_DIR = OPENTAD / "data/sails_rmm/annotations"
CLASS_NAMES = ["hands flapping", "jumping", "rocking", "spinning"]
TIOU_THRESHOLDS = [0.3, 0.5, 0.7]
FOLDS = [0, 1, 2]

ID2LABEL = {0: "hands flapping", 1: "jumping", 2: "rocking", 3: "spinning"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def load_metrics_opentad(run_dir, fold):
    p = run_dir / f"fold{fold}" / "metrics_opentad.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def load_metrics_custom(run_dir, fold):
    p = run_dir / f"fold{fold}" / "metrics_custom.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def load_predictions(run_dir, fold):
    p = run_dir / f"fold{fold}" / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "label" not in df.columns and "class_id" in df.columns:
        df["label"] = df["class_id"].map(ID2LABEL)
    if "opentad_video_key" not in df.columns and "video_key" in df.columns:
        df["opentad_video_key"] = df["video_key"]
    return df


def load_detection_json(path):
    if path.exists():
        data = json.loads(path.read_text())
        return data.get("results", {})
    return None


def load_annotation(fold):
    p = ANNO_DIR / f"fold{fold}_anno.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def tiou(seg1, seg2):
    """Temporal IoU between two segments [start, end]."""
    inter_start = max(seg1[0], seg2[0])
    inter_end = min(seg1[1], seg2[1])
    inter = max(0, inter_end - inter_start)
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
    if union <= 0:
        return 0.0
    return inter / union


def main():
    lines = []
    def W(s=""):
        lines.append(s)

    W("# Deep Analysis: Runs 5 & 5c (TriDet Two-Stage TAL)")
    W()
    W("## 1. Per-Class Average Precision")
    W()
    W("### 1a. Per-class AP at key tIoU thresholds (3-fold CV mean)")
    W()

    # Collect per-class AP across folds for each run
    per_class_data = {}
    for run_name, run_dir in RUNS.items():
        per_class_data[run_name] = {}
        for tiou_thr in TIOU_THRESHOLDS:
            thr_key = f"tIoU={tiou_thr}"
            class_aps = {cn: [] for cn in CLASS_NAMES}
            for fold in FOLDS:
                m = load_metrics_opentad(run_dir, fold)
                if m and "per_class_ap" in m and thr_key in m["per_class_ap"]:
                    for cn in CLASS_NAMES:
                        v = m["per_class_ap"][thr_key].get(cn, 0)
                        class_aps[cn].append(v if v is not None else 0)
            per_class_data[run_name][thr_key] = {
                cn: (np.mean(vals) if vals else 0, np.std(vals) if vals else 0)
                for cn, vals in class_aps.items()
            }

    for tiou_thr in TIOU_THRESHOLDS:
        thr_key = f"tIoU={tiou_thr}"
        W(f"**{thr_key}:**")
        W()
        header = "| Class | " + " | ".join(RUNS.keys()) + " |"
        sep = "|-------|" + "|".join(["---------|"] * len(RUNS))
        W(header)
        W(sep)
        for cn in CLASS_NAMES:
            cells = []
            for run_name in RUNS:
                mean, std = per_class_data[run_name][thr_key][cn]
                cells.append(f"{mean*100:.1f}% ± {std*100:.1f}%")
            W(f"| {cn} | " + " | ".join(cells) + " |")
        W()

    # 1b: Class-level deltas (Run 5 vs Run 1)
    W("### 1b. TriDet vs ActionFormer class-level delta (Run 5 − Run 1)")
    W()
    W("| Class | Δ mAP@0.3 | Δ mAP@0.5 | Δ mAP@0.7 |")
    W("|-------|----------|----------|----------|")
    for cn in CLASS_NAMES:
        deltas = []
        for tiou_thr in TIOU_THRESHOLDS:
            thr_key = f"tIoU={tiou_thr}"
            td_mean = per_class_data["Run 5 (TD+VJEPA)"][thr_key][cn][0]
            af_mean = per_class_data["Run 1 (AF+VJEPA)"][thr_key][cn][0]
            deltas.append(f"{(td_mean - af_mean)*100:+.1f} pp")
        W(f"| {cn} | " + " | ".join(deltas) + " |")
    W()

    # 2. Per-video analysis
    W("---")
    W()
    W("## 2. Per-Video Analysis")
    W()
    W("### 2a. Per-video proposal counts, predictions, and GT segments")
    W()

    anno_cache = {}
    for fold in FOLDS:
        anno = load_annotation(fold)
        if anno:
            anno_cache[fold] = anno

    # Build GT info per video per fold
    gt_info = {}
    for fold in FOLDS:
        anno = anno_cache.get(fold)
        if not anno:
            continue
        db = anno.get("database", {})
        for vid, info in db.items():
            if info.get("subset") != "validation":
                continue
            annots = info.get("annotations", [])
            gt_segs = []
            for a in annots:
                label = a.get("label", "")
                seg = a.get("segment", [0, 0])
                if label in LABEL2ID:
                    gt_segs.append({"label": label, "segment": seg})
            gt_info.setdefault(fold, {})[vid] = {
                "n_gt": len(gt_segs),
                "gt_segments": gt_segs,
                "duration": info.get("duration", 0),
                "gt_classes": list(set(a["label"] for a in gt_segs)),
            }

    # Per-video predictions analysis
    video_stats_all = []
    for run_name, run_dir in RUNS.items():
        for fold in FOLDS:
            preds = load_predictions(run_dir, fold)
            if preds is None or preds.empty:
                continue
            vid_groups = preds.groupby("opentad_video_key")
            gt_fold = gt_info.get(fold, {})
            for vid, grp in vid_groups:
                n_preds = len(grp)
                top_score = grp["score"].max()
                mean_score = grp["score"].mean()
                pred_classes = grp["label"].value_counts().to_dict()
                dominant_class = grp["label"].value_counts().idxmax() if n_preds > 0 else ""
                gt_v = gt_fold.get(vid, {})
                n_gt = gt_v.get("n_gt", 0)

                # Compute per-video "hits" at tIoU=0.3 and 0.5
                gt_segments = gt_v.get("gt_segments", [])
                hits_03 = 0
                hits_05 = 0
                for gt_seg in gt_segments:
                    best_tiou = 0
                    best_match_label = ""
                    for _, row in grp.iterrows():
                        t = tiou([row["start_sec"], row["end_sec"]], gt_seg["segment"])
                        if t > best_tiou:
                            best_tiou = t
                            best_match_label = row["label"]
                    if best_tiou >= 0.3:
                        hits_03 += 1
                    if best_tiou >= 0.5:
                        hits_05 += 1

                video_stats_all.append({
                    "run": run_name,
                    "fold": fold,
                    "video": vid,
                    "n_preds": n_preds,
                    "n_gt": n_gt,
                    "top_score": top_score,
                    "mean_score": mean_score,
                    "dominant_class": dominant_class,
                    "gt_classes": ", ".join(gt_v.get("gt_classes", [])),
                    "recall_03": hits_03 / n_gt if n_gt > 0 else 0,
                    "recall_05": hits_05 / n_gt if n_gt > 0 else 0,
                    "duration": gt_v.get("duration", 0),
                    "pred_to_gt_ratio": n_preds / n_gt if n_gt > 0 else n_preds,
                })

    vdf = pd.DataFrame(video_stats_all)

    # 2a: Per-video aggregate stats
    for run_name in RUNS:
        run_vdf = vdf[(vdf["run"] == run_name) & (vdf["n_gt"] > 0)]
        if run_vdf.empty:
            W(f"**{run_name}:** no per-video data available")
            W()
            continue
        W(f"**{run_name}** ({len(run_vdf)} videos across 3 folds):")
        W()
        W(f"| Metric | Mean | Median | Min | Max |")
        W(f"|--------|------|--------|-----|-----|")
        for col, label in [
            ("n_gt", "GT segments/video"),
            ("n_preds", "Predictions/video"),
            ("pred_to_gt_ratio", "Pred:GT ratio"),
            ("recall_03", "Recall@0.3"),
            ("recall_05", "Recall@0.5"),
            ("top_score", "Top score"),
            ("duration", "Duration (sec)"),
        ]:
            vals = run_vdf[col].values
            if col in ("recall_03", "recall_05"):
                W(f"| {label} | {np.mean(vals)*100:.1f}% | {np.median(vals)*100:.1f}% | "
                  f"{np.min(vals)*100:.0f}% | {np.max(vals)*100:.0f}% |")
            elif col == "duration":
                W(f"| {label} | {np.mean(vals):.0f} | {np.median(vals):.0f} | "
                  f"{np.min(vals):.0f} | {np.max(vals):.0f} |")
            elif col == "pred_to_gt_ratio":
                W(f"| {label} | {np.mean(vals):.1f} | {np.median(vals):.1f} | "
                  f"{np.min(vals):.1f} | {np.max(vals):.1f} |")
            else:
                W(f"| {label} | {np.mean(vals):.1f} | {np.median(vals):.1f} | "
                  f"{np.min(vals):.0f} | {np.max(vals):.0f} |")
        W()

        # Per-class recall breakdown
        W(f"Per-class video-level recall@0.3 (mean across videos with that GT class):")
        W()
        for cn in CLASS_NAMES:
            cls_vids = run_vdf[run_vdf["gt_classes"].str.contains(cn, na=False)]
            if len(cls_vids) > 0:
                W(f"- **{cn}:** {cls_vids['recall_03'].mean()*100:.1f}% "
                  f"(n={len(cls_vids)} videos)")
        W()

    # Videos with 0% recall — completely missed
    W("### 2b. Videos with 0% recall at tIoU=0.3 (completely missed GT)")
    W()
    for run_name in RUNS:
        run_vdf = vdf[(vdf["run"] == run_name) & (vdf["n_gt"] > 0)]
        missed = run_vdf[run_vdf["recall_03"] == 0]
        W(f"**{run_name}:** {len(missed)} / {len(run_vdf)} videos with 0% recall at tIoU=0.3")
        if len(missed) > 0:
            for _, row in missed.iterrows():
                W(f"  - `{row['video']}` (fold {row['fold']}, {row['n_gt']} GT segs, "
                  f"classes: {row['gt_classes']}, {row['n_preds']} preds)")
    W()

    # Hardest videos (lowest recall) — shared across runs
    W("### 2c. Hardest videos (lowest average recall@0.3 across Run 5 & Run 1)")
    W()
    if not vdf.empty:
        pivot = vdf[vdf["n_gt"] > 0].pivot_table(
            index=["video", "fold"], columns="run", values="recall_03", aggfunc="first"
        ).reset_index()
        run_cols = [c for c in pivot.columns if c not in ["video", "fold"]]
        pivot["avg_recall"] = pivot[run_cols].mean(axis=1)
        pivot = pivot.sort_values("avg_recall").head(15)
        W("| Video | Fold | n_GT | " + " | ".join(run_cols) + " | Avg |")
        W("|-------|------|------|" + "|".join(["------|"] * len(run_cols)) + "------|")
        for _, row in pivot.iterrows():
            vid = row["video"]
            fold = int(row["fold"])
            n_gt = gt_info.get(fold, {}).get(vid, {}).get("n_gt", "?")
            cells = [f"{row.get(c, 0)*100:.0f}%" if pd.notna(row.get(c)) else "—" for c in run_cols]
            W(f"| `{vid}` | {fold} | {n_gt} | " + " | ".join(cells) + f" | {row['avg_recall']*100:.0f}% |")
        W()

    # Easiest videos
    W("### 2d. Easiest videos (highest average recall@0.3)")
    W()
    if not vdf.empty:
        pivot2 = vdf[vdf["n_gt"] > 0].pivot_table(
            index=["video", "fold"], columns="run", values="recall_03", aggfunc="first"
        ).reset_index()
        run_cols2 = [c for c in pivot2.columns if c not in ["video", "fold"]]
        pivot2["avg_recall"] = pivot2[run_cols2].mean(axis=1)
        pivot2 = pivot2.sort_values("avg_recall", ascending=False).head(10)
        W("| Video | Fold | n_GT | Avg Recall@0.3 |")
        W("|-------|------|------|----------------|")
        for _, row in pivot2.iterrows():
            vid = row["video"]
            fold = int(row["fold"])
            n_gt = gt_info.get(fold, {}).get(vid, {}).get("n_gt", "?")
            W(f"| `{vid}` | {fold} | {n_gt} | {row['avg_recall']*100:.0f}% |")
        W()

    # 3. Proposal Quality Comparison (TriDet vs ActionFormer)
    W("---")
    W()
    W("## 3. Stage 1 Proposal Quality: TriDet vs ActionFormer")
    W()

    for fold in FOLDS:
        W(f"### Fold {fold}")
        W()
        for det_name, det_template in DETECTION_JSONS.items():
            det_path = Path(str(det_template).format(fold=fold))
            results = load_detection_json(det_path)
            if results is None:
                W(f"**{det_name}:** file not found")
                continue

            all_scores = []
            all_durations = []
            n_proposals_per_video = []
            for vid, props in results.items():
                n_proposals_per_video.append(len(props))
                for p in props:
                    all_scores.append(p["score"])
                    dur = p["segment"][1] - p["segment"][0]
                    all_durations.append(dur)

            scores = np.array(all_scores)
            durs = np.array(all_durations)

            W(f"**{det_name}:**")
            W(f"- Total proposals: {len(scores)}")
            W(f"- Videos with proposals: {len(n_proposals_per_video)}")
            W(f"- Proposals/video: mean={np.mean(n_proposals_per_video):.0f}, "
              f"median={np.median(n_proposals_per_video):.0f}, "
              f"max={np.max(n_proposals_per_video)}")
            W(f"- Score distribution: mean={scores.mean():.4f}, "
              f"median={np.median(scores):.4f}, "
              f"p95={np.percentile(scores, 95):.4f}, max={scores.max():.4f}")
            W(f"- Duration (sec): mean={durs.mean():.1f}, "
              f"median={np.median(durs):.1f}, "
              f"min={durs.min():.1f}, max={durs.max():.1f}")
            W(f"- Score > 0.1: {(scores > 0.1).sum()} ({(scores > 0.1).mean()*100:.1f}%)")
            W(f"- Score > 0.3: {(scores > 0.3).sum()} ({(scores > 0.3).mean()*100:.1f}%)")
            W(f"- Score > 0.5: {(scores > 0.5).sum()} ({(scores > 0.5).mean()*100:.1f}%)")
            W()

    # 4. Proposal-level IoU with GT
    W("---")
    W()
    W("## 4. Proposal-GT IoU Distribution (Stage 1)")
    W()

    for fold in FOLDS:
        anno = anno_cache.get(fold)
        if not anno:
            continue
        db = anno.get("database", {})
        val_gt = {}
        for vid, info in db.items():
            if info.get("subset") != "validation":
                continue
            annots = info.get("annotations", [])
            gt_segs = [a["segment"] for a in annots if a.get("label") in LABEL2ID]
            if gt_segs:
                val_gt[vid] = gt_segs

        W(f"### Fold {fold}")
        W()
        for det_name, det_template in DETECTION_JSONS.items():
            det_path = Path(str(det_template).format(fold=fold))
            results = load_detection_json(det_path)
            if results is None:
                continue

            best_ious = []
            for vid, props in results.items():
                gt_segs = val_gt.get(vid, [])
                if not gt_segs:
                    continue
                for p in props:
                    pseg = p["segment"]
                    best = max(tiou(pseg, gs) for gs in gt_segs)
                    best_ious.append(best)

            ious = np.array(best_ious)
            W(f"**{det_name}:** {len(ious)} proposals matched to GT")
            W(f"- Mean best-IoU: {ious.mean():.3f}")
            W(f"- Median best-IoU: {np.median(ious):.3f}")
            W(f"- IoU > 0.3: {(ious > 0.3).sum()} ({(ious > 0.3).mean()*100:.1f}%)")
            W(f"- IoU > 0.5: {(ious > 0.5).sum()} ({(ious > 0.5).mean()*100:.1f}%)")
            W(f"- IoU > 0.7: {(ious > 0.7).sum()} ({(ious > 0.7).mean()*100:.1f}%)")
            W(f"- IoU = 0.0 (no overlap): {(ious == 0).sum()} ({(ious == 0).mean()*100:.1f}%)")
            W()

    # 5. Error Taxonomy
    W("---")
    W()
    W("## 5. Error Taxonomy (Run 5 — TriDet + V-JEPA2 4-class)")
    W()
    W("Classification of prediction errors into categories.")
    W()

    for fold in FOLDS:
        anno = anno_cache.get(fold)
        if not anno:
            continue
        db = anno.get("database", {})

        preds = load_predictions(RUNS["Run 5 (TD+VJEPA)"], fold)
        if preds is None or preds.empty:
            continue

        val_gt_detailed = {}
        for vid, info in db.items():
            if info.get("subset") != "validation":
                continue
            annots = info.get("annotations", [])
            val_gt_detailed[vid] = [
                {"label": a["label"], "segment": a["segment"]}
                for a in annots if a.get("label") in LABEL2ID
            ]

        n_correct_cls = 0
        n_wrong_cls = 0
        n_no_overlap = 0
        n_low_iou = 0  # 0 < IoU < 0.3
        n_boundary_error = 0  # IoU 0.3-0.5, correct class
        confusion = defaultdict(lambda: defaultdict(int))

        for _, row in preds.iterrows():
            vid = row["opentad_video_key"]
            gt_segs = val_gt_detailed.get(vid, [])
            if not gt_segs:
                n_no_overlap += 1
                continue

            pseg = [row["start_sec"], row["end_sec"]]
            best_iou = 0
            best_gt = None
            for gs in gt_segs:
                t = tiou(pseg, gs["segment"])
                if t > best_iou:
                    best_iou = t
                    best_gt = gs

            if best_iou == 0:
                n_no_overlap += 1
            elif best_iou < 0.3:
                n_low_iou += 1
            elif best_gt is not None:
                if row["label"] == best_gt["label"]:
                    if best_iou >= 0.5:
                        n_correct_cls += 1
                    else:
                        n_boundary_error += 1
                else:
                    n_wrong_cls += 1
                    confusion[best_gt["label"]][row["label"]] += 1

        total = len(preds)
        W(f"### Fold {fold} ({total} total predictions)")
        W()
        W(f"| Error Type | Count | % |")
        W(f"|-----------|-------|---|")
        W(f"| Correct class + IoU≥0.5 | {n_correct_cls} | {n_correct_cls/total*100:.1f}% |")
        W(f"| Correct class + boundary error (0.3≤IoU<0.5) | {n_boundary_error} | {n_boundary_error/total*100:.1f}% |")
        W(f"| Wrong class (IoU≥0.3) | {n_wrong_cls} | {n_wrong_cls/total*100:.1f}% |")
        W(f"| Low IoU (0<IoU<0.3) | {n_low_iou} | {n_low_iou/total*100:.1f}% |")
        W(f"| No overlap (IoU=0) | {n_no_overlap} | {n_no_overlap/total*100:.1f}% |")
        W()

        if confusion:
            W(f"**Class confusion matrix (GT → Predicted, IoU≥0.3):**")
            W()
            W("| GT \\ Pred | " + " | ".join(CLASS_NAMES) + " |")
            W("|-----------|" + "|".join(["---------|"] * len(CLASS_NAMES)))
            for gt_cls in CLASS_NAMES:
                cells = []
                for pred_cls in CLASS_NAMES:
                    v = confusion[gt_cls][pred_cls]
                    cells.append(str(v) if v > 0 else "·")
                W(f"| {gt_cls} | " + " | ".join(cells) + " |")
            W()

    # 6. Fold variance analysis
    W("---")
    W()
    W("## 6. Fold Variance Analysis")
    W()
    W("### 6a. Per-fold mAP breakdown")
    W()

    W("| Run | Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |")
    W("|-----|------|---------|---------|---------|---------|")
    for run_name, run_dir in RUNS.items():
        for fold in FOLDS:
            m = load_metrics_opentad(run_dir, fold)
            if m:
                W(f"| {run_name} | {fold} | {m.get('mAP@0.3', 0)*100:.1f}% | "
                  f"{m.get('mAP@0.5', 0)*100:.1f}% | {m.get('mAP@0.7', 0)*100:.1f}% | "
                  f"{m.get('avg_mAP', 0)*100:.1f}% |")
    W()

    # 6b: GT distribution per fold
    W("### 6b. GT segment distribution per fold")
    W()
    W("| Fold | Total GT | " + " | ".join(CLASS_NAMES) + " | # Videos |")
    W("|------|----------|" + "|".join(["---------|"] * len(CLASS_NAMES)) + "----------|")
    for fold in FOLDS:
        anno = anno_cache.get(fold)
        if not anno:
            continue
        db = anno.get("database", {})
        class_counts = defaultdict(int)
        total = 0
        n_videos = 0
        for vid, info in db.items():
            if info.get("subset") != "validation":
                continue
            n_videos += 1
            for a in info.get("annotations", []):
                if a.get("label") in LABEL2ID:
                    class_counts[a["label"]] += 1
                    total += 1
        cells = [str(class_counts.get(cn, 0)) for cn in CLASS_NAMES]
        W(f"| {fold} | {total} | " + " | ".join(cells) + f" | {n_videos} |")
    W()

    # 7. Score distribution analysis
    W("---")
    W()
    W("## 7. Final Score Distribution (after Stage 2)")
    W()
    for run_name, run_dir in RUNS.items():
        W(f"### {run_name}")
        W()
        for fold in FOLDS:
            preds = load_predictions(run_dir, fold)
            if preds is None or preds.empty:
                continue
            scores = preds["score"].values
            W(f"**Fold {fold}:** {len(scores)} predictions")
            W(f"- Score: mean={scores.mean():.4f}, median={np.median(scores):.4f}, "
              f"max={scores.max():.4f}, p95={np.percentile(scores, 95):.4f}")
            for cls_name in CLASS_NAMES:
                cls_scores = preds[preds["label"] == cls_name]["score"].values
                if len(cls_scores) > 0:
                    W(f"  - {cls_name}: n={len(cls_scores)}, "
                      f"mean={cls_scores.mean():.4f}, max={cls_scores.max():.4f}")
            W()

    # 8. Temporal boundary analysis
    W("---")
    W()
    W("## 8. Temporal Boundary Analysis (TriDet vs ActionFormer)")
    W()
    W("For TP predictions at tIoU≥0.3, what is the average IoU? (higher = better boundaries)")
    W()

    for fold in FOLDS:
        anno = anno_cache.get(fold)
        if not anno:
            continue
        db = anno.get("database", {})
        val_gt_segs = {}
        for vid, info in db.items():
            if info.get("subset") != "validation":
                continue
            val_gt_segs[vid] = [
                {"label": a["label"], "segment": a["segment"]}
                for a in info.get("annotations", [])
                if a.get("label") in LABEL2ID
            ]

        W(f"### Fold {fold}")
        W()
        for run_name in ["Run 5 (TD+VJEPA)", "Run 1 (AF+VJEPA)"]:
            run_dir = RUNS[run_name]
            preds = load_predictions(run_dir, fold)
            if preds is None or preds.empty:
                continue

            tp_ious = []
            for _, row in preds.iterrows():
                vid = row["opentad_video_key"]
                gt_segs = val_gt_segs.get(vid, [])
                if not gt_segs:
                    continue
                pseg = [row["start_sec"], row["end_sec"]]
                best_iou = 0
                best_gt = None
                for gs in gt_segs:
                    t = tiou(pseg, gs["segment"])
                    if t > best_iou:
                        best_iou = t
                        best_gt = gs
                if best_iou >= 0.3 and best_gt and row["label"] == best_gt["label"]:
                    tp_ious.append(best_iou)

            if tp_ious:
                arr = np.array(tp_ious)
                W(f"**{run_name}:** {len(tp_ious)} TPs, mean IoU={arr.mean():.3f}, "
                  f"median={np.median(arr):.3f}, p25={np.percentile(arr, 25):.3f}")
            else:
                W(f"**{run_name}:** no TPs found")
        W()

    # 9. Detection-level detail table
    W("---")
    W()
    W("## 9. Recall & Precision Summary (from custom metrics)")
    W()
    W("### Per-class recall at tIoU=0.3 (3-fold mean)")
    W()

    recall_data = {}
    precision_data = {}
    for run_name, run_dir in RUNS.items():
        recall_data[run_name] = {cn: [] for cn in CLASS_NAMES}
        precision_data[run_name] = {cn: [] for cn in CLASS_NAMES}
        for fold in FOLDS:
            m = load_metrics_custom(run_dir, fold)
            if not m or "details" not in m:
                continue
            for thr_key in ["tIoU=0.3"]:
                details = m["details"].get(thr_key, {})
                for cls_id_str, cls_detail in details.items():
                    cls_id = int(cls_id_str)
                    cn = ID2LABEL.get(cls_id, f"class_{cls_id}")
                    if cn in recall_data[run_name]:
                        recall_data[run_name][cn].append(cls_detail.get("recall_at_threshold", 0))
                        precision_data[run_name][cn].append(cls_detail.get("precision_at_threshold", 0))

    W("| Class | " + " | ".join(f"{rn} recall" for rn in RUNS) + " |")
    W("|-------|" + "|".join(["---------|"] * len(RUNS)))
    for cn in CLASS_NAMES:
        cells = []
        for rn in RUNS:
            vals = recall_data[rn].get(cn, [])
            if vals:
                cells.append(f"{np.mean(vals)*100:.1f}%")
            else:
                cells.append("—")
        W(f"| {cn} | " + " | ".join(cells) + " |")
    W()

    # Summary
    W("---")
    W()
    W("## 10. Key Findings & Recommendations")
    W()
    W("*(Auto-generated summary — review and refine for paper.)*")
    W()

    out_path = TWO_STG / "analysis_runs_5_5c.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Analysis written to: {out_path}")
    print(f"Total lines: {len(lines)}")


if __name__ == "__main__":
    main()
