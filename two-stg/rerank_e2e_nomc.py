#!/usr/bin/env python3
"""
Re-apply class-agnostic soft NMS to existing E2E result_detection.json files.

The original E2E runs use multiclass=True NMS, which runs NMS per class and
can emit up to max_seg_num proposals per class (4 × 200 = 800/video).
This script re-runs soft NMS ignoring class labels (class-agnostic), then
re-evaluates mAP, matching what multiclass=False would produce from OpenTAD.

Output: result_detection_nomc.json alongside each input file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


# ── Soft NMS (pure NumPy, no GPU / compiled extension needed) ────────────────

def soft_nms_1d(
    segs: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    iou_threshold: float = 0.1,
    min_score: float = 0.001,
    max_seg_num: int = 200,
) -> np.ndarray:
    """
    Gaussian soft NMS for 1D temporal segments.
    Returns array of indices (into original segs/scores) to keep, sorted by
    descending score after decay.
    """
    segs = np.array(segs, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    order = np.arange(len(segs))
    keep = []

    while len(order) > 0 and len(keep) < max_seg_num:
        best = scores[order].argmax()
        best_idx = order[best]
        keep.append(int(best_idx))

        if len(order) == 1:
            break

        rest_mask = np.arange(len(order)) != best
        rest = order[rest_mask]

        bs, be = segs[best_idx]
        rs, re = segs[rest, 0], segs[rest, 1]

        inter = np.maximum(0, np.minimum(be, re) - np.maximum(bs, rs))
        union = (be - bs) + (re - rs) - inter
        iou = inter / np.maximum(union, 1e-6)

        # Gaussian decay
        decay = np.exp(-(iou ** 2) / sigma)
        scores[rest] *= decay

        # Drop below min_score
        rest = rest[scores[rest] > min_score]
        order = rest

    return np.array(keep, dtype=np.int64)


# ── OpenTAD-style mAP evaluation (identical to analyse_classifier_effect.py) ─

ACTREG = Path("/orcd/data/satra/001/users/brukew/actreg")
ANNO_DIR = ACTREG / "OpenTAD/data/sails_rmm/annotations"
FOLDS = [0, 1, 2]
LABEL2ID = {"hands flapping": 0, "jumping": 1, "rocking": 2, "spinning": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def tiou(s1, s2) -> float:
    inter = max(0, min(s1[1], s2[1]) - max(s1[0], s2[0]))
    union = (s1[1] - s1[0]) + (s2[1] - s2[0]) - inter
    return inter / union if union > 0 else 0.0


def interpolated_ap(tp_arr, fp_arr, n_gt: int) -> float:
    """Standard interpolated AP at 11 recall points."""
    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        mask = recall >= thr
        ap += precision[mask].max() if mask.any() else 0.0
    return ap / 11.0


def eval_map(
    result_json: dict,
    anno: dict,
    tiou_thresholds: list[float],
) -> dict:
    """Compute per-threshold mAP from result_detection.json dict and fold annotation."""
    db = anno.get("database", {})
    gt_by_video: dict[str, dict[str, list]] = {}
    for vid, info in db.items():
        if info.get("subset") != "validation":
            continue
        for ann in info.get("annotations", []):
            lbl = ann.get("label", "")
            if lbl not in LABEL2ID:
                continue
            gt_by_video.setdefault(vid, {}).setdefault(lbl, []).append(ann["segment"])

    results = result_json.get("results", {})
    maps = {}
    for thr in tiou_thresholds:
        aps = []
        for cls_name in LABEL2ID:
            # Collect all predictions for this class, sorted by score desc
            preds = []
            for vid, props in results.items():
                for p in props:
                    if p["label"] == cls_name:
                        preds.append((p["score"], vid, p["segment"]))
            preds.sort(key=lambda x: -x[0])

            n_gt = sum(
                len(gt_by_video.get(vid, {}).get(cls_name, []))
                for vid in gt_by_video
            )
            if n_gt == 0:
                continue

            # Track which GT has been matched
            matched: dict[str, list[bool]] = {
                vid: [False] * len(segs)
                for vid, cls_dict in gt_by_video.items()
                for lbl, segs in cls_dict.items()
                if lbl == cls_name
            }

            tp_arr, fp_arr = [], []
            for score, vid, seg in preds:
                gt_segs = gt_by_video.get(vid, {}).get(cls_name, [])
                best_iou, best_j = 0.0, -1
                for j, gs in enumerate(gt_segs):
                    t = tiou(seg, gs)
                    if t > best_iou:
                        best_iou, best_j = t, j
                if best_iou >= thr and best_j >= 0 and not matched[vid][best_j]:
                    matched[vid][best_j] = True
                    tp_arr.append(1); fp_arr.append(0)
                else:
                    tp_arr.append(0); fp_arr.append(1)

            aps.append(interpolated_ap(np.array(tp_arr), np.array(fp_arr), n_gt))

        maps[thr] = float(np.mean(aps)) * 100 if aps else 0.0
    return maps


# ── Main ─────────────────────────────────────────────────────────────────────

NMS_PARAMS = dict(sigma=0.5, iou_threshold=0.1, min_score=0.001, max_seg_num=200)
TIOU_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def process_model(model_name: str, det_type: str, exp_prefix: str) -> None:
    """Re-NMS one model (all folds), print per-fold and average mAP."""
    print(f"\n{'='*60}")
    print(f"  {model_name}  (class-agnostic soft NMS)")
    print(f"{'='*60}")

    fold_maps: list[dict] = []
    total_preds_mc, total_preds_nomc = 0, 0

    for fold in FOLDS:
        json_path = (
            ACTREG / "OpenTAD" / "exps/sails_rmm"
            / f"{exp_prefix}_fold{fold}/gpu1_id99/result_detection.json"
        )
        anno = json.loads((ANNO_DIR / f"fold{fold}_anno.json").read_text())

        if not json_path.exists():
            print(f"  fold{fold}: MISSING {json_path}")
            continue

        data = json.loads(json_path.read_text())
        results = data.get("results", {})

        new_results: dict[str, list] = {}
        for vid, props in results.items():
            if not props:
                new_results[vid] = []
                continue

            segs = np.array([[p["segment"][0], p["segment"][1]] for p in props])
            scores = np.array([p["score"] for p in props])

            # Class-agnostic soft NMS
            keep_idx = soft_nms_1d(segs, scores, **NMS_PARAMS)

            kept = [props[i] for i in keep_idx]
            new_results[vid] = kept

        n_mc = sum(len(v) for v in results.values())
        n_nomc = sum(len(v) for v in new_results.values())
        total_preds_mc += n_mc
        total_preds_nomc += n_nomc

        # Evaluate
        new_data = {"results": new_results}
        fold_map = eval_map(new_data, anno, TIOU_THRESHOLDS)
        fold_maps.append(fold_map)

        avg_map = np.mean(list(fold_map.values()))
        print(
            f"  fold{fold}: {n_mc:,} → {n_nomc:,} proposals "
            f"| avg_mAP={avg_map:.2f}% "
            f"| @0.3={fold_map[0.3]:.2f}% @0.5={fold_map[0.5]:.2f}% @0.7={fold_map[0.7]:.2f}%"
        )

        # Save
        out_path = json_path.parent / "result_detection_nomc.json"
        out_path.write_text(json.dumps(new_data))

    if fold_maps:
        avg_across_folds = {
            thr: float(np.mean([fm[thr] for fm in fold_maps]))
            for thr in TIOU_THRESHOLDS
        }
        overall_avg = float(np.mean(list(avg_across_folds.values())))
        print(
            f"\n  3-FOLD AVERAGE: avg_mAP={overall_avg:.2f}% "
            f"| @0.3={avg_across_folds[0.3]:.2f}% "
            f"| @0.5={avg_across_folds[0.5]:.2f}% "
            f"| @0.7={avg_across_folds[0.7]:.2f}%"
        )
        print(
            f"  Total predictions: {total_preds_mc:,} (multiclass) → "
            f"{total_preds_nomc:,} (class-agnostic)"
        )


def main():
    process_model(
        "ActionFormer E2E (multiclass=True, original)",
        "actionformer",
        "actionformer_vjepa_balanced",
    )
    # Print original mAP from AF result for reference
    print("\n  [Reference: original multiclass=True mAP from RUNS.md: avg 16.70%]")

    process_model(
        "TriDet E2E (multiclass=True → class-agnostic)",
        "tridet",
        "tridet_vjepa_balanced",
    )
    print("\n  [Reference: original multiclass=True mAP from RUNS.md: avg 19.72%]")


if __name__ == "__main__":
    main()
