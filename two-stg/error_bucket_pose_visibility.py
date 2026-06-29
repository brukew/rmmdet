"""
For Run 5 predictions: assign each row to the same error taxonomy as analyze_runs_5_5c.py
and summarize pose child_presence + longest-run ratio per bucket.

Usage:
  conda activate opentad && python two-stg/error_bucket_pose_visibility.py
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
RUN5 = TWO_STG / "eval_results_tridet"
ANNO_DIR = OPENTAD / "data/sails_rmm/annotations"
FOLDS = [0, 1, 2]
ID2LABEL = {0: "hands flapping", 1: "jumping", 2: "spinning", 3: "rocking"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

sys.path.insert(0, str(TWO_STG))
from analyze_runs_5_5c import load_predictions, tiou  # noqa: E402
from analyze_pose_coverage_proposals import coverage_and_longest_run  # noqa: E402
from pose_filter_eval import build_filename_lookup, build_video_fps_lookup, load_pose_cache_lazy  # noqa: E402


def load_gt(fold: int):
    p = ANNO_DIR / f"fold{fold}_anno.json"
    if not p.exists():
        return {}
    anno = json.loads(p.read_text())
    db = anno.get("database", {})
    val_gt_detailed = {}
    for vid, info in db.items():
        if info.get("subset") != "validation":
            continue
        annots = info.get("annotations", [])
        val_gt_detailed[vid] = [
            {"label": a["label"], "segment": a["segment"]}
            for a in annots if a.get("label") in LABEL2ID
        ]
    return val_gt_detailed


def classify(row, val_gt_detailed: dict) -> str:
    vid = row["opentad_video_key"]
    gt_segs = val_gt_detailed.get(vid, [])
    if not gt_segs:
        return "no_overlap"

    pseg = [row["start_sec"], row["end_sec"]]
    best_iou = 0.0
    best_gt = None
    for gs in gt_segs:
        t = tiou(pseg, gs["segment"])
        if t > best_iou:
            best_iou = t
            best_gt = gs

    if best_iou == 0:
        return "no_overlap"
    if best_iou < 0.3:
        return "low_iou"
    if best_gt is not None:
        if row["label"] == best_gt["label"]:
            if best_iou >= 0.5:
                return "correct"
            return "boundary"
        return "wrong_class"
    return "no_overlap"


def main():
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for fold in FOLDS:
        val_gt = load_gt(fold)
        preds = load_predictions(RUN5, fold)
        if preds is None or preds.empty:
            continue
        if "label" not in preds.columns and "class_id" in preds.columns:
            preds = preds.copy()
            preds["label"] = preds["class_id"].map(ID2LABEL)
        fps_lu = build_video_fps_lookup(fold)
        fn_lu = build_filename_lookup(fold)

        for _, row in preds.iterrows():
            bucket = classify(row, val_gt)
            vk = row["video_key"]
            fps = fps_lu.get(vk, 30.0)
            stem = fn_lu.get(vk, Path(vk).stem)
            pose = load_pose_cache_lazy(stem)
            stats = coverage_and_longest_run(pose, row["start_sec"], row["end_sec"], fps)
            buckets[bucket].append((stats["child_presence"], stats["longest_run_ratio"]))

    order = [
        "correct",
        "boundary",
        "wrong_class",
        "low_iou",
        "no_overlap",
    ]
    labels = {
        "correct": "Correct (class OK, IoU≥0.5)",
        "boundary": "Boundary error (class OK, 0.3≤IoU<0.5)",
        "wrong_class": "Wrong class (IoU≥0.3)",
        "low_iou": "Low IoU (0<IoU<0.3)",
        "no_overlap": "No overlap (IoU=0 or no GT video)",
    }

    print("# Child visibility by error bucket (Run 5, all folds)\n")
    print(
        "| Bucket | n | mean presence | median | p25 | p75 | frac =0 | frac <0.5 | mean longest-run ratio |"
    )
    print("|--------|---:|--------------:|-------:|----:|----:|--------:|----------:|------------------------:|")

    for b in order:
        rows = buckets[b]
        if not rows:
            continue
        pres = np.array([r[0] for r in rows])
        lr = np.array([r[1] for r in rows])
        print(
            f"| {labels[b]} | {len(rows)} | {pres.mean():.3f} | {np.median(pres):.3f} | "
            f"{np.percentile(pres,25):.3f} | {np.percentile(pres,75):.3f} | "
            f"{(pres == 0).mean()*100:.1f}% | {(pres < 0.5).mean()*100:.1f}% | {lr.mean():.3f} |"
        )


if __name__ == "__main__":
    main()
