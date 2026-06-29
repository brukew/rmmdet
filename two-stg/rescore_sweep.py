"""
Re-evaluate two-stage predictions with different score combination formulas.

Recovers det_score from the original detection JSON and cls_conf from
the predictions CSV (cls_conf = combined_score / det_score), then
recomputes the final score with various alpha values:

  additive:       score = alpha * det_score + (1 - alpha) * cls_conf
  multiplicative: score = det_score * cls_conf  (original baseline)

Usage:
  python two-stg/rescore_sweep.py --run 5 --folds 0 1 2
  python two-stg/rescore_sweep.py --run 5c --folds 0 1 2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ACTREG = Path("/orcd/data/satra/001/users/brukew/actreg")

RUN_CONFIG = {
    "5": {
        "pred_dir": ACTREG / "two-stg/eval_results_tridet",
        "det_json_template": str(
            ACTREG / "OpenTAD/exps/sails_rmm/tridet_vjepa_binary_fold{fold}/gpu1_id99/result_detection.json"
        ),
    },
    "5c": {
        "pred_dir": ACTREG / "two-stg/eval_results_tridet_3way_live",
        "det_json_template": str(
            ACTREG / "OpenTAD/exps/sails_rmm/tridet_vjepa_binary_fold{fold}/gpu1_id99/result_detection.json"
        ),
    },
    "1": {
        "pred_dir": ACTREG / "two-stg/eval_results",
        "det_json_template": str(
            ACTREG / "OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold{fold}/gpu1_id99/result_detection.json"
        ),
    },
    "3c": {
        "pred_dir": ACTREG / "two-stg/eval_results_3way_live",
        "det_json_template": str(
            ACTREG / "OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold{fold}/gpu1_id99/result_detection.json"
        ),
    },
}

ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]


def load_det_scores(det_json_path):
    """Build lookup: (video_key, start, end) -> det_score."""
    with open(det_json_path) as f:
        data = json.load(f)
    lookup = {}
    for video_key, proposals in data["results"].items():
        for p in proposals:
            seg = tuple(round(x, 4) for x in p["segment"])
            lookup[(video_key, seg[0], seg[1])] = p["score"]
    return lookup


def recover_components(pred_df, det_lookup):
    """Add det_score and cls_conf columns to predictions DataFrame."""
    det_scores = []
    cls_confs = []
    matched = 0
    for _, row in pred_df.iterrows():
        key = (row["opentad_video_key"], round(row["start_sec"], 4), round(row["end_sec"], 4))
        det_score = det_lookup.get(key)
        if det_score is None:
            key2 = (row["opentad_video_key"], round(row["start_sec"], 2), round(row["end_sec"], 2))
            for lk, lv in det_lookup.items():
                if lk[0] == key2[0] and abs(lk[1] - key2[1]) < 0.02 and abs(lk[2] - key2[2]) < 0.02:
                    det_score = lv
                    break
        if det_score is not None and det_score > 0:
            cls_conf = row["score"] / det_score
            matched += 1
        else:
            det_score = row["score"]
            cls_conf = 1.0
        det_scores.append(det_score)
        cls_confs.append(cls_conf)

    pred_df = pred_df.copy()
    pred_df["det_score"] = det_scores
    pred_df["cls_conf"] = cls_confs
    print(f"  Matched {matched}/{len(pred_df)} proposals to detection JSON")
    return pred_df


def rescore(pred_df, alpha=None, mode="additive"):
    """Recompute score column with given formula."""
    df = pred_df.copy()
    if mode == "multiplicative":
        df["score"] = df["det_score"] * df["cls_conf"]
    elif mode == "additive":
        df["score"] = alpha * df["det_score"] + (1 - alpha) * df["cls_conf"]
    elif mode == "geometric":
        df["score"] = (df["det_score"] ** alpha) * (df["cls_conf"] ** (1 - alpha))
    elif mode == "cls_only":
        df["score"] = df["cls_conf"]
    elif mode == "det_only":
        df["score"] = df["det_score"]
    return df


def run_opentad_eval(pred_df, fold):
    """Run OpenTAD evaluation and return metrics dict."""
    sys.path.insert(0, str(ACTREG / "two-stg"))
    from opentad_eval import prediction_df_to_opentad_results, evaluate_with_opentad

    results_dict = prediction_df_to_opentad_results(pred_df)
    metrics = evaluate_with_opentad(results_dict, fold=fold)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=list(RUN_CONFIG.keys()))
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    cfg = RUN_CONFIG[args.run]
    print(f"=== Score Sweep for Run {args.run} ===\n")

    all_results = {}

    for fold in args.folds:
        print(f"\n--- Fold {fold} ---")
        pred_csv = cfg["pred_dir"] / f"fold{fold}" / "predictions.csv"
        det_json = cfg["det_json_template"].format(fold=fold)

        if not pred_csv.exists():
            print(f"  SKIP: {pred_csv} not found")
            continue

        pred_df = pd.read_csv(pred_csv)
        det_lookup = load_det_scores(det_json)
        pred_df = recover_components(pred_df, det_lookup)

        fold_results = {}

        # Baseline: multiplicative (current)
        df_mult = rescore(pred_df, mode="multiplicative")
        m = run_opentad_eval(df_mult, fold)
        fold_results["mult (baseline)"] = m
        print(f"  mult (baseline):  avg_mAP={m['average_mAP']*100:.2f}%  @0.3={m['mAP@0.3']*100:.2f}  @0.5={m['mAP@0.5']*100:.2f}  @0.7={m['mAP@0.7']*100:.2f}")

        # Alpha sweep (geometric): det^α * cls^(1-α)
        for alpha in ALPHAS:
            label = f"geo α={alpha}"
            df_g = rescore(pred_df, alpha=alpha, mode="geometric")
            m = run_opentad_eval(df_g, fold)
            fold_results[label] = m
            print(f"  {label}:  avg_mAP={m['average_mAP']*100:.2f}%  @0.3={m['mAP@0.3']*100:.2f}  @0.5={m['mAP@0.5']*100:.2f}  @0.7={m['mAP@0.7']*100:.2f}")

        all_results[fold] = fold_results

    # Cross-fold summary
    print("\n\n=== Cross-Fold Summary (Run {}) ===".format(args.run))
    formulas = ["mult (baseline)"] + [f"geo α={a}" for a in ALPHAS]
    print(f"{'Formula':<20} {'avg_mAP':>8} {'@0.3':>8} {'@0.5':>8} {'@0.7':>8}")
    print("-" * 60)

    for formula in formulas:
        fold_metrics = [all_results[f][formula] for f in args.folds if f in all_results and formula in all_results[f]]
        if not fold_metrics:
            continue
        avg = lambda key: np.mean([m[key] for m in fold_metrics]) * 100
        std = lambda key: np.std([m[key] for m in fold_metrics]) * 100
        print(f"{formula:<20} {avg('average_mAP'):>7.2f}% {avg('mAP@0.3'):>7.2f}% {avg('mAP@0.5'):>7.2f}% {avg('mAP@0.7'):>7.2f}%")


if __name__ == "__main__":
    main()
