#!/usr/bin/env python3
"""Aggregate per-fold metrics from two-stg eval into cv_summary.json (mean ± std)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("eval_results"), help="Directory containing fold0/, fold1/, fold2/")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: input-dir/cv_summary.json)")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output = args.output or (input_dir / "cv_summary.json")

    fold_metrics = []
    for fold in [0, 1, 2]:
        metrics_path = input_dir / f"fold{fold}" / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing {metrics_path}")
        with open(metrics_path) as f:
            m = json.load(f)
        fold_metrics.append(m)

    # Collect top-level scalar metrics (skip nested dicts like per_class_ap, details)
    metrics_keys = [k for k in fold_metrics[0].keys() if isinstance(fold_metrics[0].get(k), (int, float))]
    summary = {}
    for key in metrics_keys:
        values = []
        for m in fold_metrics:
            v = m.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                values.append(float(v))
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        else:
            summary[key] = {"mean": None, "std": None}

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {output}")
    for key, v in summary.items():
        if v["mean"] is not None:
            print(f"  {key}: {v['mean']*100:.2f}% ± {v['std']*100:.2f}%")


if __name__ == "__main__":
    main()
