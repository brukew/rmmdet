#!/usr/bin/env python3
"""
Fuse four STGCN++ modality prediction CSVs (j, b, jm, bm) with weights [2, 2, 1, 1].

Joins on ``segment_id`` (inner intersection). Writes ``predictions_clip.csv`` compatible
with downstream loaders.

Usage:
  python two-stg/fuse_stgcn_4stream.py \\
    --j-csv path/j/predictions_clip.csv \\
    --b-csv path/b/predictions_clip.csv \\
    --jm-csv path/jm/predictions_clip.csv \\
    --bm-csv path/bm/predictions_clip.csv \\
    --output path/fused/predictions_clip.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = (2.0, 2.0, 1.0, 1.0)


def fuse_csvs(
    paths: list[Path],
    weights: tuple[float, float, float, float],
    output: Path,
    num_classes: int = 4,
) -> None:
    score_cols = [f"score_class{i}" for i in range(num_classes)]
    parts: list[pd.DataFrame] = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        if not all(c in df.columns for c in score_cols):
            raise ValueError(f"Missing score columns in {p}: {df.columns.tolist()}")
        rename = {c: f"{c}_s{i}" for c in score_cols}
        parts.append(df[["segment_id"] + score_cols].rename(columns=rename))

    merged = parts[0]
    for d in parts[1:]:
        merged = merged.merge(d, on="segment_id", how="inner")

    wsum = float(sum(weights))
    fused = np.zeros((len(merged), num_classes), dtype=np.float64)
    for stream_i, w in enumerate(weights):
        for j in range(num_classes):
            c = f"score_class{j}_s{stream_i}"
            fused[:, j] += w * merged[c].astype(np.float64).values
    fused /= wsum

    row_sums = fused.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    fused = fused / row_sums

    out_df = pd.DataFrame({"segment_id": merged["segment_id"].values})
    for j in range(num_classes):
        out_df[f"score_class{j}"] = fused[:, j]

    pred_label = fused.argmax(axis=1)
    out_df["true_label"] = 0
    out_df["pred_label"] = pred_label
    out_df["true_class"] = "dummy"
    out_df["pred_class"] = pred_label.astype(str)

    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False)
    print(f"Wrote {output} ({len(out_df)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description="Fuse 4 STGCN++ modality CSVs")
    p.add_argument("--j-csv", type=Path, required=True)
    p.add_argument("--b-csv", type=Path, required=True)
    p.add_argument("--jm-csv", type=Path, required=True)
    p.add_argument("--bm-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-classes", type=int, default=4)
    args = p.parse_args()

    fuse_csvs(
        [args.j_csv, args.b_csv, args.jm_csv, args.bm_csv],
        DEFAULT_WEIGHTS,
        args.output,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
