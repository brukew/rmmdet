#!/usr/bin/env python3
"""
Find contiguous multi-label overlap segments (>=K classes) from TAL window CSVs.

This script operates on the *window-level* TAL CSVs produced by
`make_tal_window_splits.py`, using the `labels` column (JSON list of class indices).

Goal
----
Identify maximal contiguous time ranges within each video where at least K RMM
classes are present simultaneously (by class; i.e., unique indices in `labels`).

Why this exists
---------------
You may want to cut out these multi-label regions to:
- inspect overlap behavior (e.g., jumping + hands flapping)
- optionally exclude multi-label regions for single-label training/evaluation

Key behaviors / assumptions
---------------------------
- Background windows (labels=[]) are ignored entirely.
- CV directories contain the *same* underlying windows repeated across folds;
  we dedupe by `window_id` so each unique window contributes once.
- "Contiguous" means windows that overlap or touch in time are mergeable.
- By default we only merge segments when the *exact label combination* is the same
  (so (jumping, flapping) stays separate from (jumping, rocking)).

Output
------
Writes a CSV where each row is a merged segment:
- video_key
- start_sec, end_sec (half-open semantics inherited from window CSVs)
- num_labels (size of label set)
- labels (JSON list of label indices)
- labels_names (JSON list of label names, if label map available)
- num_windows (how many windows were merged into the segment)

Examples
--------
Analyze a CV directory for 4-class task (run from the repo root):
    python dataprep/tal/analyze_multilabel_overlap_segments.py \
        --split-dir dataprep/tal/splits_cv_4class \
        --label-map dataprep/tal/label_maps/label_map_4class.json \
        --min-k 2 \
        --require-same-combo \
        --output-csv dataprep/tal/overlap_segments/multilabel_ge2_cv_4class_samecombo.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _load_label_map(label_map_path: Optional[Path]) -> Dict[int, str]:
    """
    Load label_map_{task}.json (name -> index) into an index -> name mapping.
    """
    if not label_map_path:
        return {}
    if not label_map_path.exists():
        return {}
    with label_map_path.open("r", encoding="utf-8") as f:
        name_to_idx = json.load(f)
    return {int(idx): str(name) for name, idx in name_to_idx.items()}


def _parse_labels(labels_json: str) -> Tuple[int, ...]:
    """
    Parse the `labels` JSON field from window CSV.

    Returns:
        Sorted, unique tuple of label indices (by-class semantics).
    """
    labels = json.loads(labels_json)
    return tuple(sorted({int(x) for x in labels}))


def load_multilabel_windows(split_dir: Path, min_k: int) -> Dict[str, List[dict]]:
    """
    Load and dedupe windows from a split directory, keeping only windows with >=min_k labels.

    Args:
        split_dir: Directory containing *_windows.csv files.
        min_k: Minimum number of labels in a window to be kept.

    Returns:
        dict[video_key] -> list of window dicts with keys:
            window_id, start_sec, end_sec, labels_set
    """
    by_video: Dict[str, List[dict]] = defaultdict(list)
    seen_window_ids = set()

    csv_files = sorted(split_dir.glob("*_windows.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No '*_windows.csv' files found under: {split_dir}")

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                window_id = row.get("window_id", "")
                if not window_id:
                    continue
                if window_id in seen_window_ids:
                    continue
                seen_window_ids.add(window_id)

                labels_set = _parse_labels(row.get("labels", "[]"))
                if len(labels_set) < min_k:
                    continue  # includes background labels=[]

                by_video[row["video_key"]].append(
                    {
                        "window_id": window_id,
                        "start_sec": float(row["start_sec"]),
                        "end_sec": float(row["end_sec"]),
                        "labels_set": labels_set,
                    }
                )

    # Sort windows per video by time
    for video_key in list(by_video.keys()):
        by_video[video_key].sort(key=lambda w: (w["start_sec"], w["end_sec"], w["window_id"]))

    return by_video


def merge_contiguous_segments(
    windows: Iterable[dict],
    require_same_combo: bool = True,
    eps: float = 1e-9,
) -> List[dict]:
    """
    Merge windows into maximal contiguous segments.

    A window can extend an existing segment if:
    - it overlaps/touches (w.start <= cur.end + eps)
    - and (if require_same_combo) the labels_set matches exactly

    Args:
        windows: Iterable of window dicts (sorted by time recommended).
        require_same_combo: If True, only merge windows with identical label sets.
        eps: Small epsilon for float comparisons.

    Returns:
        List of merged segments:
            start_sec, end_sec, labels_set, num_labels, num_windows
    """
    merged: List[dict] = []
    cur: Optional[dict] = None

    for w in windows:
        if cur is None:
            cur = {
                "start_sec": w["start_sec"],
                "end_sec": w["end_sec"],
                "labels_set": w["labels_set"],
                "num_labels": len(w["labels_set"]),
                "num_windows": 1,
            }
            continue

        overlaps_or_touches = w["start_sec"] <= cur["end_sec"] + eps
        same_combo = w["labels_set"] == cur["labels_set"]

        if overlaps_or_touches and (same_combo or not require_same_combo):
            cur["end_sec"] = max(cur["end_sec"], w["end_sec"])
            cur["num_windows"] += 1
            if not require_same_combo:
                # If we merge across combos, track the union.
                union = set(cur["labels_set"])
                union.update(w["labels_set"])
                cur["labels_set"] = tuple(sorted(union))
                cur["num_labels"] = len(cur["labels_set"])
        else:
            merged.append(cur)
            cur = {
                "start_sec": w["start_sec"],
                "end_sec": w["end_sec"],
                "labels_set": w["labels_set"],
                "num_labels": len(w["labels_set"]),
                "num_windows": 1,
            }

    if cur is not None:
        merged.append(cur)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract contiguous multi-label overlap segments (>=K labels) from TAL window CSVs."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="Directory containing *_windows.csv files (e.g., splits_cv_4class).",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional label_map_{task}.json (name->index) to add labels_names.",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=2,
        help="Minimum number of classes required in a window to be included (default: 2).",
    )
    parser.add_argument(
        "--require-same-combo",
        action="store_true",
        help="Merge only when the exact label combination stays the same.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to write output CSV with merged segments.",
    )
    args = parser.parse_args()

    idx_to_name = _load_label_map(args.label_map)

    by_video = load_multilabel_windows(args.split_dir, min_k=args.min_k)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for video_key, wins in by_video.items():
        merged = merge_contiguous_segments(wins, require_same_combo=args.require_same_combo)
        for seg in merged:
            labels = list(seg["labels_set"])
            labels_names = [idx_to_name.get(i, f"class_{i}") for i in labels] if idx_to_name else []
            rows.append(
                {
                    "video_key": video_key,
                    "start_sec": f"{seg['start_sec']:.3f}",
                    "end_sec": f"{seg['end_sec']:.3f}",
                    "num_labels": seg["num_labels"],
                    "labels": json.dumps(labels),
                    "labels_names": json.dumps(labels_names),
                    "num_windows": seg["num_windows"],
                }
            )

    # Sort output for readability
    rows.sort(key=lambda r: (r["video_key"], float(r["start_sec"]), float(r["end_sec"]), int(r["num_labels"])))

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_key",
                "start_sec",
                "end_sec",
                "num_labels",
                "labels",
                "labels_names",
                "num_windows",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Split dir: {args.split_dir}")
    print(f"Unique videos with >= {args.min_k}-label windows: {len(by_video)}")
    print(f"Total merged segments written: {len(rows)}")
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()




