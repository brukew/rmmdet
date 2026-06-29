"""
Summarize pose "child present" coverage on Stage-1 proposals (same rules as pose_filter_eval).

Also reports longest *contiguous* stretch of child-present frames within each proposal —
relevant for variant (B): crop Stage-2 to that interval.

Usage:
  conda activate opentad && python two-stg/analyze_pose_coverage_proposals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import shared pose helpers (same H5 paths, FPS lookup, etc.)
ACTREG = Path("/orcd/data/satra/001/users/brukew/actreg")
sys.path.insert(0, str(ACTREG / "two-stg"))
from pose_filter_eval import (  # noqa: E402
    ARM_KEYPOINT_INDICES,
    COCO_BODY_INDICES,
    build_filename_lookup,
    build_video_fps_lookup,
    load_pose_cache_lazy,
)

RUN5_PRED = ACTREG / "two-stg/eval_results_tridet"


def frame_child_present(pose_data: dict[int, np.ndarray], frame_idx: int) -> bool:
    if frame_idx not in pose_data:
        return False
    kp = pose_data[frame_idx]
    if kp.shape[0] < 17 or kp.shape[1] < 3:
        return False
    body_scores = kp[COCO_BODY_INDICES, 2]
    return bool(np.any(body_scores > 0.3))


def coverage_and_longest_run(
    pose_data: dict[int, np.ndarray] | None,
    start_sec: float,
    end_sec: float,
    fps: float,
) -> dict:
    """Fraction of frames with child present; longest contiguous run (frames and ratio)."""
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    if end_frame <= start_frame:
        end_frame = start_frame + int(fps)

    total = end_frame - start_frame
    if total <= 0 or pose_data is None:
        return {
            "child_presence": 0.0,
            "longest_run_frames": 0,
            "longest_run_ratio": 0.0,
            "n_runs": 0,
            "duration_sec": 0.0,
        }

    present = [
        frame_child_present(pose_data, f) for f in range(start_frame, end_frame)
    ]
    n_present = sum(present)
    # Longest contiguous True run
    longest = cur = 0
    n_runs = 0
    in_run = False
    for p in present:
        if p:
            cur += 1
            longest = max(longest, cur)
            if not in_run:
                n_runs += 1
                in_run = True
        else:
            cur = 0
            in_run = False

    return {
        "child_presence": n_present / total,
        "longest_run_frames": longest,
        "longest_run_ratio": longest / total,
        "n_runs": n_runs,
        "duration_sec": (end_sec - start_sec),
    }


def main():
    all_rows = []
    for fold in (0, 1, 2):
        csv_path = RUN5_PRED / f"fold{fold}" / "predictions.csv"
        if not csv_path.exists():
            print(f"Missing {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        fps_lu = build_video_fps_lookup(fold)
        fn_lu = build_filename_lookup(fold)

        for _, row in df.iterrows():
            vk = row["video_key"]
            fps = fps_lu.get(vk, 30.0)
            stem = fn_lu.get(vk, Path(vk).stem)
            pose = load_pose_cache_lazy(stem)
            stats = coverage_and_longest_run(
                pose, row["start_sec"], row["end_sec"], fps
            )
            stats["fold"] = fold
            all_rows.append(stats)

    d = pd.DataFrame(all_rows)
    n = len(d)
    print(f"Proposals analyzed: {n} (Run 5, folds 0–2)\n")

    def pct(s: pd.Series, q: float) -> float:
        return float(np.percentile(s.dropna(), q))

    cp = d["child_presence"]
    lr = d["longest_run_ratio"]

    print("=== Child presence (any frame in proposal with body keypoint conf > 0.3) ===")
    print(f"  mean:   {cp.mean():.3f}")
    print(f"  median: {cp.median():.3f}")
    print(f"  std:    {cp.std():.3f}")
    print(f"  p10 / p25 / p75 / p90 / p95:")
    print(f"    {pct(cp,10):.3f}  {pct(cp,25):.3f}  {pct(cp,75):.3f}  {pct(cp,90):.3f}  {pct(cp,95):.3f}")
    for thr in (0.1, 0.3, 0.5, 0.8, 1.0):
        frac = (cp >= thr - 1e-9).mean()
        print(f"  fraction >= {thr}: {frac*100:.1f}%")
    print(f"  fraction == 0: {(cp == 0).mean()*100:.1f}%")

    print("\n=== Longest contiguous child-present run / proposal length ===")
    print("  (Variant B would crop Stage-2 to this sub-interval, padded to model min length.)")
    print(f"  mean:   {lr.mean():.3f}")
    print(f"  median: {lr.median():.3f}")
    print(f"  p10 / p25 / p75 / p90 / p95:")
    print(f"    {pct(lr,10):.3f}  {pct(lr,25):.3f}  {pct(lr,75):.3f}  {pct(lr,90):.3f}  {pct(lr,95):.3f}")

    print("\n=== Disjoint child-visible segments per proposal ===")
    nr = d["n_runs"]
    print(f"  mean runs: {nr.mean():.2f}  median: {nr.median():.0f}")
    print(f"  fraction with >1 run: {(nr > 1).mean()*100:.1f}%")

    print("\n=== Proposal duration (seconds) ===")
    dur = d["duration_sec"]
    print(f"  mean: {dur.mean():.2f}s  median: {dur.median():.2f}s")

    # Example: how many proposals have longest run covering at least 50% but overall presence < 100%?
    mask = (lr >= 0.5) & (cp < 0.99)
    print(f"\n=== Variant B vs full window ===")
    print(f"  Proposals where longest contiguous child run >= 50% of span but overall presence < 99%:")
    print(f"    {mask.sum()} ({100*mask.mean():.1f}%) — intermittent visibility; B picks one contiguous piece.")


if __name__ == "__main__":
    main()
