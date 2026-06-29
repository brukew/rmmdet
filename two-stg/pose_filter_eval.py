"""
Pose-based proposal filtering for two-stage TAL evaluation.

Level 1: Drop proposals where child presence < threshold (default 50%)
Level 2: Drop "hands flapping" proposals where arm keypoints avg conf < threshold (default 0.6)

Usage:
  python two-stg/pose_filter_eval.py --run 5 --folds 0 1 2 --level 1
  python two-stg/pose_filter_eval.py --run 5 --folds 0 1 2 --level 2
  python two-stg/pose_filter_eval.py --run 5 --folds 0 1 2 --level both
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Resolve filesystem locations from the repo's single source of truth (config.yaml).
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import PATHS  # noqa: E402

ACTREG = PATHS.repo_root
POSE_CACHE_BASE = PATHS.cache_for_tracking
POSE_CACHE_FILENAME = "dino-5scale_swin-l_8xb2-36e_coco_0.5_td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288_sam3guided.h5"
SPLITS_DIR = ACTREG / "dataprep/tal/splits_cv_4class"

ARM_KEYPOINT_INDICES = [7, 8, 9, 10]  # L/R elbow + L/R wrist
COCO_BODY_INDICES = list(range(17))

RUN_CONFIG = {
    "5": ACTREG / "two-stg/eval_results_tridet",
    "5c": ACTREG / "two-stg/eval_results_tridet_3way_live",
    "1": ACTREG / "two-stg/eval_results",
    "3c": ACTREG / "two-stg/eval_results_3way_live",
}

_pose_cache_memo: dict[str, dict | None] = {}


def find_pose_cache_path(filename_stem: str) -> Path | None:
    primary = POSE_CACHE_BASE / "pose_sam3" / filename_stem / POSE_CACHE_FILENAME
    if primary.exists() and filename_stem != "IMG_5399":
        return primary
    fallbacks = [
        POSE_CACHE_BASE / filename_stem / POSE_CACHE_FILENAME,
        POSE_CACHE_BASE / f"{filename_stem}_pose.h5",
        POSE_CACHE_BASE / f"{filename_stem}.h5",
    ]
    for fb in fallbacks:
        if fb.exists():
            return fb
    return None


def load_pose_cache_lazy(filename_stem: str) -> dict[int, np.ndarray] | None:
    """Load pose cache, return {frame_idx: keypoints_133x3} for first person."""
    if filename_stem in _pose_cache_memo:
        return _pose_cache_memo[filename_stem]

    cache_path = find_pose_cache_path(filename_stem)
    if cache_path is None:
        _pose_cache_memo[filename_stem] = None
        return None

    poses = {}
    try:
        with h5py.File(str(cache_path), "r") as f:
            for key in f.keys():
                if not key.startswith("frame_"):
                    continue
                frame_idx = int(key.split("_")[1])
                if "pose_0" in f[key] and "keypoints" in f[key]["pose_0"]:
                    poses[frame_idx] = np.array(f[key]["pose_0"]["keypoints"])
    except Exception as e:
        print(f"  Warning: failed to load {cache_path}: {e}")
        _pose_cache_memo[filename_stem] = None
        return None

    _pose_cache_memo[filename_stem] = poses if poses else None
    return _pose_cache_memo[filename_stem]


def build_video_fps_lookup(fold: int) -> dict[str, float]:
    """Map video_key (from predictions.csv) -> fps."""
    csv_path = SPLITS_DIR / f"fold_{fold}_val_windows.csv"
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        vk = row["video_key"] if pd.notna(row.get("video_key")) else row["video_file"]
        fps = float(row["fps"]) if pd.notna(row.get("fps")) and float(row["fps"]) > 0 else 30.0
        lookup[vk] = fps
    return lookup


def build_filename_lookup(fold: int) -> dict[str, str]:
    """Map video_key -> filename stem for pose cache lookup."""
    csv_path = SPLITS_DIR / f"fold_{fold}_val_windows.csv"
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        vk = row["video_key"] if pd.notna(row.get("video_key")) else row["video_file"]
        fn = row["filename"] if pd.notna(row.get("filename")) else Path(row["video_file"]).name
        lookup[vk] = Path(fn).stem
    return lookup


def compute_pose_stats(
    pose_data: dict[int, np.ndarray],
    start_sec: float,
    end_sec: float,
    fps: float,
) -> dict:
    """Compute pose coverage and keypoint confidence stats for a proposal."""
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    if end_frame <= start_frame:
        end_frame = start_frame + int(fps)

    total_frames = end_frame - start_frame
    if total_frames <= 0:
        return {"child_presence": 0.0, "arm_avg_conf": 0.0, "total_frames": 0}

    frames_with_pose = 0
    arm_confs = []

    for frame_idx in range(start_frame, end_frame):
        if frame_idx in pose_data:
            kp = pose_data[frame_idx]  # [133, 3] or similar
            if kp.shape[0] >= 17 and kp.shape[1] >= 3:
                body_scores = kp[COCO_BODY_INDICES, 2]
                if np.any(body_scores > 0.3):
                    frames_with_pose += 1
                arm_scores = kp[ARM_KEYPOINT_INDICES, 2]
                arm_confs.append(np.mean(arm_scores))

    child_presence = frames_with_pose / total_frames
    arm_avg_conf = float(np.mean(arm_confs)) if arm_confs else 0.0

    return {
        "child_presence": child_presence,
        "arm_avg_conf": arm_avg_conf,
        "total_frames": total_frames,
    }


def annotate_proposals(pred_df: pd.DataFrame, fold: int) -> pd.DataFrame:
    """Add pose stats columns to predictions DataFrame."""
    fps_lookup = build_video_fps_lookup(fold)
    fn_lookup = build_filename_lookup(fold)

    child_presence = []
    arm_avg_conf = []
    n_no_cache = 0

    video_keys = pred_df["video_key"].unique()
    print(f"  Loading pose caches for {len(video_keys)} videos...")

    for _, row in pred_df.iterrows():
        vk = row["video_key"]
        fps = fps_lookup.get(vk, 30.0)
        fn_stem = fn_lookup.get(vk)

        if fn_stem is None:
            fn_stem = Path(vk).stem

        pose_data = load_pose_cache_lazy(fn_stem)
        if pose_data is None:
            child_presence.append(0.0)
            arm_avg_conf.append(0.0)
            n_no_cache += 1
            continue

        stats = compute_pose_stats(pose_data, row["start_sec"], row["end_sec"], fps)
        child_presence.append(stats["child_presence"])
        arm_avg_conf.append(stats["arm_avg_conf"])

    df = pred_df.copy()
    df["child_presence"] = child_presence
    df["arm_avg_conf"] = arm_avg_conf

    if n_no_cache > 0:
        print(f"  Warning: {n_no_cache}/{len(pred_df)} proposals had no pose cache")

    return df


def run_opentad_eval(pred_df, fold):
    sys.path.insert(0, str(ACTREG / "two-stg"))
    from opentad_eval import prediction_df_to_opentad_results, evaluate_with_opentad
    results_dict = prediction_df_to_opentad_results(pred_df)
    metrics = evaluate_with_opentad(results_dict, fold=fold)
    return metrics


def fmt(m):
    return (
        f"avg_mAP={m['average_mAP']*100:.2f}%  "
        f"@0.3={m['mAP@0.3']*100:.2f}  "
        f"@0.5={m['mAP@0.5']*100:.2f}  "
        f"@0.7={m['mAP@0.7']*100:.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=list(RUN_CONFIG.keys()))
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--level", choices=["1", "2", "both"], default="both")
    parser.add_argument("--presence-thresh", type=float, default=0.50)
    parser.add_argument("--arm-conf-thresh", type=float, default=0.60)
    args = parser.parse_args()

    pred_dir = RUN_CONFIG[args.run]
    print(f"=== Pose Filter Eval — Run {args.run} ===")
    print(f"Level: {args.level}  Presence thresh: {args.presence_thresh}  Arm conf thresh: {args.arm_conf_thresh}\n")

    all_results = {}

    for fold in args.folds:
        print(f"\n--- Fold {fold} ---")
        csv_path = pred_dir / f"fold{fold}" / "predictions.csv"
        if not csv_path.exists():
            print(f"  SKIP: {csv_path} not found")
            continue

        pred_df = pd.read_csv(csv_path)
        n_total = len(pred_df)

        t0 = time.time()
        pred_df = annotate_proposals(pred_df, fold)
        print(f"  Pose annotation took {time.time()-t0:.1f}s")

        # Stats
        print(f"  Total proposals: {n_total}")
        n_hf = (pred_df["label"] == "hands flapping").sum()
        print(f"  Hands flapping proposals: {n_hf}")
        print(f"  Child presence: mean={pred_df['child_presence'].mean():.2f}, "
              f"median={pred_df['child_presence'].median():.2f}, "
              f"<{args.presence_thresh}: {(pred_df['child_presence'] < args.presence_thresh).sum()}")
        hf_mask = pred_df["label"] == "hands flapping"
        if hf_mask.any():
            print(f"  Arm conf (hands flapping only): mean={pred_df.loc[hf_mask, 'arm_avg_conf'].mean():.2f}, "
                  f"<{args.arm_conf_thresh}: {(pred_df.loc[hf_mask, 'arm_avg_conf'] < args.arm_conf_thresh).sum()}/{n_hf}")

        fold_results = {}

        # Baseline (no filter)
        m = run_opentad_eval(pred_df, fold)
        fold_results["baseline"] = m
        print(f"\n  baseline ({n_total} props):  {fmt(m)}")

        # Level 1: child presence filter
        if args.level in ("1", "both"):
            df_l1 = pred_df[pred_df["child_presence"] >= args.presence_thresh].copy()
            n_l1 = len(df_l1)
            m = run_opentad_eval(df_l1, fold)
            fold_results["L1"] = m
            print(f"  L1 presence≥{args.presence_thresh} ({n_l1} props, dropped {n_total-n_l1}):  {fmt(m)}")

        # Level 2: arm keypoint filter for hands flapping
        if args.level in ("2", "both"):
            hf_bad = (pred_df["label"] == "hands flapping") & (pred_df["arm_avg_conf"] < args.arm_conf_thresh)
            df_l2 = pred_df[~hf_bad].copy()
            n_l2 = len(df_l2)
            m = run_opentad_eval(df_l2, fold)
            fold_results["L2"] = m
            print(f"  L2 arm_conf≥{args.arm_conf_thresh} for HF ({n_l2} props, dropped {n_total-n_l2}):  {fmt(m)}")

        # Combined
        if args.level == "both":
            combined_mask = (pred_df["child_presence"] >= args.presence_thresh)
            hf_bad = (pred_df["label"] == "hands flapping") & (pred_df["arm_avg_conf"] < args.arm_conf_thresh)
            df_both = pred_df[combined_mask & ~hf_bad].copy()
            n_both = len(df_both)
            m = run_opentad_eval(df_both, fold)
            fold_results["L1+L2"] = m
            print(f"  L1+L2 combined ({n_both} props, dropped {n_total-n_both}):  {fmt(m)}")

        all_results[fold] = fold_results

    # Summary
    print(f"\n\n=== Cross-Fold Summary (Run {args.run}) ===")
    filter_names = ["baseline"]
    if args.level in ("1", "both"):
        filter_names.append("L1")
    if args.level in ("2", "both"):
        filter_names.append("L2")
    if args.level == "both":
        filter_names.append("L1+L2")

    print(f"{'Filter':<20} {'avg_mAP':>8} {'@0.3':>8} {'@0.5':>8} {'@0.7':>8}")
    print("-" * 56)

    for name in filter_names:
        fold_metrics = [all_results[f][name] for f in args.folds if f in all_results and name in all_results[f]]
        if not fold_metrics:
            continue
        avg = lambda key: np.mean([m[key] for m in fold_metrics]) * 100
        print(f"{name:<20} {avg('average_mAP'):>7.2f}% {avg('mAP@0.3'):>7.2f}% {avg('mAP@0.5'):>7.2f}% {avg('mAP@0.7'):>7.2f}%")


if __name__ == "__main__":
    main()
