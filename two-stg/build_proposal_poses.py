#!/usr/bin/env python3
"""
Build pyskl-format pickle files from ActionFormer proposals + SAM3 pose H5 caches.

Uses the same keypoint extraction as pyskl/tools/data/create_sails_annotations.py.

Each proposal gets a deterministic ``proposal_id`` matching
``stage2_three_way.format_proposal_id`` / eval_two_stage_tal.py.

**Pose cache coverage:** SAM3/HDF5 caches under ``pose_sam3/<video_stem>/`` may only
contain the **first N frames** of a long video. Proposals whose time range maps to
frame indices **above** ``max(frame_*`` in the H5) are skipped (see
``create_annotation`` messages: "starts after end of pose cache"). Fix: re-run pose
extraction for the full video length, or accept skips for late segments.

**Skip log:** Each run writes ``<output-dir>/proposals_fold{N}_skips.json`` listing every
skipped proposal (clip id, video paths, ``reason`` code, ``detail``).

Usage (from actreg root):
  python two-stg/build_proposal_poses.py --fold 0
  python two-stg/build_proposal_poses.py --all-folds
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

ACTREG_ROOT = Path(__file__).resolve().parent.parent
PYSKL_ROOT = ACTREG_ROOT / "pyskl"
if str(PYSKL_ROOT) not in sys.path:
    sys.path.insert(0, str(PYSKL_ROOT))
# Resolve filesystem locations from the repo's single source of truth (config.yaml).
if str(ACTREG_ROOT) not in sys.path:
    sys.path.insert(0, str(ACTREG_ROOT))
from paths import PATHS  # noqa: E402

from tools.data.create_sails_annotations import (  # noqa: E402
    DEFAULT_VIDEO_META_JSON,
    SegmentInfo,
    create_annotation,
    find_pose_cache,
    load_pose_cache,
    load_video_meta,
)

from stage2_three_way import format_proposal_id  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SPLITS_DIR = ACTREG_ROOT / "dataprep/tal/splits_cv_4class"
DEFAULT_DET_ROOT = ACTREG_ROOT / "OpenTAD/exps/sails_rmm"
DEFAULT_POSE_CACHE_BASE = PATHS.cache_for_tracking
DEFAULT_OUT_DIR = ACTREG_ROOT / "two-stg" / "proposal_pickles"


def _count_val_proposals(results: dict[str, list], video_df: pd.DataFrame) -> int:
    valid = set(video_df["af_key"].values)
    return sum(len(proposals) for af_key, proposals in results.items() if af_key in valid)


def _make_progress_bar(
    total: int,
    *,
    desc: str,
    enabled: bool,
) -> Any:
    """tqdm on stderr when available; else periodic logger lines."""
    if not enabled or total <= 0:

        class _Noop:
            def update(self, n: int = 1) -> None:
                pass

            def set_postfix_str(self, s: str, *, refresh: bool = True) -> None:
                pass

            def close(self) -> None:
                pass

        return _Noop()

    if tqdm is not None:
        return tqdm(
            total=total,
            desc=desc,
            unit="prop",
            file=sys.stderr,
            dynamic_ncols=True,
            miniters=1,
            smoothing=0.05,
        )

    class _Fallback:
        def __init__(self) -> None:
            self._n = 0
            self._step = max(1, total // 100)

        def update(self, n: int = 1) -> None:
            self._n += n
            if self._n == total or self._n % self._step == 0:
                logger.info(
                    "%s %d / %d (%.1f%%)",
                    desc,
                    self._n,
                    total,
                    100.0 * self._n / total,
                )

        def set_postfix_str(self, s: str, *, refresh: bool = True) -> None:
            pass

        def close(self) -> None:
            logger.info("%s finished %d / %d", desc, self._n, total)

    return _Fallback()


def load_af_key_rows(window_splits_dir: Path, fold: int) -> pd.DataFrame:
    val_csv = window_splits_dir / f"fold_{fold}_val_windows.csv"
    if not val_csv.exists():
        raise FileNotFoundError(val_csv)
    df = pd.read_csv(val_csv)
    df["filename_stem"] = df["filename"].apply(lambda x: Path(str(x)).stem)
    df["af_key"] = df["child_id"].astype(str) + "_" + df["filename_stem"].astype(str)
    return df.groupby("af_key").first().reset_index()


def build_video_meta_lookup(video_df: pd.DataFrame, video_meta_path: Path) -> dict:
    """FPS/shape per video_file; merge with global video_meta.json for filename aliases."""
    lookup: dict[str, dict] = {}
    for _, row in video_df.iterrows():
        vf = str(row["video_file"]).replace("\\", "/").strip()
        fps = float(row["fps"]) if pd.notna(row.get("fps")) and float(row["fps"]) > 0 else 30.0
        lookup[vf] = {"fps": fps, "width": 640, "height": 480}

    extra = load_video_meta(video_meta_path)
    # Map by FileName (basename) for find_pose_cache fallbacks
    for k, v in extra.items():
        if k and k not in lookup:
            lookup[k] = v
    return lookup


def build_fold_pickle(
    fold: int,
    *,
    window_splits_dir: Path,
    det_json: Path,
    pose_cache_base: Path,
    video_meta_path: Path,
    out_path: Path,
    min_keypoint_conf: float,
    max_proposals: int | None = None,
    show_progress: bool = True,
    write_skip_json: bool = True,
) -> tuple[int, int, int]:
    """
    Returns:
        (n_proposals_total, n_in_pickle, n_skipped_no_pose)
    """
    with open(det_json, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", {})

    video_df = load_af_key_rows(window_splits_dir, fold)
    meta_lookup = build_video_meta_lookup(video_df, video_meta_path)

    raw_total = _count_val_proposals(results, video_df)
    bar_total = raw_total if max_proposals is None else min(raw_total, max_proposals)
    pbar = _make_progress_bar(
        bar_total,
        desc=f"fold {fold} proposals",
        enabled=show_progress,
    )

    annotations = []
    split_ids: list[str] = []
    skip_records: list[dict[str, Any]] = []
    n_total = 0
    n_skip = 0

    video_root = ACTREG_ROOT  # unused when meta_lookup has fps
    pose_cache_memo: dict[str, dict | None] = {}
    n_h5_loads = 0
    t0 = time.perf_counter()

    try:
        for af_key, proposals in results.items():
            if max_proposals is not None and n_total >= max_proposals:
                break
            if af_key not in video_df["af_key"].values:
                continue
            row = video_df.loc[video_df["af_key"] == af_key].iloc[0]
            video_file = str(row["video_file"]).replace("\\", "/").strip()
            filename = str(row["filename"]).strip()
            child_id = str(row["child_id"])
            timepoint = "tal_prop"

            stub = SegmentInfo(
                segment_id="__probe__",
                video_file=video_file,
                filename=filename,
                start_sec=0.0,
                end_sec=1.0,
                label=0,
                label_name="dummy",
                child_id=child_id,
                timepoint=timepoint,
            )
            cache_path = find_pose_cache(stub, pose_cache_base, meta_lookup)
            cache_key = str(cache_path.resolve()) if cache_path else f"__MISS__{af_key}"

            if cache_key not in pose_cache_memo:
                if cache_path is not None:
                    pose_cache_memo[cache_key] = load_pose_cache(cache_path)
                    n_h5_loads += 1
                else:
                    pose_cache_memo[cache_key] = None
            video_pose = pose_cache_memo[cache_key]

            for prop in proposals:
                if max_proposals is not None and n_total >= max_proposals:
                    break
                n_total += 1
                seg = prop["segment"]
                start_sec = float(seg[0])
                end_sec = float(seg[1])
                pid = format_proposal_id(af_key, start_sec, end_sec)

                segment = SegmentInfo(
                    segment_id=pid,
                    video_file=video_file,
                    filename=filename,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    label=0,
                    label_name="dummy",
                    child_id=child_id,
                    timepoint=timepoint,
                )
                ann = create_annotation(
                    segment,
                    pose_cache_base,
                    video_root,
                    video_meta_lookup=meta_lookup,
                    min_keypoint_conf=min_keypoint_conf,
                    pose_data=video_pose,
                    skip_events=skip_records,
                )
                if ann is None:
                    n_skip += 1
                else:
                    annotations.append(ann)
                    split_ids.append(pid)
                pbar.update(1)
                pbar.set_postfix_str(
                    f"ok={len(annotations)} skip={n_skip}",
                    refresh=False,
                )

            if max_proposals is not None and n_total >= max_proposals:
                break

    finally:
        pbar.close()

    elapsed = time.perf_counter() - t0
    logger.info(
        "Fold %d timing: %.1fs total, %d H5 loads (of %d videos), %.3fs/proposal",
        fold, elapsed, n_h5_loads,
        len([k for k in results if k in set(video_df["af_key"].values)]),
        elapsed / max(n_total, 1),
    )

    pkl_data = {"split": {"val": split_ids}, "annotations": annotations}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pkl_data, f, protocol=4)

    if write_skip_json:
        skip_json_path = out_path.parent / f"{out_path.stem}_skips.json"
        skip_payload = {
            "fold": fold,
            "unix_time": time.time(),
            "detection_json": str(det_json.resolve()),
            "pickle_output": str(out_path.resolve()),
            "pose_cache_base": str(pose_cache_base.resolve()),
            "video_meta": str(video_meta_path.resolve()),
            "min_keypoint_conf": min_keypoint_conf,
            "max_proposals": max_proposals,
            "n_proposals_attempted": n_total,
            "n_in_pickle": len(annotations),
            "n_skipped": n_skip,
            "reason_counts": _count_skip_reasons(skip_records),
            "skipped_proposals": skip_records,
        }
        with open(skip_json_path, "w", encoding="utf-8") as jf:
            json.dump(skip_payload, jf, indent=2, ensure_ascii=False)
        logger.info("Wrote skip log -> %s", skip_json_path)

    trunc = f" (truncated at max_proposals={max_proposals})" if max_proposals is not None else ""
    logger.info(
        "Fold %d: proposals total=%d, in pickle=%d, skipped (no/weak pose)=%d -> %s%s",
        fold,
        n_total,
        len(annotations),
        n_skip,
        out_path,
        trunc,
    )
    return n_total, len(annotations), n_skip


def _count_skip_reasons(records: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        k = str(r.get("reason", "unknown"))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda x: (-x[1], x[0])))


def main() -> None:
    p = argparse.ArgumentParser(description="ActionFormer proposals -> pyskl pickle (pose)")
    p.add_argument("--fold", type=int, default=None, choices=[0, 1, 2])
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--window-splits-dir", type=Path, default=DEFAULT_WINDOW_SPLITS_DIR)
    p.add_argument(
        "--detection-json",
        type=Path,
        default=None,
        help="Override detection JSON (default: OpenTAD .../actionformer_vjepa_binary_foldN/.../result_detection.json)",
    )
    p.add_argument("--pose-cache-base", type=Path, default=DEFAULT_POSE_CACHE_BASE)
    p.add_argument(
        "--video-meta",
        type=Path,
        default=Path(DEFAULT_VIDEO_META_JSON),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    p.add_argument(
        "--min-keypoint-conf",
        type=float,
        default=0.4,
        help="Same default as CV pickles (0.4)",
    )
    p.add_argument(
        "--max-proposals",
        type=int,
        default=None,
        metavar="N",
        help="Smoke test: stop after attempting N proposals (default: no limit).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (tqdm on stderr, or periodic logs if tqdm missing).",
    )
    p.add_argument(
        "--no-skip-json",
        action="store_true",
        help="Do not write proposals_fold{N}_skips.json next to the pickle.",
    )
    args = p.parse_args()

    folds = [0, 1, 2] if args.all_folds else [args.fold]
    if folds == [None]:
        p.error("Provide --fold N or --all-folds")

    for fold in folds:
        det = args.detection_json
        if det is None:
            det = (
                DEFAULT_DET_ROOT
                / f"actionformer_vjepa_binary_fold{fold}"
                / "gpu1_id99"
                / "result_detection.json"
            )
        if not det.exists():
            raise FileNotFoundError(det)
        out = args.output_dir / f"proposals_fold{fold}.pkl"
        build_fold_pickle(
            fold,
            window_splits_dir=args.window_splits_dir,
            det_json=det,
            pose_cache_base=args.pose_cache_base,
            video_meta_path=args.video_meta,
            out_path=out,
            min_keypoint_conf=args.min_keypoint_conf,
            max_proposals=args.max_proposals,
            show_progress=not args.no_progress,
            write_skip_json=not args.no_skip_json,
        )


if __name__ == "__main__":
    main()
