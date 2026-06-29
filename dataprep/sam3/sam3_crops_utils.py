"""
Shared helpers originally prototyped in visualize_sam3_crops.ipynb.

These utilities resolve raw video paths, SAM3 outputs, and mask caches; they
also handle rotation metadata and box alignment so other scripts (e.g., pose
estimation on crops) can import them without duplicating notebook logic.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from pycocotools import mask as maskUtils

# Default paths mirror the notebook configuration.
DEFAULT_PARSED_CSV = Path("/orcd/data/satra/001/users/brukew/actreg/dataprep/rmm_sam3_parsed.csv")
DEFAULT_ORIGINAL_VIDEO_BASE = Path(
    "/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized"
)
# Resolve filesystem locations from the repo's single source of truth (config.yaml).
import sys as _sys
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from paths import PATHS  # noqa: E402

DEFAULT_SAM3_OUTPUT_BASE = PATHS.rmm_sam_numbered
DEFAULT_MASK_CACHE_BASE = PATHS.cache_for_tracking
DEFAULT_ROTATION_REPORT = Path(
    "/orcd/data/satra/001/users/brukew/sailsprep/feature_processing/tracker/sam3/rotation_inconsistencies_report.csv"
)


def _slugify(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text.strip())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "none"


def load_parsed_csv(path: Path = DEFAULT_PARSED_CSV) -> List[Dict]:
    """Load the parsed SAM3 CSV with time intervals and attach row indices."""
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            row["row_idx"] = idx
            try:
                row["intervals"] = json.loads(row.get("child_sam3_ids", "[]"))
            except Exception:
                row["intervals"] = []
            rows.append(row)
    return rows


def load_rotation_report(path: Path = DEFAULT_ROTATION_REPORT) -> Dict[str, Dict]:
    """Load rotation metadata keyed by video stem."""
    rotation_data: Dict[str, Dict] = {}
    if not path.exists():
        return rotation_data
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "")
            stem = Path(filename).stem
            rotation_data[stem] = {
                "rotation_metadata": row.get("rotation_metadata"),
                "best_rotation": row.get("best_rotation"),
                "rotation_issue": row.get("rotation_issue") == "True",
            }
    return rotation_data


def get_video_paths(
    row_idx: int,
    videos_data: List[Dict],
    original_base: Path = DEFAULT_ORIGINAL_VIDEO_BASE,
    sam3_output_base: Path = DEFAULT_SAM3_OUTPUT_BASE,
    mask_cache_base: Path = DEFAULT_MASK_CACHE_BASE,
    prompt: str = "person",
    model_name: str = "facebook-sam3",
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Resolve paths for the original video, the SAM3 overlay, and the mask cache.

    The SAM3 overlay path follows the numbered-directory convention in the CSV
    parsing (dir = row_idx + 1, file = <stem>_segmented.mp4).
    """
    row = videos_data[row_idx - 1]
    source_file = row.get("SourceFile", "")
    stem = Path(source_file).stem

    source_parts = source_file.split("/")
    if len(source_parts) > 1:
        original_path = original_base / "/".join(source_parts[:-1]) / f"{stem}.mp4"
    else:
        original_path = original_base / source_parts[0] / f"{stem}.mp4"
    if not original_path.exists():
        original_path = None

    sam3_dir = sam3_output_base / f"{row_idx + 1:03d}_{row_idx + 1}"
    sam3_path = sam3_dir / f"{stem}_segmented.mp4"
    if not sam3_path.exists():
        sam3_path = None

    video_basename = f"{stem}_segmented"
    mask_cache_path = (
        mask_cache_base
        / "masks"
        / video_basename
        / f"{model_name}__prompt-{_slugify(prompt)}.h5"
    )
    if not mask_cache_path.exists():
        mask_cache_path = None

    return original_path, sam3_path, mask_cache_path


def get_video_rotation(video_path: Path) -> int:
    """Return rotation degrees from ffprobe side data (0 if absent)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream_side_data=rotation",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        if result.stdout.strip():
            return int(float(result.stdout.strip()))
    except Exception:
        pass
    return 0


def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate frame by multiples of 90 degrees."""
    if rotation == 0:
        return frame
    k = {90: 1, -90: 3, 180: 2, -180: 2, 270: 3, -270: 1}.get(rotation, 0)
    if k != 0:
        return np.rot90(frame, k)
    return frame


def get_frame_from_video(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Extract a specific frame as RGB."""
    if not video_path or not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def get_video_info(video_path: Path) -> Dict:
    """Collect basic video metadata."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    return info


def load_mask_cache(
    cache_path: Optional[Path],
    video_basename: Optional[str] = None,
    prompt: str = "person",
    model_name: str = "facebook-sam3",
    mask_cache_base: Path = DEFAULT_MASK_CACHE_BASE,
) -> Optional[Dict]:
    """
    Load SAM3 mask cache from HDF5 file.

    Resolution order:
        1) If cache_path is a file, use it.
        2) If cache_path is a directory, prefer `<model>__prompt-<slug>.h5`, else first .h5.
        3) Build the path using the MaskCacheManager layout:
           `<mask_cache_base>/masks/<video_basename>/<model>__prompt-<slug>.h5`
           (also tries `_segmented` / `_noviz` suffixes).
    """

    def add_candidates(base_dir: Path, slug: str, out: List[Path]) -> None:
        out.append(base_dir / f"{model_name}__prompt-{slug}.h5")
        out.extend(sorted(base_dir.glob("*.h5")))

    prompt_slug = _slugify(prompt)
    candidates: List[Path] = []

    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.is_file():
            candidates.append(cache_path)
        elif cache_path.is_dir():
            add_candidates(cache_path, prompt_slug, candidates)

    def add_from_basename(basename: str) -> None:
        base_dir = mask_cache_base / "masks" / basename
        add_candidates(base_dir, prompt_slug, candidates)

    if video_basename:
        add_from_basename(video_basename)
        if not video_basename.endswith("_segmented"):
            add_from_basename(f"{video_basename}_segmented")
        if not video_basename.endswith("_noviz"):
            add_from_basename(f"{video_basename}_noviz")

    chosen = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            chosen = cand
            break
    if chosen is None:
        return None

    try:
        with h5py.File(chosen, "r") as f:
            frame_indices = []
            obj_ids = {}
            boxes = {}
            scores = {}
            rles = {}

            for key in f.keys():
                if not key.startswith("frame_"):
                    continue
                try:
                    frame_idx = int(key.split("_", 1)[1])
                except ValueError:
                    continue
                frame_indices.append(frame_idx)
                grp = f[key]
                obj_ids[frame_idx] = grp["obj_ids"][:]
                boxes[frame_idx] = grp["boxes"][:]
                scores[frame_idx] = grp["scores"][:]
                if "rles" in grp:
                    rles[frame_idx] = [np.array(rle, dtype=np.uint8) for rle in grp["rles"][:]]

            if not frame_indices:
                return None

            attrs = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in f.attrs.items()}

            return {
                "path": chosen,
                "frame_indices": np.array(sorted(frame_indices)),
                "obj_ids": obj_ids,
                "boxes": boxes,
                "scores": scores,
                "rles": rles,
                "attrs": attrs,
            }
    except Exception:
        return None


def rotate_boxes(boxes: np.ndarray, rotation: int, width: int, height: int) -> np.ndarray:
    """Rotate bounding boxes according to video rotation metadata."""
    if rotation == 0:
        return boxes
    rotated = boxes.copy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        if rotation in [-90, 270]:
            new_x1, new_y1 = y1, width - x2
            new_x2, new_y2 = y2, width - x1
            rotated[i] = [new_x1, new_y1, new_x2, new_y2]
        elif rotation in [90, -270]:
            new_x1, new_y1 = height - y2, x1
            new_x2, new_y2 = height - y1, x2
            rotated[i] = [new_x1, new_y1, new_x2, new_y2]
        elif rotation in [180, -180]:
            new_x1, new_y1 = width - x2, height - y2
            new_x2, new_y2 = width - x1, height - y1
            rotated[i] = [new_x1, new_y1, new_x2, new_y2]
    return rotated


def find_child_ids(intervals: List[Dict], frame_time: float) -> List[int]:
    """Find child IDs active at a given timestamp."""
    child_ids: List[int] = []
    for iv in intervals:
        start = iv.get("start_sec", 0)
        end = iv.get("end_sec") or float("inf")
        if start <= frame_time < end:
            try:
                child_ids.append(int(iv["id"]))
            except Exception:
                continue
    return child_ids


def prepare_frame_and_boxes(
    row_idx: int,
    frame_idx: int,
    videos_data: List[Dict],
    rotation_data: Dict[str, Dict],
    apply_rotation_fix: bool = True,
    rotation_override: Optional[int] = None,
    autorotated_same_dims: Optional[bool] = None,
    return_masks: bool = False,
    target_obj_ids: Optional[Iterable[int]] = None,
) -> Optional[Dict]:
    """
    Load frame + aligned boxes for a given CSV row and frame index.

    Returns a dict with:
        frame (RGB), boxes, obj_ids, scores, child_ids, frame_time,
        matched_indices, meta (paths/info/rotation used)
    """
    row = videos_data[row_idx - 1]
    filename = row.get("FileName", "")
    stem = Path(filename).stem
    intervals = row.get("intervals", [])

    original_path, sam3_path, cache_path = get_video_paths(row_idx, videos_data)
    video_basename = sam3_path.stem if sam3_path else f"{stem}_segmented"

    if not original_path:
        return None

    video_info = get_video_info(original_path)
    rotation_meta = get_video_rotation(original_path) * -1 if apply_rotation_fix else 0
    rotation_to_use = rotation_override if rotation_override is not None else rotation_meta

    cache = load_mask_cache(cache_path, video_basename=video_basename, prompt="person", model_name="facebook-sam3")
    cache_width = cache_height = None
    cache_stride = None
    if cache and cache.get("attrs"):
        attrs = cache["attrs"]
        cache_width = attrs.get("width")
        cache_height = attrs.get("height")
        cache_stride = attrs.get("frame_stride")

    frame = get_frame_from_video(original_path, frame_idx)
    if frame is None:
        return None

    frame_w, frame_h = frame.shape[1], frame.shape[0]
    base_width = cache_width or video_info["width"]
    base_height = cache_height or video_info["height"]

    auto_rotated_guess = bool(rotation_to_use) and frame_w == base_height and frame_h == base_width
    if autorotated_same_dims is True:
        auto_rotated = bool(rotation_to_use)
    elif autorotated_same_dims is False:
        auto_rotated = False
    else:
        auto_rotated = auto_rotated_guess

    rotate_frame = bool(rotation_to_use) and apply_rotation_fix and not auto_rotated
    rotate_boxes_flag = bool(rotation_to_use) and apply_rotation_fix

    print(f"Rotation: {rotate_frame}")

    if rotate_frame:
        frame = apply_rotation(frame, rotation_to_use)
        frame_w, frame_h = frame.shape[1], frame.shape[0]

    frame_time = frame_idx / video_info["fps"] if video_info["fps"] else 0.0
    child_ids = find_child_ids(intervals, frame_time)

    boxes = obj_ids = scores = None
    closest_frame_idx = None
    decoded_masks = None
    if cache:
        frame_indices = cache["frame_indices"]
        if len(frame_indices) > 0:
            pos = int(np.argmin(np.abs(frame_indices - frame_idx)))
            closest_frame_idx = int(frame_indices[pos])
            boxes = cache["boxes"][closest_frame_idx]
            obj_ids = cache["obj_ids"][closest_frame_idx]
            scores = cache["scores"][closest_frame_idx]
            if return_masks and "rles" in cache and closest_frame_idx in cache["rles"]:
                # Decode masks for this frame to match the cache dimensions (attrs if present)
                decoded_masks = []
                decode_h = cache_height or base_height
                decode_w = cache_width or base_width
                for rle in cache["rles"][closest_frame_idx]:
                    rle_dict = {"counts": rle.tobytes(), "size": [decode_h, decode_w]}
                    mask = maskUtils.decode(rle_dict)
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    decoded_masks.append(mask.astype(bool))

    boxes_rotated = None
    matched_indices: List[int] = []
    if boxes is not None and obj_ids is not None:
        boxes_rotated = boxes.copy()
        expected_w, expected_h = base_width, base_height

        if rotate_boxes_flag and rotation_to_use:
            boxes_rotated = rotate_boxes(boxes_rotated, rotation_to_use, base_width, base_height)
            if rotation_to_use in (-90, 90, -270, 270):
                expected_w, expected_h = base_height, base_width
            elif rotation_to_use in (-180, 180):
                expected_w, expected_h = base_width, base_height

            if decoded_masks is not None:
                rotated_masks = []
                for m in decoded_masks:
                    rotated_masks.append(apply_rotation(m, rotation_to_use))
                decoded_masks = rotated_masks

        if expected_w and expected_h and (frame_w != expected_w or frame_h != expected_h):
            scale_x = frame_w / expected_w
            scale_y = frame_h / expected_h
            boxes_rotated[:, [0, 2]] *= scale_x
            boxes_rotated[:, [1, 3]] *= scale_y
            if decoded_masks is not None:
                resized_masks = []
                for m in decoded_masks:
                    resized = cv2.resize(m.astype(np.uint8), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                    resized_masks.append(resized.astype(bool))
                decoded_masks = resized_masks

        if child_ids:
            matched_indices = [i for i, obj in enumerate(obj_ids) if int(obj) in child_ids]

        # Ensure masks list aligns with boxes length; otherwise drop masks to avoid misalignment.
        if decoded_masks is not None and len(decoded_masks) != len(boxes_rotated):
            decoded_masks = None

        # If masks exist, heuristically fix 180° flips by comparing IoU against boxes.
        if decoded_masks is not None:
            def mask_bbox(mask_arr: np.ndarray) -> Optional[np.ndarray]:
                ys, xs = np.nonzero(mask_arr)
                if len(xs) == 0 or len(ys) == 0:
                    return None
                return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

            def mean_iou(masks_list: List[np.ndarray]) -> float:
                ious = []
                for m, b in zip(masks_list, boxes_rotated):
                    mb = mask_bbox(m)
                    if mb is None:
                        continue
                    # compute IoU between two boxes
                    ix1 = max(mb[0], b[0]); iy1 = max(mb[1], b[1])
                    ix2 = min(mb[2], b[2]); iy2 = min(mb[3], b[3])
                    iw = max(0.0, ix2 - ix1)
                    ih = max(0.0, iy2 - iy1)
                    inter = iw * ih
                    area_m = (mb[2] - mb[0]) * (mb[3] - mb[1])
                    area_b = (b[2] - b[0]) * (b[3] - b[1])
                    union = area_m + area_b - inter if (area_m + area_b - inter) > 0 else 1.0
                    ious.append(inter / union)
                return float(np.mean(ious)) if ious else 0.0

            base_iou = mean_iou(decoded_masks)
            # Flip masks 180° about center to see if alignment improves
            flipped = [np.flipud(np.fliplr(m)) for m in decoded_masks]
            flipped_iou = mean_iou(flipped)
            if flipped_iou > base_iou + 0.05:  # small margin to avoid noise
                decoded_masks = flipped

        # Optional filtering to target object IDs
        if target_obj_ids:
            target_set = {int(t) for t in target_obj_ids}
            keep_indices = [i for i, oid in enumerate(obj_ids) if int(oid) in target_set]
            if not keep_indices:
                # No POI in this frame
                boxes_rotated = None
                obj_ids = None
                scores = None
                decoded_masks = None
            else:
                boxes_rotated = boxes_rotated[keep_indices]
                obj_ids = obj_ids[keep_indices]
                scores = scores[keep_indices]
                if decoded_masks is not None:
                    decoded_masks = [decoded_masks[i] for i in keep_indices]

    return {
        "frame": frame,
        "boxes": boxes_rotated,
        "obj_ids": obj_ids,
        "scores": scores,
        "masks": decoded_masks,
        "child_ids": child_ids,
        "matched_indices": matched_indices,
        "frame_time": frame_time,
        "closest_frame_idx": closest_frame_idx,
        "meta": {
            "filename": filename,
            "original_path": original_path,
            "sam3_path": sam3_path,
            "mask_cache_path": cache.get("path") if cache else None,
            "rotation_used": rotation_to_use,
            "rotation_meta": rotation_meta,
            "cache_stride": cache_stride,
            "frame_dims": (frame_w, frame_h),
            "base_dims": (base_width, base_height),
            "cache_dims": (cache_width, cache_height),
        },
    }
