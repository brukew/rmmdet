#!/usr/bin/env python3
"""
Create trimmed video clips for V-JEPA finetuning using the actreg fold CSVs.

For each row in the CSV we cut the clip between ``start_sec`` and ``end_sec``
from the standardized video file (``Videos_from_external_standardized``) and
write it to the requested output directory with one subfolder per CSV.
"""

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    import h5py  # type: ignore
except Exception:
    h5py = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from decord import VideoReader, cpu  # type: ignore

    _HAVE_DECORD = True
except Exception:
    # Decord is optional; validation will fall back to ffprobe only.
    VideoReader = None
    cpu = None
    _HAVE_DECORD = False


# Resolve filesystem locations from the repo's single source of truth (config.yaml).
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "paths.py").exists())
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from paths import PATHS  # noqa: E402

DEFAULT_VIDEO_ROOT = "/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized"
DEFAULT_OUTPUT_ROOT = str(PATHS.classification_clips)
DEFAULT_CROP_OUTPUT_ROOT = str(PATHS.classification_clips_cropped)
DEFAULT_PARSED_CSV = str(PATHS.repo_root / "dataprep/rmm_sam3_parsed.csv")
DEFAULT_MASK_CACHE_BASE = str(PATHS.cache_for_tracking)
DEFAULT_MASK_MODEL = "facebook-sam3"
DEFAULT_MASK_PROMPT = "person"
DEFAULT_PADDING = 20
MASK_CACHE_BASE = Path(DEFAULT_MASK_CACHE_BASE)


@dataclass
class ClipTask:
    segment_id: str
    source: Path
    output: Path
    start: float
    end: float
    duration: float
    from_zero_length: bool
    metadata: Dict[str, str]


@dataclass
class CropConfig:
    enabled: bool
    mask_cache_base: Path
    mask_model: str
    mask_prompt: str
    padding: int
    rotation_override: Optional[int]
    fallback: str  # "full" or "skip"
    apply_rotation_fix: bool = True


@dataclass
class CropResult:
    applied: bool
    reason: str
    box: Optional[Tuple[int, int, int, int]]
    matched_ids: List[int]
    frame_idx: Optional[int]
    cache_frame_idx: Optional[int]
    cache_path: Optional[Path]
    rotation_used: Optional[int]


def normalize_fieldnames(fieldnames: Sequence[str]) -> List[str]:
    """Return cleaned field names so we survive odd prefixes like 'hm howsegment_id'."""
    cleaned: List[str] = []
    for name in fieldnames:
        cleaned_name = name.replace("\ufeff", "").strip()
        lowered = cleaned_name.lower()

        if lowered == "segment_global_id":
            cleaned.append("segment_global_id")
        elif lowered == "window_id":
            # TAL window CSVs use window_id instead of segment_id
            cleaned.append("segment_id")
        elif lowered.endswith("segment_id"):
            cleaned.append("segment_id")
        elif lowered == "video_file":
            cleaned.append("video_file")
        elif lowered == "filename":
            cleaned.append("filename")
        elif lowered == "start_sec":
            cleaned.append("start_sec")
        elif lowered == "end_sec":
            cleaned.append("end_sec")
        elif lowered == "annotator_label":
            cleaned.append("annotator_label")
        elif lowered == "rmm_type":
            cleaned.append("rmm_type")
        elif lowered == "child_id":
            cleaned.append("child_id")
        elif lowered == "timepoint":
            cleaned.append("timepoint")
        else:
            cleaned.append(cleaned_name)
    return cleaned


def load_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = normalize_fieldnames(reader.fieldnames)
        yield from reader


def resolve_video_path(video_root: Path, video_file: str) -> Path:
    """
    Convert the CSV video path (often with .mov/.MOV) to the standardized .mp4 path.
    The files live under Videos_from_external_standardized.
    """
    path_str = video_file.replace("\\", "/")
    prefix = "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/"
    if path_str.startswith(prefix):
        path_str = path_str[len(prefix) :]

    relative_path = Path(path_str.lstrip("/"))
    candidates = [
        video_root / relative_path.with_suffix(".mp4"),
        video_root / relative_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find video for '{video_file}' (looked in {candidates})")


def load_parsed_sam3_csv(parsed_csv_path: Path) -> Dict[str, Dict]:
    """
    Load rmm_sam3_parsed.csv rows keyed by FileName, with child ID intervals parsed.
    """
    rows: Dict[str, Dict] = {}
    if not parsed_csv_path.exists():
        print(f"⚠️  Parsed SAM3 CSV not found at {parsed_csv_path}; cropping will fall back.")
        return rows
    with parsed_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            row["row_idx"] = idx
            try:
                row["intervals"] = json.loads(row.get("child_sam3_ids", "[]"))
            except Exception:
                row["intervals"] = []
            rows[row.get("FileName", "")] = row
    return rows


def child_ids_for_time(row: Dict, time_sec: float) -> List[int]:
    ids: List[int] = []
    for iv in row.get("intervals", []):
        start = iv.get("start_sec", 0)
        end = iv.get("end_sec") if iv.get("end_sec") is not None else float("inf")
        try:
            start_f = float(start)
        except Exception:
            start_f = 0.0
        try:
            end_f = float(end)
        except Exception:
            end_f = float("inf")
        if start_f <= time_sec < end_f:
            try:
                ids.append(int(iv["id"]))
            except Exception:
                pass
    return ids


def get_video_info(video_path: Path) -> Dict:
    """Get width/height/fps/frame_count using OpenCV when available, else best-effort ffprobe."""
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
    else:
        info = {"width": 0, "height": 0, "fps": 0.0, "frame_count": 0}
        try:
            probe = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames",
                "-of",
                "csv=p=0",
                str(video_path),
            ]
            res = subprocess.run(probe, capture_output=True, text=True)
            if res.stdout:
                parts = res.stdout.strip().split(",")
                if len(parts) >= 4:
                    info["width"] = int(float(parts[0]))
                    info["height"] = int(float(parts[1]))
                    num, den = parts[2].split("/") if "/" in parts[2] else (parts[2], "1")
                    info["fps"] = float(num) / float(den) if float(den) else 0.0
                    try:
                        info["frame_count"] = int(float(parts[3]))
                    except Exception:
                        info["frame_count"] = 0
        except Exception:
            pass
    info["duration"] = info["frame_count"] / info["fps"] if info.get("fps", 0) else 0
    return info


def get_video_rotation(video_path: Path) -> int:
    """Read rotation side data from the video."""
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.stdout.strip():
            return int(float(result.stdout.strip()))
    except Exception:
        pass
    return 0


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


def load_mask_cache(
    cache_path: Optional[Path],
    video_basename: Optional[str] = None,
    prompt: str = DEFAULT_MASK_PROMPT,
    model_name: str = DEFAULT_MASK_MODEL,
) -> Optional[Dict]:
    """
    Port of visualize_sam3_crops.ipynb logic to resolve mask caches.

    Resolution order:
        1) If cache_path is a file, use it.
        2) If cache_path is a directory, prefer `<model>__prompt-<slug>.h5`, else first .h5.
        3) If still unresolved, build the path using the MaskCacheManager layout:
           `<MASK_CACHE_BASE>/masks/<video_basename>/<model>__prompt-<slug>.h5`.
           Also tries `<video_basename>_segmented` and `<video_basename>_noviz`.
    """
    if h5py is None or np is None:
        return None

    def _slugify(text: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text.strip())
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug.strip("-") or "none"

    prompt_slug = _slugify(prompt)
    candidates: List[Path] = []

    # 1) Explicit path provided
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.is_file():
            candidates.append(cache_path)
        elif cache_path.is_dir():
            candidates.append(cache_path / f"{model_name}__prompt-{prompt_slug}.h5")
            candidates.extend(sorted(cache_path.glob("*.h5")))

    # 2) Paths derived from MaskCacheManager layout
    def add_from_basename(basename: str) -> None:
        base_dir = MASK_CACHE_BASE / "masks" / basename
        candidates.append(base_dir / f"{model_name}__prompt-{prompt_slug}.h5")
        candidates.extend(sorted(base_dir.glob("*.h5")))

    if video_basename:
        add_from_basename(video_basename)
        if not video_basename.endswith("_segmented"):
            add_from_basename(f"{video_basename}_segmented")
        if not video_basename.endswith("_noviz"):
            add_from_basename(f"{video_basename}_noviz")

    chosen = None
    for cand in candidates:
        if cand and cand.exists() and cand.is_file():
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
                print(f"⚠️  No frame groups found in cache {chosen}")
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
    except Exception as e:
        print(f"Error loading cache {chosen}: {e}")
        return None


def rotation_filter_for_degrees(rotation: int) -> Optional[str]:
    """Map rotation degrees to an ffmpeg filter string mirroring numpy rot90 usage."""
    if rotation in (90, -270):
        return "transpose=2"  # 90 CCW
    if rotation in (-90, 270):
        return "transpose=1"  # 90 CW
    if rotation in (180, -180):
        return "transpose=2,transpose=2"
    return None


def compute_crop(task: ClipTask, crop_cfg: CropConfig, sam3_rows: Dict[str, Dict]) -> CropResult:
    """Resolve crop box for a clip using SAM3 mask cache and label intervals."""
    if not crop_cfg.enabled:
        return CropResult(False, "cropping disabled", None, [], None, None, None, None)
    if np is None or h5py is None:
        return CropResult(False, "numpy/h5py unavailable", None, [], None, None, None, None)

    filename = task.metadata.get("filename") or Path(task.metadata.get("video_file", "")).name
    sam3_row = sam3_rows.get(filename)
    if not sam3_row:
        return CropResult(False, f"no SAM3 row for {filename}", None, [], None, None, None, None)

    video_info = get_video_info(task.source)
    fps = video_info.get("fps", 0) or 0
    if fps <= 0:
        return CropResult(False, "fps unavailable", None, [], None, None, None, None)

    frame_idx = int(round(task.start * fps))
    frame_time = task.start

    rotation_meta = get_video_rotation(task.source) * -1 if crop_cfg.apply_rotation_fix else 0
    rotation_to_use = crop_cfg.rotation_override if crop_cfg.rotation_override is not None else rotation_meta

    stem = Path(sam3_row.get("FileName", filename)).stem
    video_basename = f"{stem}_segmented"
    cache = load_mask_cache(None, video_basename=video_basename, prompt=crop_cfg.mask_prompt, model_name=crop_cfg.mask_model)
    if not cache:
        return CropResult(False, f"no mask cache for {video_basename}", None, [], frame_idx, None, None, rotation_to_use)

    attrs = cache.get("attrs") or {}
    cache_width = attrs.get("width")
    cache_height = attrs.get("height")
    base_width = cache_width or video_info.get("width", 0)
    base_height = cache_height or video_info.get("height", 0)
    if base_width == 0 or base_height == 0:
        return CropResult(False, "missing base dimensions", None, [], frame_idx, None, cache.get("path"), rotation_to_use)

    child_ids = child_ids_for_time(sam3_row, frame_time)

    frame_indices = cache["frame_indices"]
    closest_pos = int(np.argmin(np.abs(frame_indices - frame_idx)))
    closest_frame_idx = int(frame_indices[closest_pos])
    boxes = cache["boxes"][closest_frame_idx]
    obj_ids = cache["obj_ids"][closest_frame_idx]
    scores = cache["scores"][closest_frame_idx]
    if boxes is None or len(boxes) == 0:
        return CropResult(False, "no boxes in cache", None, [], frame_idx, closest_frame_idx, cache.get("path"), rotation_to_use)

    boxes_rotated = boxes.copy()
    expected_w, expected_h = base_width, base_height
    if rotation_to_use:
        boxes_rotated = rotate_boxes(boxes_rotated, rotation_to_use, base_width, base_height)
        if rotation_to_use in (-90, 90, -270, 270):
            expected_w, expected_h = base_height, base_width
        elif rotation_to_use in (-180, 180):
            expected_w, expected_h = base_width, base_height

    # Compute target frame dimensions after rotation so we can scale boxes to what ffmpeg will see.
    frame_w, frame_h = video_info.get("width", 0), video_info.get("height", 0)
    if rotation_to_use in (-90, 90, -270, 270):
        frame_w, frame_h = frame_h, frame_w
    elif rotation_to_use in (-180, 180):
        frame_w, frame_h = frame_w, frame_h

    if expected_w and expected_h and (frame_w != expected_w or frame_h != expected_h):
        scale_x = frame_w / expected_w
        scale_y = frame_h / expected_h
        boxes_rotated[:, [0, 2]] *= scale_x
        boxes_rotated[:, [1, 3]] *= scale_y

    matches = []
    if child_ids:
        matches = [i for i, obj in enumerate(obj_ids) if int(obj) in child_ids]
    matched_ids = [int(obj_ids[i]) for i in matches] if matches else []

    if matches:
        xs: List[float] = []
        ys: List[float] = []
        for det_idx in matches:
            x1, y1, x2, y2 = boxes_rotated[det_idx]
            xs.extend([x1, x2])
            ys.extend([y1, y2])
    else:
        xs = []
        ys = []

    if not xs or not ys:
        return CropResult(False, "no matching child IDs", None, matched_ids, frame_idx, closest_frame_idx, cache.get("path"), rotation_to_use)

    x1 = max(0, int(np.floor(min(xs) - crop_cfg.padding)))
    y1 = max(0, int(np.floor(min(ys) - crop_cfg.padding)))
    x2 = min(int(frame_w), int(np.ceil(max(xs) + crop_cfg.padding)))
    y2 = min(int(frame_h), int(np.ceil(max(ys) + crop_cfg.padding)))

    if x2 <= x1 or y2 <= y1:
        return CropResult(False, "invalid crop box after scaling", None, matched_ids, frame_idx, closest_frame_idx, cache.get("path"), rotation_to_use)

    return CropResult(
        True,
        "crop ok",
        (x1, y1, x2, y2),
        matched_ids,
        frame_idx,
        closest_frame_idx,
        cache.get("path"),
        rotation_to_use,
    )


def build_tasks(csv_path: Path, video_root: Path, output_root: Path, min_duration: float) -> List[ClipTask]:
    tasks: List[ClipTask] = []
    for row in load_rows(csv_path):
        segment_id = row.get("segment_id") or row.get("segment_global_id")
        if not segment_id:
            print(f"[{csv_path.name}] Skipping row with no segment_id: {row}")
            continue

        try:
            start = float(row["start_sec"])
            end = float(row["end_sec"])
        except Exception as exc:
            print(f"[{csv_path.name}] Skipping {segment_id}: could not parse start/end -> {exc}")
            continue

        if end <= start:
            # Generate a minimal-duration clip instead of skipping zero-length rows.
            end = start + min_duration
            from_zero_length = True
        else:
            from_zero_length = False

        try:
            source = resolve_video_path(video_root, row["video_file"])
        except Exception as exc:
            print(f"[{csv_path.name}] Skipping {segment_id}: {exc}")
            continue

        csv_subdir = csv_path.stem
        output = output_root / csv_subdir / f"{segment_id}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)

        tasks.append(
            ClipTask(
                segment_id=segment_id,
                source=source,
                output=output,
                start=start,
                end=end,
                duration=end - start,
                from_zero_length=from_zero_length,
                metadata={
                    "csv_file": csv_path.name,
                    "video_file": row.get("video_file", ""),
                    "filename": row.get("filename", ""),
                    "child_id": row.get("child_id", ""),
                    "timepoint": row.get("timepoint", ""),
                    "label": row.get("annotator_label", "") or row.get("rmm_type", ""),
                },
            )
        )
    return tasks


def clip_is_valid(path: Path, lenient: bool = False) -> bool:
    """
    Heuristic check that an mp4 is readable (non-empty and ffprobe succeeds).
    
    Args:
        path: Path to the clip file to validate
        lenient: If True, use OpenCV as fallback if decord fails (useful for low FPS videos)
    """
    try:
        if not path.exists():
            return False
        # Resolve symlink targets; broken links are invalid
        if path.is_symlink():
            try:
                _ = path.resolve(strict=True)
            except FileNotFoundError:
                return False
        if path.stat().st_size == 0:
            return False
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(probe_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            return False

        # Optionally validate that decord can read at least one frame. This helps catch
        # files that pass ffprobe but still break the DataLoader.
        if _HAVE_DECORD:
            try:
                vr = VideoReader(str(path), ctx=cpu(0))  # type: ignore[operator]
                if len(vr) == 0:
                    return False
                _ = vr[0]
            except Exception:
                if lenient and cv2 is not None:
                    # Fallback to OpenCV validation for low FPS videos that decord can't handle
                    try:
                        cap = cv2.VideoCapture(str(path))
                        if cap.isOpened():
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            ret = frame_count > 0
                            cap.release()
                            if ret:
                                return True
                    except Exception:
                        pass
                return False

        return True
    except Exception:
        return False


def run_ffmpeg(
    task: ClipTask,
    codec: str,
    overwrite: bool,
    include_end_second: bool,
    end_second_pad: float,
    target_path: Path,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    rotation: Optional[int] = None,
    verbose_ffmpeg: bool = False,
    lenient_validation: bool = False,
) -> bool:
    if target_path.exists() and not overwrite:
        if clip_is_valid(target_path, lenient=lenient_validation):
            print(f"[skip] {task.segment_id} -> {target_path} (already exists)")
            return True
        else:
            print(f"[rebuild-corrupt] {task.segment_id} -> {target_path} (existing file unreadable)")

    duration = max(task.duration, 0.0)
    # Include the full final second if requested (and if not a zero-length-imputed clip).
    if include_end_second and not task.from_zero_length:
        duration += end_second_pad

    filters: List[str] = []
    rotation_filter = rotation_filter_for_degrees(rotation or 0) if rotation else None
    if rotation_filter:
        filters.append(rotation_filter)
    if crop_box:
        x1, y1, x2, y2 = crop_box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        filters.append(f"crop={w}:{h}:{x1}:{y1}")

    loglevel = "info" if verbose_ffmpeg else "error"
    ffmpeg_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", loglevel]
    if rotation_filter:
        ffmpeg_cmd.append("-noautorotate")
    
    # Use accurate seeking: -ss after -i for frame-accurate seeking (slower but more reliable)
    # This is especially important for low FPS videos where keyframe seeking can fail
    ffmpeg_cmd += [
        "-i",
        str(task.source),
        "-ss",
        f"{task.start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-avoid_negative_ts",
        "make_zero",
    ]

    if codec == "copy":
        ffmpeg_cmd += ["-c", "copy"]
    else:
        ffmpeg_cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
        ]

    if filters:
        ffmpeg_cmd += ["-vf", ",".join(filters)]

    ffmpeg_cmd.append(str(target_path))

    try:
        if verbose_ffmpeg:
            result = subprocess.run(ffmpeg_cmd, check=True, text=True, capture_output=True)
            if result.stdout:
                print(f"[ffmpeg] {task.segment_id} stdout:\n{result.stdout}")
            if result.stderr:
                print(f"[ffmpeg] {task.segment_id} stderr:\n{result.stderr}")
        else:
            subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.DEVNULL)
        if clip_is_valid(target_path, lenient=lenient_validation):
            print(f"[ok] {task.segment_id} ({task.duration:.2f}s) -> {target_path}")
            return True
        print(f"[fail] {task.segment_id}: output failed validation (ffprobe/decord)")
        return False
    except subprocess.CalledProcessError as exc:
        if verbose_ffmpeg:
            print(f"[fail] {task.segment_id}: ffmpeg error -> {exc}")
            if hasattr(exc, 'stdout') and exc.stdout:
                print(f"[ffmpeg] stdout: {exc.stdout}")
            if hasattr(exc, 'stderr') and exc.stderr:
                print(f"[ffmpeg] stderr: {exc.stderr}")
        else:
            print(f"[fail] {task.segment_id}: ffmpeg error -> {exc}")
        return False


def ensure_link_or_copy(src: Path, dst: Path, overwrite: bool, lenient_validation: bool = False) -> bool:
    """Create dst pointing to src via symlink; copy if symlink fails."""
    if dst.exists() or dst.is_symlink():
        if not overwrite and clip_is_valid(dst, lenient=lenient_validation):
            return True
        try:
            dst.unlink()
        except Exception as exc:
            print(f"[fail] could not remove existing {dst}: {exc}")
            return False
    try:
        dst.symlink_to(src)
        return True
    except OSError:
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as exc:
            print(f"[fail] could not link/copy {src} -> {dst}: {exc}")
            return False


def write_manifest(manifest_path: Path, rows: List[ClipTask]) -> None:
    fieldnames = [
        "csv_file",
        "segment_id",
        "clip_path",
        "source_video",
        "start_sec",
        "end_sec",
        "duration",
        "child_id",
        "timepoint",
        "label",
        "crop_applied",
        "crop_box",
        "crop_padding",
        "crop_matched_ids",
        "crop_frame_idx",
        "crop_cache_frame",
        "crop_cache_path",
        "crop_rotation",
        "crop_reason",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "csv_file": row.metadata.get("csv_file", ""),
                    "segment_id": row.segment_id,
                    "clip_path": str(row.output),
                    "source_video": str(row.source),
                    "start_sec": f"{row.start:.3f}",
                    "end_sec": f"{row.end:.3f}",
                    "duration": f"{row.duration:.3f}",
                    "child_id": row.metadata.get("child_id", ""),
                    "timepoint": row.metadata.get("timepoint", ""),
                    "label": row.metadata.get("label", ""),
                    "crop_applied": row.metadata.get("crop_applied", ""),
                    "crop_box": row.metadata.get("crop_box", ""),
                    "crop_padding": row.metadata.get("crop_padding", ""),
                    "crop_matched_ids": row.metadata.get("crop_matched_ids", ""),
                    "crop_frame_idx": row.metadata.get("crop_frame_idx", ""),
                    "crop_cache_frame": row.metadata.get("crop_cache_frame", ""),
                    "crop_cache_path": row.metadata.get("crop_cache_path", ""),
                    "crop_rotation": row.metadata.get("crop_rotation", ""),
                    "crop_reason": row.metadata.get("crop_reason", ""),
                }
            )


def discover_csvs(csv_dir: Path, csv_files: Sequence[str]) -> List[Path]:
    if csv_files:
        return [Path(p) for p in csv_files]
    return sorted(csv_dir.glob("fold_*_*.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut clips for V-JEPA finetuning from actreg CSVs.")
    parser.add_argument(
        "--csv-files",
        nargs="+",
        help="Explicit CSV files to process. If omitted, we'll process all fold_*_*.csv in --csv-dir.",
    )
    parser.add_argument(
        "--csv-dir",
        default="actreg/dataprep/cv_folds",
        help="Directory containing fold CSVs (used when --csv-files is not supplied).",
    )
    parser.add_argument(
        "--video-root",
        default=DEFAULT_VIDEO_ROOT,
        help="Root directory containing Videos_from_external_standardized files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_ROOT,
        help="Where to write the cut clips (one subdir per CSV).",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Enable cropping using SAM3 detections. When enabled, outputs are written to --crop-output-dir.",
    )
    parser.add_argument(
        "--crop-output-dir",
        default=None,
        help="Output root for cropped clips (defaults to <output-dir>_cropped when --crop is set).",
    )
    parser.add_argument(
        "--sam3-parsed-csv",
        default=DEFAULT_PARSED_CSV,
        help="Path to rmm_sam3_parsed.csv with child_sam3_ids intervals.",
    )
    parser.add_argument(
        "--mask-cache-base",
        default=DEFAULT_MASK_CACHE_BASE,
        help="Root directory containing mask caches (MaskCacheManager layout).",
    )
    parser.add_argument(
        "--mask-model",
        default=DEFAULT_MASK_MODEL,
        help="Mask cache model name (prefix of cache filename).",
    )
    parser.add_argument(
        "--mask-prompt",
        default=DEFAULT_MASK_PROMPT,
        help="Mask cache prompt used to resolve cache filename.",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=DEFAULT_PADDING,
        help="Pixels of padding to add around the union of matched boxes.",
    )
    parser.add_argument(
        "--rotation-override",
        type=int,
        default=None,
        help="Force rotation correction in degrees (e.g., 90, -90). If unset, use video metadata.",
    )
    parser.add_argument(
        "--crop-fallback",
        choices=["full", "skip"],
        default="full",
        help="Behavior when no crop is resolved: full = cut full frame; skip = mark as failure.",
    )
    parser.add_argument(
        "--codec",
        choices=["h264", "copy"],
        default="h264",
        help="Encoding mode: h264 re-encodes for precise cuts; copy is faster but depends on source keyframes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate clips even if the output file already exists.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="If end_sec <= start_sec, extend the clip to at least this many seconds (default 1.0).",
    )
    parser.add_argument(
        "--include-end-second",
        dest="include_end_second",
        action="store_true",
        help="Include all frames in the final whole second (start/end inclusive behavior).",
    )
    parser.add_argument(
        "--no-include-end-second",
        dest="include_end_second",
        action="store_false",
        help="Treat end_sec as exclusive (half-open interval).",
    )
    parser.set_defaults(include_end_second=True)
    parser.add_argument(
        "--end-second-pad",
        type=float,
        default=1.0,
        help="How much time (seconds) to add for the inclusive end second. Default 1.0.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel ffmpeg workers (per CSV). Default 1 (sequential).",
    )
    parser.add_argument(
        "--dedup-by-segment",
        dest="dedup_by_segment",
        action="store_true",
        help="Skip re-encoding if a segment_id has already been cut (across CSVs) and just link to it.",
    )
    parser.add_argument(
        "--no-dedup-by-segment",
        dest="dedup_by_segment",
        action="store_false",
        help="Always cut even if a segment_id was already processed.",
    )
    parser.set_defaults(dedup_by_segment=True)
    parser.add_argument(
        "--canonical-dir",
        default=None,
        help="Directory to store deduplicated canonical clips. Defaults to <output-dir>/canonical_clips when dedup is on.",
    )
    parser.add_argument(
        "--verbose-ffmpeg",
        action="store_true",
        help="Show ffmpeg output (useful for debugging encoding issues with low FPS videos).",
    )
    parser.add_argument(
        "--lenient-validation",
        action="store_true",
        help="Use OpenCV as fallback validation if decord fails (useful for low FPS videos that decord can't handle).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = discover_csvs(Path(args.csv_dir), args.csv_files or [])
    if not csv_paths:
        raise SystemExit("No CSV files found to process.")

    video_root = Path(args.video_root)
    base_output_root = Path(args.output_dir)
    crop_output_root = Path(args.crop_output_dir) if args.crop_output_dir else Path(DEFAULT_CROP_OUTPUT_ROOT)
    output_root = crop_output_root if args.crop else base_output_root
    output_root.mkdir(parents=True, exist_ok=True)

    global MASK_CACHE_BASE
    MASK_CACHE_BASE = Path(args.mask_cache_base)

    crop_cfg = CropConfig(
        enabled=args.crop,
        mask_cache_base=MASK_CACHE_BASE,
        mask_model=args.mask_model,
        mask_prompt=args.mask_prompt,
        padding=args.crop_padding,
        rotation_override=args.rotation_override,
        fallback=args.crop_fallback,
    )
    sam3_rows = load_parsed_sam3_csv(Path(args.sam3_parsed_csv)) if args.crop else {}
    if args.crop and not sam3_rows:
        print("⚠️  Cropping requested but no SAM3 rows loaded; will fall back per --crop-fallback.")

    all_successful_tasks: List[ClipTask] = []
    total = 0
    failures = 0
    dedup_map: Dict[str, Path] = {}
    crop_meta_cache: Dict[str, Dict[str, str]] = {}

    canonical_dir = None
    if args.dedup_by_segment:
        canonical_dir = Path(args.canonical_dir) if args.canonical_dir else output_root / "canonical_clips"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        if not args.overwrite:
            for p in canonical_dir.glob("*.mp4"):
                if clip_is_valid(p, lenient=args.lenient_validation):
                    dedup_map[p.stem] = p

    for csv_path in csv_paths:
        tasks = build_tasks(csv_path, video_root, output_root, min_duration=args.min_duration)
        print(f"\nProcessing {csv_path} ({len(tasks)} clips) with {args.jobs} worker(s)")

        if args.dedup_by_segment:
            # Avoid re-encoding rows we've already seen across CSVs.
            unique_tasks: List[ClipTask] = []
            link_only_tasks: List[ClipTask] = []
            seen = set(dedup_map.keys())
            for task in tasks:
                if task.segment_id in seen:
                    link_only_tasks.append(task)
                else:
                    seen.add(task.segment_id)
                    unique_tasks.append(task)
            tasks_to_process = unique_tasks
        else:
            tasks_to_process = tasks
            link_only_tasks = []

        def _run(task: ClipTask) -> tuple[ClipTask, bool, Path]:
            crop_res: Optional[CropResult] = None
            if crop_cfg.enabled:
                crop_res = compute_crop(task, crop_cfg, sam3_rows)
                task.metadata["crop_padding"] = str(crop_cfg.padding)
                task.metadata["crop_applied"] = "1" if crop_res and crop_res.applied else "0"
                task.metadata["crop_box"] = (
                    "" if not crop_res or not crop_res.box else ":".join(str(v) for v in crop_res.box)
                )
                task.metadata["crop_matched_ids"] = (
                    "" if not crop_res else ",".join(str(i) for i in crop_res.matched_ids)
                )
                task.metadata["crop_frame_idx"] = "" if not crop_res or crop_res.frame_idx is None else str(crop_res.frame_idx)
                task.metadata["crop_cache_frame"] = (
                    "" if not crop_res or crop_res.cache_frame_idx is None else str(crop_res.cache_frame_idx)
                )
                task.metadata["crop_cache_path"] = (
                    "" if not crop_res or not crop_res.cache_path else str(crop_res.cache_path)
                )
                task.metadata["crop_rotation"] = (
                    "" if not crop_res or crop_res.rotation_used is None else str(crop_res.rotation_used)
                )
                task.metadata["crop_reason"] = "" if not crop_res else crop_res.reason
                crop_meta_cache[task.segment_id] = dict(task.metadata)

            if crop_cfg.enabled and crop_res and not crop_res.applied:
                if crop_cfg.fallback == "skip":
                    print(f"[skip] {task.segment_id}: {crop_res.reason}")
                    return task, False, task.output
                else:
                    print(f"[full-frame] {task.segment_id}: {crop_res.reason}")

            target_path = canonical_dir / f"{task.segment_id}.mp4" if canonical_dir else task.output
            ok = run_ffmpeg(
                task,
                codec=args.codec,
                overwrite=args.overwrite,
                include_end_second=args.include_end_second,
                end_second_pad=args.end_second_pad,
                target_path=target_path,
                crop_box=crop_res.box if crop_res and crop_res.applied else None,
                rotation=crop_res.rotation_used if crop_res else None,
                verbose_ffmpeg=args.verbose_ffmpeg,
                lenient_validation=args.lenient_validation,
            )
            return task, ok, target_path

        if args.jobs > 1:
            with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                futures = {ex.submit(_run, task): task for task in tasks_to_process}
                for future in as_completed(futures):
                    task, ok, target_path = future.result()
                    total += 1
                    if ok:
                        if canonical_dir:
                            dedup_map[task.segment_id] = target_path
                            if not ensure_link_or_copy(target_path, task.output, overwrite=args.overwrite, lenient_validation=args.lenient_validation):
                                failures += 1
                                continue
                        all_successful_tasks.append(task)
                    else:
                        failures += 1
        else:
            for task in tasks_to_process:
                total += 1
                _, ok, target_path = _run(task)
                if ok:
                    if canonical_dir:
                        dedup_map[task.segment_id] = target_path
                        if not ensure_link_or_copy(target_path, task.output, overwrite=args.overwrite, lenient_validation=args.lenient_validation):
                            failures += 1
                            continue
                    all_successful_tasks.append(task)
                else:
                    failures += 1

        # Link duplicate rows that were already cut in this or earlier CSVs.
        if link_only_tasks:
            for task in link_only_tasks:
                total += 1
                existing = dedup_map.get(task.segment_id)
                if existing and clip_is_valid(existing, lenient=args.lenient_validation) and ensure_link_or_copy(existing, task.output, overwrite=args.overwrite, lenient_validation=args.lenient_validation):
                    if task.segment_id in crop_meta_cache:
                        task.metadata.update(crop_meta_cache[task.segment_id])
                    all_successful_tasks.append(task)
                else:
                    print(f"[fail] {task.segment_id}: no canonical clip found to link")
                    failures += 1

    manifest_path = output_root / "clip_manifest.csv"
    write_manifest(manifest_path, all_successful_tasks)

    print("\n=== Done ===")
    print(f"Clips succeeded: {len(all_successful_tasks)} / {total}")
    print(f"Failures: {failures}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
