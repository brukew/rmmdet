#!/usr/bin/env python3
"""
Create trimmed video clips for V-JEPA finetuning using the actreg fold CSVs.

For each row in the CSV we cut the clip between ``start_sec`` and ``end_sec``
from the standardized video file (``Videos_from_external_standardized``) and
write it to the requested output directory with one subfolder per CSV.
"""

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    from decord import VideoReader, cpu  # type: ignore

    _HAVE_DECORD = True
except Exception:
    # Decord is optional; validation will fall back to ffprobe only.
    VideoReader = None
    cpu = None
    _HAVE_DECORD = False


DEFAULT_VIDEO_ROOT = "/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized"
DEFAULT_OUTPUT_ROOT = "/orcd/scratch/bcs/001/sensein/sails/rmm/vjepa2_finetune_clips"


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


def normalize_fieldnames(fieldnames: Sequence[str]) -> List[str]:
    """Return cleaned field names so we survive odd prefixes like 'hm howsegment_id'."""
    cleaned: List[str] = []
    for name in fieldnames:
        cleaned_name = name.replace("\ufeff", "").strip()
        lowered = cleaned_name.lower()

        if lowered == "segment_global_id":
            cleaned.append("segment_global_id")
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
                    "child_id": row.get("child_id", ""),
                    "timepoint": row.get("timepoint", ""),
                    "label": row.get("annotator_label", "") or row.get("rmm_type", ""),
                },
            )
        )
    return tasks


def clip_is_valid(path: Path) -> bool:
    """Heuristic check that an mp4 is readable (non-empty and ffprobe succeeds)."""
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
) -> bool:
    if target_path.exists() and not overwrite:
        if clip_is_valid(target_path):
            print(f"[skip] {task.segment_id} -> {target_path} (already exists)")
            return True
        else:
            print(f"[rebuild-corrupt] {task.segment_id} -> {target_path} (existing file unreadable)")

    duration = max(task.duration, 0.0)
    # Include the full final second if requested (and if not a zero-length-imputed clip).
    if include_end_second and not task.from_zero_length:
        duration += end_second_pad

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{task.start:.3f}",
        "-i",
        str(task.source),
        "-t",
        f"{duration:.3f}",
    ]

    if codec == "copy":
        ffmpeg_cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
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

    ffmpeg_cmd.append(str(target_path))

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        if clip_is_valid(target_path):
            print(f"[ok] {task.segment_id} ({task.duration:.2f}s) -> {target_path}")
            return True
        print(f"[fail] {task.segment_id}: output failed validation (ffprobe/decord)")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"[fail] {task.segment_id}: ffmpeg error -> {exc}")
        return False


def ensure_link_or_copy(src: Path, dst: Path, overwrite: bool) -> bool:
    """Create dst pointing to src via symlink; copy if symlink fails."""
    if dst.exists() or dst.is_symlink():
        if not overwrite and clip_is_valid(dst):
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = discover_csvs(Path(args.csv_dir), args.csv_files or [])
    if not csv_paths:
        raise SystemExit("No CSV files found to process.")

    video_root = Path(args.video_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_successful_tasks: List[ClipTask] = []
    total = 0
    failures = 0
    dedup_map: Dict[str, Path] = {}

    canonical_dir = None
    if args.dedup_by_segment:
        canonical_dir = Path(args.canonical_dir) if args.canonical_dir else output_root / "canonical_clips"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        if not args.overwrite:
            for p in canonical_dir.glob("*.mp4"):
                if clip_is_valid(p):
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
            target_path = canonical_dir / f"{task.segment_id}.mp4" if canonical_dir else task.output
            ok = run_ffmpeg(
                task,
                codec=args.codec,
                overwrite=args.overwrite,
                include_end_second=args.include_end_second,
                end_second_pad=args.end_second_pad,
                target_path=target_path,
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
                            if not ensure_link_or_copy(target_path, task.output, overwrite=args.overwrite):
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
                        if not ensure_link_or_copy(target_path, task.output, overwrite=args.overwrite):
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
                if existing and clip_is_valid(existing) and ensure_link_or_copy(existing, task.output, overwrite=args.overwrite):
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
