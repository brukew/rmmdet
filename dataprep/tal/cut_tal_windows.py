#!/usr/bin/env python3
"""
Cut video clips for TAL windows.

This script cuts 2-second window clips from the TAL window CSVs.
It uses the video_path column directly (standardized mp4 paths) and
creates clips named by window_id.

Usage:
    python cut_tal_windows.py \
        --csv splits_single_4class/train_windows.csv \
        --output-dir /path/to/output/clips \
        --max-clips 100  # Optional: limit for testing

For cropped clips (using SAM3 detections), you can adapt the existing
create_clip_segments.py by providing a window CSV with the window_id
renamed to segment_id.
"""

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional


def load_window_csv(csv_path: Path, max_rows: Optional[int] = None) -> List[Dict]:
    """Load window CSV."""
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(row)
    return rows


def clip_is_valid(path: Path) -> bool:
    """Check if clip exists and is readable."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(probe_cmd, capture_output=True)
    return result.returncode == 0


def cut_clip(
    window: Dict,
    output_dir: Path,
    overwrite: bool = False,
    codec: str = "h264",
) -> tuple:
    """
    Cut a single window clip.
    
    Returns (window_id, success, output_path).
    """
    window_id = window.get("window_id", "")
    video_path = window.get("video_path", "")
    start_sec = float(window.get("start_sec", 0))
    end_sec = float(window.get("end_sec", 0))
    
    if not video_path or not Path(video_path).exists():
        return window_id, False, f"Video not found: {video_path}"
    
    output_path = output_dir / f"{window_id}.mp4"
    
    if output_path.exists() and not overwrite:
        if clip_is_valid(output_path):
            return window_id, True, str(output_path)
    
    duration = end_sec - start_sec
    
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration:.3f}",
        "-avoid_negative_ts", "make_zero",
    ]
    
    if codec == "copy":
        ffmpeg_cmd += ["-c", "copy"]
    else:
        ffmpeg_cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-c:a", "aac",
            "-movflags", "+faststart",
        ]
    
    ffmpeg_cmd.append(str(output_path))
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE)
        if clip_is_valid(output_path):
            return window_id, True, str(output_path)
        return window_id, False, "Output validation failed"
    except subprocess.CalledProcessError as e:
        return window_id, False, f"ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Cut video clips for TAL windows."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to TAL window CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for clips.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Maximum number of clips to cut (for testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing clips.",
    )
    parser.add_argument(
        "--codec",
        choices=["h264", "copy"],
        default="h264",
        help="Video codec (h264 for re-encoding, copy for stream copy).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--filter-positive",
        action="store_true",
        help="Only cut windows with at least one positive label.",
    )
    parser.add_argument(
        "--filter-background",
        action="store_true",
        help="Only cut background windows.",
    )
    args = parser.parse_args()
    
    print(f"Loading windows from {args.csv}")
    windows = load_window_csv(args.csv, args.max_clips)
    print(f"  Loaded {len(windows)} windows")
    
    # Apply filters
    if args.filter_positive:
        windows = [w for w in windows if w.get("is_background", "1") == "0"]
        print(f"  After positive filter: {len(windows)} windows")
    elif args.filter_background:
        windows = [w for w in windows if w.get("is_background", "0") == "1"]
        print(f"  After background filter: {len(windows)} windows")
    
    if not windows:
        print("No windows to process.")
        return
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Cutting clips to {args.output_dir} with {args.jobs} workers...")
    
    success = 0
    failed = 0
    errors = []
    
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                cut_clip, window, args.output_dir, args.overwrite, args.codec
            ): window
            for window in windows
        }
        
        for future in as_completed(futures):
            window_id, ok, result = future.result()
            if ok:
                success += 1
                if success % 100 == 0:
                    print(f"  Progress: {success}/{len(windows)} clips")
            else:
                failed += 1
                errors.append(f"{window_id}: {result}")
    
    print(f"\nDone: {success} success, {failed} failed")
    
    if errors and len(errors) <= 10:
        print("Errors:")
        for err in errors:
            print(f"  {err}")
    elif errors:
        print(f"First 10 errors (of {len(errors)}):")
        for err in errors[:10]:
            print(f"  {err}")


if __name__ == "__main__":
    main()


