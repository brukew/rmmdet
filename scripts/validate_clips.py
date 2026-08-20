#!/usr/bin/env python3
"""
Validate all video clips using Decord's VideoReader.

Tests each clip file to identify which ones will fail during training.
Produces a report of:
- Total clips
- Valid clips
- Failed clips with error messages
- CSV of bad clips for filtering
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import sys

try:
    from decord import VideoReader, cpu
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package. Install with: pip install decord numpy")
    sys.exit(1)


# Resolve filesystem locations from the repo's single source of truth (config.yaml).
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import PATHS  # noqa: E402

DEFAULT_CLIPS_ROOT = str(PATHS.vjepa2_finetune_clips)
DEFAULT_CSV_DIR = str(PATHS.repo_root / "dataprep/cv_folds")


def test_clip_with_decord(clip_path: Path, num_frames: int = 16) -> Tuple[bool, str, Dict]:
    """
    Test if a clip can be loaded with Decord VideoReader and sampled.

    Returns:
        (success, error_message, metadata)
    """
    metadata = {
        "path": str(clip_path),
        "exists": clip_path.exists(),
        "size_bytes": 0,
        "total_frames": 0,
        "duration": 0.0,
    }

    if not clip_path.exists():
        return False, "File does not exist", metadata

    try:
        metadata["size_bytes"] = clip_path.stat().st_size
        if metadata["size_bytes"] == 0:
            return False, "File is empty (0 bytes)", metadata
    except Exception as e:
        return False, f"Cannot stat file: {e}", metadata

    # Test VideoReader initialization
    try:
        vr = VideoReader(str(clip_path), ctx=cpu(0))
    except Exception as e:
        return False, f"VideoReader initialization failed: {e}", metadata

    # Get basic video info
    try:
        metadata["total_frames"] = len(vr)
        if metadata["total_frames"] == 0:
            return False, "Video has 0 frames", metadata

        # Get FPS and duration if available
        if hasattr(vr, 'get_avg_fps'):
            fps = vr.get_avg_fps()
            if fps > 0:
                metadata["duration"] = metadata["total_frames"] / fps
    except Exception as e:
        return False, f"Cannot read video metadata: {e}", metadata

    # Test frame sampling (simulate training pipeline)
    try:
        total = len(vr)
        indices = np.round(np.linspace(0, total - 1, num_frames)).astype("int64")
        frames = vr.get_batch(indices).asnumpy()

        # Validate frame shape
        if frames.shape[0] != num_frames:
            return False, f"Expected {num_frames} frames, got {frames.shape[0]}", metadata

        if len(frames.shape) != 4:  # (T, H, W, C)
            return False, f"Invalid frame shape: {frames.shape}", metadata

        metadata["frame_shape"] = str(frames.shape)

    except Exception as e:
        return False, f"Frame sampling failed: {e}", metadata

    return True, "", metadata


def discover_clips_from_csv(csv_dir: Path, clips_root: Path) -> List[Dict]:
    """
    Load all clips referenced in fold CSV files.

    Returns:
        List of dicts with segment_id, csv_file, clip_path, label
    """
    clips = []
    csv_files = sorted(csv_dir.glob("fold_*.csv"))

    print(f"Scanning {len(csv_files)} CSV files in {csv_dir}")

    for csv_path in csv_files:
        stem = csv_path.stem
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Handle BOM and normalize field names
            if reader.fieldnames:
                fieldnames = [fn.strip().replace("\ufeff", "") for fn in reader.fieldnames]
                reader.fieldnames = fieldnames

            for row in reader:
                segment_id = row.get("segment_id") or row.get("segment_global_id")
                label = row.get("rmm_type") or row.get("annotator_label")

                if not segment_id:
                    continue

                clip_path = clips_root / stem / f"{segment_id}.mp4"
                clips.append({
                    "segment_id": segment_id,
                    "csv_file": csv_path.name,
                    "csv_stem": stem,
                    "clip_path": clip_path,
                    "label": label or "",
                })

    return clips


def discover_clips_from_filesystem(clips_root: Path) -> List[Dict]:
    """
    Discover all MP4 clips by walking the filesystem.

    Returns:
        List of dicts with segment_id, csv_stem, clip_path
    """
    clips = []

    print(f"Scanning filesystem under {clips_root}")

    for csv_dir in clips_root.iterdir():
        if not csv_dir.is_dir():
            continue

        # Skip canonical_clips directory
        if csv_dir.name == "canonical_clips":
            continue

        csv_stem = csv_dir.name
        for clip_path in csv_dir.glob("*.mp4"):
            segment_id = clip_path.stem
            clips.append({
                "segment_id": segment_id,
                "csv_file": f"{csv_stem}.csv",
                "csv_stem": csv_stem,
                "clip_path": clip_path,
                "label": "",
            })

    return clips


def validate_all_clips(
    clips: List[Dict],
    num_frames: int = 16,
    verbose: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Validate all clips with Decord.

    Returns:
        (valid_clips, failed_clips)
    """
    valid = []
    failed = []

    total = len(clips)
    print(f"\nValidating {total} clips...")

    for i, clip_info in enumerate(clips, 1):
        clip_path = clip_info["clip_path"]

        if i % 100 == 0 or i == total:
            print(f"  Progress: {i}/{total} ({100*i/total:.1f}%)")

        success, error, metadata = test_clip_with_decord(clip_path, num_frames)

        result = {
            **clip_info,
            "success": success,
            "error": error,
            **metadata,
        }

        if success:
            valid.append(result)
            if verbose:
                print(f"  [OK] {clip_info['segment_id']}")
        else:
            failed.append(result)
            print(f"  [FAIL] {clip_info['segment_id']}: {error}")

    return valid, failed


def write_report(
    output_path: Path,
    valid: List[Dict],
    failed: List[Dict],
    total: int,
):
    """Write validation report to file."""
    with output_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("CLIP VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total clips:   {total}\n")
        f.write(f"Valid clips:   {len(valid)} ({100*len(valid)/total:.1f}%)\n")
        f.write(f"Failed clips:  {len(failed)} ({100*len(failed)/total:.1f}%)\n\n")

        if failed:
            f.write("=" * 80 + "\n")
            f.write("FAILED CLIPS\n")
            f.write("=" * 80 + "\n\n")

            for clip in failed:
                f.write(f"Segment ID:   {clip['segment_id']}\n")
                f.write(f"CSV:          {clip['csv_file']}\n")
                f.write(f"Path:         {clip['path']}\n")
                f.write(f"Exists:       {clip['exists']}\n")
                f.write(f"Size:         {clip['size_bytes']} bytes\n")
                f.write(f"Total frames: {clip['total_frames']}\n")
                f.write(f"Error:        {clip['error']}\n")
                f.write("-" * 80 + "\n")

    print(f"\nReport written to: {output_path}")


def write_csv_reports(
    output_dir: Path,
    valid: List[Dict],
    failed: List[Dict],
):
    """Write CSV files for valid and failed clips."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write failed clips CSV
    failed_csv = output_dir / "failed_clips.csv"
    if failed:
        fieldnames = [
            "segment_id", "csv_file", "csv_stem", "label",
            "path", "exists", "size_bytes", "total_frames",
            "duration", "error"
        ]
        with failed_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(failed)
        print(f"Failed clips CSV: {failed_csv}")

    # Write valid clips CSV
    valid_csv = output_dir / "valid_clips.csv"
    if valid:
        fieldnames = [
            "segment_id", "csv_file", "csv_stem", "label",
            "path", "size_bytes", "total_frames", "duration", "frame_shape"
        ]
        with valid_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(valid)
        print(f"Valid clips CSV:  {valid_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate video clips with Decord VideoReader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate using CSV manifest
  python validate_clips.py --csv-dir actreg/dataprep/cv_folds

  # Validate by scanning filesystem
  python validate_clips.py --scan-filesystem

  # Validate with verbose output
  python validate_clips.py --verbose

  # Custom output directory
  python validate_clips.py --output-dir ./validation_results
        """
    )

    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help="Directory containing fold CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--clips-root",
        type=Path,
        default=DEFAULT_CLIPS_ROOT,
        help="Root directory containing clip subdirectories (default: %(default)s)",
    )
    parser.add_argument(
        "--scan-filesystem",
        action="store_true",
        help="Discover clips by scanning filesystem instead of CSV files",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to sample per clip (default: 16)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./clip_validation"),
        help="Output directory for validation reports (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print success messages for each valid clip",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("CLIP VALIDATION WITH DECORD")
    print("=" * 80)

    # Discover clips
    if args.scan_filesystem:
        clips = discover_clips_from_filesystem(args.clips_root)
    else:
        clips = discover_clips_from_csv(args.csv_dir, args.clips_root)

    if not clips:
        print("No clips found!")
        return 1

    print(f"Found {len(clips)} clips to validate")

    # Validate clips
    valid, failed = validate_all_clips(
        clips,
        num_frames=args.num_frames,
        verbose=args.verbose,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total clips:   {len(clips)}")
    print(f"Valid clips:   {len(valid)} ({100*len(valid)/len(clips):.1f}%)")
    print(f"Failed clips:  {len(failed)} ({100*len(failed)/len(clips):.1f}%)")

    # Write reports
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "validation_report.txt"
    write_report(report_path, valid, failed, len(clips))
    write_csv_reports(args.output_dir, valid, failed)

    print("\n" + "=" * 80)

    if failed:
        print(f"\n⚠️  {len(failed)} clips failed validation!")
        print("Review failed_clips.csv for details.")
        return 1
    else:
        print("\n✓ All clips passed validation!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
