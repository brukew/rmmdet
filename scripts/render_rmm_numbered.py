#!/usr/bin/env python3
"""
Render tracked videos for RMM.csv in numbered folders.

This script creates the same numbered folder structure as rmm_numbered/, but only
for videos that have matched tracks in target identification JSON.

Output structure:
    /path/to/output/002_2/4-10-2020.mp4  (row 0 in CSV)
    /path/to/output/003_3/IMG_0983.mp4   (row 1 in CSV)
    etc.

Folders are only created if the video has a matched track.

Usage:
    python render_rmm_numbered.py /path/to/RMM.csv \
        --dirs /path/to/pipeline_outputs \
        --video-dir /path/to/videos \
        --output /path/to/rmm_numbered_target \
        --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# COCO pose skeleton connections (17 keypoints)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

POSE_CONF_THRESHOLD = 0.65

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Track data loaded from H5 file"""
    id: int
    start_frame: int
    end_frame: int
    fps: float
    keypoints: List[Optional[list]]
    bboxes: List[Optional[tuple]]
    frame_numbers: List[int]
    meta: Dict


@dataclass
class VideoConfig:
    """Configuration for a single video to render"""
    video_id: str
    id: str
    coder: str
    filename: str
    track_id: int
    video_path: Optional[Path] = None
    tracking_dir: Optional[Path] = None
    h5_path: Optional[Path] = None
    row_idx: int = 0


def load_frame_group(frame_group: h5py.Group) -> Dict:
    """Load frame data from H5 group"""
    data = {}

    if "bbox" in frame_group:
        bbox_data = frame_group["bbox"][:]
        if bbox_data.size >= 4:
            data["bbox"] = tuple(bbox_data[:4])

    if "keypoints" in frame_group:
        kp_data = frame_group["keypoints"][:]
        if kp_data.size > 0:
            # Reshape to (N, 3) for N keypoints with (x, y, conf)
            if kp_data.ndim == 1 and kp_data.size % 3 == 0:
                kp_data = kp_data.reshape(-1, 3)
            data["keypoints"] = kp_data.tolist()

    return data


def load_track_from_h5(h5_path: Path) -> Track:
    """Load a single tracking HDF5 file into a Track dataclass"""
    with h5py.File(str(h5_path), "r") as f:
        metadata = f["metadata"]
        start_frame = int(metadata.attrs.get("start_frame", 0))
        end_frame = int(metadata.attrs.get("end_frame", start_frame))
        fps = float(metadata.attrs.get("video_fps", 0.0) or 0.0)
        num_frames_attr = int(metadata.attrs.get("num_frames", 0))
        track_id_attr = metadata.attrs.get("track_id")

        if track_id_attr is None:
            # Fallback to file name (e.g., track_0007.h5)
            stem = h5_path.stem
            try:
                track_id_attr = int(stem.split("_")[-1])
            except Exception:
                track_id_attr = -1

        frames = f["frames"]
        frame_entries = sorted(frames.keys())

        frame_numbers: List[int] = []
        bboxes: List[Optional[tuple]] = []
        keypoints: List[Optional[list]] = []

        for frame_key in frame_entries:
            try:
                frame_number = int(frame_key.split("_")[-1])
            except Exception:
                frame_number = len(frame_numbers)

            frame_numbers.append(frame_number)

            frame = frames[frame_key]
            frame_data = load_frame_group(frame)

            bboxes.append(frame_data.get("bbox"))
            keypoints.append(frame_data.get("keypoints"))

        track = Track(
            id=int(track_id_attr),
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            keypoints=keypoints,
            bboxes=bboxes,
            frame_numbers=frame_numbers,
            meta={
                "num_frames": num_frames_attr or len(frame_numbers),
                "video_width": metadata.attrs.get("video_width"),
                "video_height": metadata.attrs.get("video_height"),
                "source_h5": str(h5_path),
            },
        )

    return track


def resolve_track_path(tracking_dir: Path, track_id: int) -> Optional[Path]:
    """Find the HDF5 file for a given track ID inside a tracking directory"""
    candidates = [
        tracking_dir / f"track_{track_id:04d}.h5",
        tracking_dir / f"track_{track_id:05d}.h5",
        tracking_dir / f"track_{track_id}.h5",
    ]

    for cand in candidates:
        if cand.exists():
            return cand

    # Try glob patterns
    glob_matches = sorted(tracking_dir.glob(f"track_{track_id:04d}*.h5"))
    if glob_matches:
        return glob_matches[0]

    glob_matches = sorted(tracking_dir.glob(f"track_{track_id}*.h5"))
    if glob_matches:
        return glob_matches[0]

    return None


def get_video_id(row: pd.Series) -> str:
    """Construct video_id from CSV row: {ID}_{CODER}_{FILENAME_STEM}"""
    id_val = row['ID']
    coder = row['Original_Coder']
    filename = row['FileName']
    filename_stem = Path(filename).stem
    return f"{id_val}_{coder}_{filename_stem}"


def convert_source_path(source_path: str, video_base_dir: Optional[Path]) -> Optional[Path]:
    """Convert a SourceFile entry into an actual filesystem path"""
    if not source_path:
        return None

    source_candidate = Path(source_path)
    if source_candidate.exists():
        return source_candidate

    if video_base_dir:
        # Try direct join with video_base_dir
        rel_candidate = video_base_dir / source_candidate
        if rel_candidate.exists():
            return rel_candidate

        # Try with .mp4 suffix
        rel_candidate_mp4 = rel_candidate.with_suffix('.mp4')
        if rel_candidate_mp4.exists():
            return rel_candidate_mp4

    return None


def find_dir_in_dirs(dirname: str, parent_dirs: List[Path], subdir: str) -> Optional[Path]:
    """Search for a directory in multiple parent directories under a specific subdirectory"""
    for parent in parent_dirs:
        candidate = parent / subdir / dirname
        if candidate.is_dir():
            return candidate
    return None


def load_target_json(parent_dirs: List[Path], child_id: str) -> Optional[Dict]:
    """Load target identification JSON from {parent}/target/results/{ID}/{ID}_target_identification.json"""
    for parent in parent_dirs:
        json_path = parent / "target" / "results" / child_id / f"{child_id}_target_identification.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse JSON at {json_path}: {e}")
                continue
    return None


def get_folder_name(row_idx: int) -> str:
    """Calculate folder name from row index (row 0 -> 002_2)"""
    folder_num = row_idx + 2
    return f"{folder_num:03d}_{folder_num}"


def resolve_video_config(row: pd.Series, row_idx: int, parent_dirs: List[Path], video_base_dir: Optional[Path]) -> Optional[VideoConfig]:
    """Resolve all paths for a video from CSV row and parent directories"""
    child_id = row['ID']
    coder = row['Original_Coder']
    filename = row['FileName']
    source_file = row['SourceFile']
    filename_stem = Path(filename).stem
    video_id = f"{child_id}_{coder}_{filename_stem}"

    # Load target identification JSON
    target_data = load_target_json(parent_dirs, child_id)
    if not target_data:
        logger.debug(f"[Row {row_idx}] [{video_id}] No target identification JSON found")
        return None

    # Find the matched track for this video
    matches = target_data.get('matches', [])
    matched_track = None
    for match in matches:
        if match['video_id'] == video_id:
            matched_track = match
            break

    if not matched_track:
        logger.debug(f"[Row {row_idx}] [{video_id}] No matched track found in target JSON")
        return None

    track_id = matched_track['track_id']

    # Resolve video path using SourceFile and video_base_dir
    video_path = convert_source_path(source_file, video_base_dir)
    if not video_path:
        logger.warning(f"[Row {row_idx}] [{video_id}] Video file not found: {source_file}")
        return None

    # Find tracking directory
    tracking_dirname = f"{video_id}_tracking"
    tracking_dir = find_dir_in_dirs(tracking_dirname, parent_dirs, "tracking")
    if not tracking_dir:
        tracking_dirname = f"{video_id}_tracking_hdf5"
        tracking_dir = find_dir_in_dirs(tracking_dirname, parent_dirs, "tracking")
        if not tracking_dir:
            logger.warning(f"[Row {row_idx}] [{video_id}] Tracking directory not found: {tracking_dirname}")
            return None

    # Find H5 track file
    h5_path = resolve_track_path(tracking_dir, track_id)
    if not h5_path:
        logger.warning(f"[Row {row_idx}] [{video_id}] Track H5 file not found for track_id={track_id}")
        return None

    return VideoConfig(
        video_id=video_id,
        id=child_id,
        coder=coder,
        filename=filename,
        track_id=track_id,
        video_path=video_path,
        tracking_dir=tracking_dir,
        h5_path=h5_path,
        row_idx=row_idx
    )


def render_video(config: VideoConfig, output_folder: Path) -> bool:
    """Render a single video with track overlays to numbered folder"""
    folder_name = get_folder_name(config.row_idx)
    output_dir = output_folder / folder_name
    output_path = output_dir / config.filename

    logger.info(f"[Row {config.row_idx}] [{config.video_id}] Starting render (track_id={config.track_id})")
    logger.info(f"  Output: {output_path}")

    # Load track data
    try:
        track = load_track_from_h5(config.h5_path)
    except Exception as e:
        logger.error(f"[Row {config.row_idx}] [{config.video_id}] Failed to load track H5: {e}")
        return False

    # Build frame data lookup
    frame_data: Dict[int, Dict[str, np.ndarray]] = {}
    for idx in range(len(track.frame_numbers)):
        frame_num = track.frame_numbers[idx]
        if frame_num is None:
            continue
        data: Dict[str, np.ndarray] = {}

        if track.bboxes and idx < len(track.bboxes):
            bbox = track.bboxes[idx]
            if bbox is not None:
                data['bbox'] = np.asarray(bbox, dtype=float)

        if track.keypoints and idx < len(track.keypoints):
            keypoints = track.keypoints[idx]
            if keypoints is not None:
                data['keypoints'] = np.asarray(keypoints, dtype=float)

        if data:
            frame_data[int(frame_num)] = data

    if not frame_data:
        logger.warning(f"[Row {config.row_idx}] [{config.video_id}] No drawable track data, skipping")
        return False

    # Open video
    cv2.setLogLevel(0)  # Suppress OpenCV warnings
    cap = cv2.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        logger.error(f"[Row {config.row_idx}] [{config.video_id}] Cannot open video: {config.video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not width or not height or width <= 0 or height <= 0:
        logger.error(f"[Row {config.row_idx}] [{config.video_id}] Invalid video dimensions: {width}x{height}")
        cap.release()
        return False

    # Test video readability
    test_ret, test_frame = cap.read()
    if not test_ret or test_frame is None:
        logger.error(f"[Row {config.row_idx}] [{config.video_id}] Cannot read frames from video")
        cap.release()
        return False

    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup ffmpeg subprocess
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-crf", "20",
        "-movflags", "+faststart",
        str(output_path)
    ]

    try:
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL
        )
    except FileNotFoundError:
        logger.error(f"[Row {config.row_idx}] [{config.video_id}] ffmpeg not found in PATH")
        cap.release()
        return False

    # Drawing configuration
    color = (0, 255, 0)  # Green
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Render video frame by frame
    frame_count = 0
    max_frames = 100000  # Safety limit
    consecutive_errors = 0
    max_consecutive_errors = 50

    try:
        with tqdm(total=total_frames, desc=f"Row {config.row_idx}: {config.filename}") as pbar:
            while frame_count < max_frames:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"[Row {config.row_idx}] [{config.video_id}] Too many consecutive read errors")
                        break
                    continue

                frame_count += 1
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                # Draw track data if available for this frame
                data = frame_data.get(current_frame)
                if data:
                    try:
                        # Draw bounding box
                        bbox = data.get('bbox')
                        if bbox is not None and bbox.size >= 4:
                            x1, y1, x2, y2 = bbox.astype(int)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label_y = max(y1 - 10, 20)
                            cv2.putText(
                                frame,
                                f"TARGET (Track {config.track_id})",
                                (x1, label_y),
                                font,
                                0.7,
                                color,
                                2,
                                cv2.LINE_AA
                            )

                        # Draw pose keypoints and skeleton
                        keypoints = data.get('keypoints')
                        if keypoints is not None and keypoints.size >= 3:
                            num_points = keypoints.shape[0]

                            # Draw keypoints
                            for idx in range(min(num_points, 17)):
                                x, y, conf = keypoints[idx]
                                if conf >= POSE_CONF_THRESHOLD:
                                    cv2.circle(frame, (int(x), int(y)), 3, color, -1)

                            # Draw skeleton connections
                            for a, b in COCO_SKELETON:
                                if a < num_points and b < num_points:
                                    x1, y1, c1 = keypoints[a]
                                    x2, y2, c2 = keypoints[b]
                                    if c1 >= POSE_CONF_THRESHOLD and c2 >= POSE_CONF_THRESHOLD:
                                        cv2.line(
                                            frame,
                                            (int(x1), int(y1)),
                                            (int(x2), int(y2)),
                                            color,
                                            2
                                        )
                    except Exception as e:
                        logger.warning(f"[Row {config.row_idx}] [{config.video_id}] Error drawing frame {current_frame}: {e}")

                # Write frame to ffmpeg
                try:
                    proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    logger.error(f"[Row {config.row_idx}] [{config.video_id}] FFmpeg terminated unexpectedly at frame {current_frame}")
                    break

                pbar.update(1)

    except Exception as e:
        logger.error(f"[Row {config.row_idx}] [{config.video_id}] Unexpected error in render loop: {e}")
        return False

    finally:
        try:
            cap.release()
        except Exception as e:
            logger.warning(f"[Row {config.row_idx}] [{config.video_id}] Error releasing video capture: {e}")

        if proc:
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=5)
            except Exception as e:
                logger.warning(f"[Row {config.row_idx}] [{config.video_id}] Error closing ffmpeg: {e}")
                try:
                    proc.kill()
                except:
                    pass

    logger.info(f"[Row {config.row_idx}] [{config.video_id}] Rendered successfully: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Render tracked videos for RMM.csv in numbered folders"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to RMM.csv file"
    )
    parser.add_argument(
        "--dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Parent directories to search for tracking/target data"
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized"),
        help="Base directory containing source videos"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/orcd/scratch/bcs/001/sensein/sails/rmm/rmm_numbered_target"),
        help="Output directory for numbered folders"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to process (for testing)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start processing from this row index (default: 0)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.csv_path.exists():
        logger.error(f"CSV file not found: {args.csv_path}")
        sys.exit(1)

    for parent_dir in args.dirs:
        if not parent_dir.exists():
            logger.error(f"Parent directory not found: {parent_dir}")
            sys.exit(1)

    if not args.video_dir.exists():
        logger.error(f"Video directory not found: {args.video_dir}")
        sys.exit(1)

    # Load CSV
    logger.info(f"Loading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    logger.info(f"Found {len(df)} videos in CSV")

    # Apply start and limit
    if args.start > 0:
        df = df.iloc[args.start:]
        logger.info(f"Starting from row {args.start}")

    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited to {len(df)} videos")

    # Process videos sequentially
    success_count = 0
    skip_count = 0
    skipped_rows = []

    for idx, row in df.iterrows():
        row_idx = int(idx)
        logger.info(f"\n--- Processing row {row_idx} ({row_idx - df.index[0] + 1}/{len(df)}) ---")

        # Resolve all paths
        config = resolve_video_config(row, row_idx, args.dirs, args.video_dir)
        if not config:
            skip_count += 1
            skipped_rows.append(row_idx)
            continue

        # Render video
        success = render_video(config, args.output)
        if success:
            success_count += 1
        else:
            skip_count += 1
            skipped_rows.append(row_idx)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total videos processed: {len(df)}")
    logger.info(f"Successfully rendered: {success_count}")
    logger.info(f"Skipped (no match/error): {skip_count}")
    if skipped_rows:
        logger.info(f"Skipped row indices: {', '.join(map(str, skipped_rows[:20]))}")
        if len(skipped_rows) > 20:
            logger.info(f"  ... and {len(skipped_rows) - 20} more")
    logger.info(f"Output directory: {args.output.absolute()}")


if __name__ == "__main__":
    main()
