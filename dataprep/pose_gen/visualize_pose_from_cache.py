#!/usr/bin/env python3
"""
Visualize pose keypoints from SAM3-guided pose cache.

Loads pose cache from HDF5 and overlays keypoints on video frames.
Supports filtering by keypoint confidence threshold.

Usage:
    python visualize_pose_from_cache.py --row-idx 53 --kpt-thresh 0.3
    python visualize_pose_from_cache.py --row-idx 320 --kpt-thresh 0.25 --output custom_output.mp4
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np


# COCO skeleton connections for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# COCO keypoint names (first 17 of wholebody)
COCO_KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Resolve filesystem locations from the repo's single source of truth (config.yaml).
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import PATHS  # noqa: E402

# Default paths
DEFAULT_VIDEO_META_JSON = str(PATHS.repo_root / "dataprep/video_meta.json")
DEFAULT_POSE_CACHE_BASE = str(PATHS.cache_for_tracking)
POSE_CACHE_FILENAME = "dino-5scale_swin-l_8xb2-36e_coco_0.5_td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288_sam3guided.h5"


def load_video_meta(video_meta_path: Path) -> Dict[int, Dict]:
    """Load video metadata and create lookup by row_idx."""
    with open(video_meta_path, 'r') as f:
        data = json.load(f)
    
    lookup = {}
    for rec in data['records']:
        lookup[rec['row_idx']] = rec
    
    return lookup


def find_pose_cache_path(
    row_idx: int,
    video_meta_lookup: Dict[int, Dict],
    pose_cache_base: Path,
) -> Optional[Path]:
    """
    Find pose cache path for a video by row_idx.
    
    Matches the logic from create_sails_annotations.py find_pose_cache().
    """
    if row_idx not in video_meta_lookup:
        return None
    
    video_info = video_meta_lookup[row_idx]
    filename = video_info.get('FileName', '')
    
    # Try altered_basename first (for special cases like IMG_5399)
    altered_basename = video_info.get('altered_basename', '')
    if altered_basename:
        alt_path = pose_cache_base / "pose_sam3" / altered_basename / POSE_CACHE_FILENAME
        if alt_path.exists():
            return alt_path
    
    # Try original_path stem
    original_path = video_info.get('original_path', '')
    if original_path:
        video_basename = Path(original_path).stem
        primary_path = pose_cache_base / "pose_sam3" / video_basename / POSE_CACHE_FILENAME
        if primary_path.exists():
            return primary_path
    
    # Try filename stem (but skip IMG_5399 special case)
    if filename:
        filename_stem = Path(filename).stem
        if filename_stem != "IMG_5399":
            primary_path = pose_cache_base / "pose_sam3" / filename_stem / POSE_CACHE_FILENAME
            if primary_path.exists():
                return primary_path
    
    # Fallback candidates
    if filename:
        filename_stem = Path(filename).stem
        fallback_candidates = [
            pose_cache_base / filename_stem / POSE_CACHE_FILENAME,
            pose_cache_base / f"{filename_stem}_pose.h5",
            pose_cache_base / f"{filename_stem}.h5",
        ]
        for candidate in fallback_candidates:
            if candidate.exists():
                return candidate
    
    return None


def load_pose_cache(cache_path: Path) -> Optional[Dict[int, List[Dict]]]:
    """
    Load pose results from HDF5 cache.
    
    Returns:
        Dictionary mapping frame_idx to list of pose dicts:
        {
            frame_idx: [
                {'keypoints': ndarray (num_kpts, 3), 'bbox': ndarray (4,)},
                ...
            ],
            ...
        }
    """
    if not cache_path.exists():
        return None
    
    poses = {}
    try:
        with h5py.File(cache_path, 'r') as f:
            for frame_key in f.keys():
                if not frame_key.startswith('frame_'):
                    continue
                
                frame_idx = int(frame_key.split('_')[1])
                frame_group = f[frame_key]
                
                frame_poses = []
                for pose_key in sorted(frame_group.keys()):
                    if not pose_key.startswith('pose_'):
                        continue
                    
                    pose_group = frame_group[pose_key]
                    frame_poses.append({
                        'keypoints': pose_group['keypoints'][:],
                        'bbox': pose_group['bbox'][:] if 'bbox' in pose_group else None,
                    })
                
                if frame_poses:
                    poses[frame_idx] = frame_poses
        
        return poses
    except Exception as e:
        print(f"Error loading pose cache: {e}")
        return None


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray,
                  conf_thresh: float = 0.3,
                  kpt_color: Tuple[int, int, int] = (0, 255, 0),
                  link_color: Tuple[int, int, int] = (255, 128, 0),
                  radius: int = 3,
                  thickness: int = 2):
    """
    Draw skeleton with keypoints and connections on a frame.
    
    Args:
        frame: BGR image to draw on (modified in place)
        keypoints: Array of shape (num_keypoints, 3) with (x, y, score)
        conf_thresh: Minimum confidence to draw
        kpt_color: BGR color for keypoints
        link_color: BGR color for skeleton links
        radius: Keypoint circle radius
        thickness: Line thickness for skeleton
    """
    # Extract first 17 keypoints if wholebody (133 keypoints)
    if keypoints.shape[0] > 17:
        keypoints = keypoints[:17]
    
    # Draw skeleton links
    for start_idx, end_idx in COCO_SKELETON:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_kpt = keypoints[start_idx]
            end_kpt = keypoints[end_idx]
            
            if len(start_kpt) >= 3 and len(end_kpt) >= 3:
                start_score = start_kpt[2]
                end_score = end_kpt[2]
                
                if start_score > conf_thresh and end_score > conf_thresh:
                    start_pt = (int(start_kpt[0]), int(start_kpt[1]))
                    end_pt = (int(end_kpt[0]), int(end_kpt[1]))
                    cv2.line(frame, start_pt, end_pt, link_color, thickness, lineType=cv2.LINE_AA)
    
    # Draw keypoints
    for idx, kpt in enumerate(keypoints):
        if len(kpt) >= 3:
            x, y, score = kpt[0], kpt[1], kpt[2]
            if score > conf_thresh:
                center = (int(x), int(y))
                cv2.circle(frame, center, radius, kpt_color, -1, lineType=cv2.LINE_AA)
                # Optionally draw keypoint label
                # label = f"{COCO_KPT_NAMES[idx]}:{score:.2f}"
                # cv2.putText(frame, label, (center[0] + 6, center[1] - 6),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def visualize_pose_video(
    video_path: Path,
    pose_cache: Dict[int, List[Dict]],
    output_path: Path,
    kpt_thresh: float = 0.3,
):
    """
    Generate video with pose keypoints overlayed.
    
    Args:
        video_path: Path to input video
        pose_cache: Dictionary mapping frame_idx to list of pose dicts
        output_path: Path to output video
        kpt_thresh: Keypoint confidence threshold
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup ffmpeg for encoding
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        "-an",  # No audio
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "18",
        str(output_path),
    ]
    
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    frame_idx = -1
    frames_with_poses = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_idx += 1
            if not ret:
                break
            
            # Draw poses for this frame
            if frame_idx in pose_cache:
                frames_with_poses += 1
                for pose in pose_cache[frame_idx]:
                    keypoints = pose['keypoints']
                    draw_skeleton(frame, keypoints, conf_thresh=kpt_thresh)
            
            # Write frame to ffmpeg
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                raise RuntimeError("ffmpeg exited early while writing frames")
    
    finally:
        cap.release()
        if proc.stdin:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        rc = proc.wait()
        if rc != 0:
            err = None
            if proc.stderr:
                try:
                    err = proc.stderr.read().decode("utf-8", errors="ignore")
                except Exception:
                    err = None
            raise RuntimeError(f"ffmpeg failed with code {rc}" + (f": {err[:500]}" if err else ""))
    
    print(f"Processed {frame_idx + 1} frames, {frames_with_poses} frames with pose data")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize pose keypoints from SAM3-guided pose cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--row-idx",
        type=int,
        required=True,
        help="Row index from video_meta.json to visualize"
    )
    parser.add_argument(
        "--kpt-thresh",
        type=float,
        default=0.3,
        help="Keypoint confidence threshold for visualization"
    )
    parser.add_argument(
        "--video-meta",
        type=Path,
        default=Path(DEFAULT_VIDEO_META_JSON),
        help="Path to video_meta.json"
    )
    parser.add_argument(
        "--pose-cache-base",
        type=Path,
        default=Path(DEFAULT_POSE_CACHE_BASE),
        help="Base directory for pose caches"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output video path (default: <video_basename>_pose_vis.mp4 in current directory)"
    )
    
    args = parser.parse_args()
    
    # Load video metadata
    print(f"Loading video metadata from {args.video_meta}...")
    video_meta_lookup = load_video_meta(args.video_meta)
    
    if args.row_idx not in video_meta_lookup:
        print(f"Error: row_idx {args.row_idx} not found in video_meta.json")
        print(f"Available row indices: {sorted(video_meta_lookup.keys())[:10]}...")
        sys.exit(1)
    
    video_info = video_meta_lookup[args.row_idx]
    video_path = Path(video_info['original_path'])
    video_basename = Path(video_info['original_path']).stem
    
    print(f"Video: {video_info['FileName']}")
    print(f"Path: {video_path}")
    print(f"Basename: {video_basename}")
    
    # Resolve pose cache path using the same logic as create_sails_annotations.py
    print(f"Finding pose cache path...")
    pose_cache_path = find_pose_cache_path(
        row_idx=args.row_idx,
        video_meta_lookup=video_meta_lookup,
        pose_cache_base=args.pose_cache_base,
    )
    
    if pose_cache_path is None:
        print(f"Error: Could not find pose cache for row_idx {args.row_idx}")
        print(f"  Tried looking up: FileName={video_info.get('FileName')}, "
              f"altered_basename={video_info.get('altered_basename')}, "
              f"original_path={video_info.get('original_path')}")
        sys.exit(1)
    
    print(f"Pose cache path: {pose_cache_path}")
    
    # Load pose cache
    print("Loading pose cache...")
    pose_cache = load_pose_cache(pose_cache_path)
    
    if pose_cache is None or len(pose_cache) == 0:
        print("Error: Pose cache is empty or could not be loaded")
        sys.exit(1)
    
    print(f"Loaded poses for {len(pose_cache)} frames")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = Path.cwd() / f"{video_basename}_pose_vis.mp4"
    
    print(f"Generating visualization with kpt_thresh={args.kpt_thresh}...")
    print(f"Output: {output_path}")
    
    # Generate video
    visualize_pose_video(
        video_path=video_path,
        pose_cache=pose_cache,
        output_path=output_path,
        kpt_thresh=args.kpt_thresh,
    )
    
    print(f"✓ Successfully created visualization: {output_path}")


if __name__ == "__main__":
    main()

