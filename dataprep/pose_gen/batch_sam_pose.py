#!/usr/bin/env python3
"""
Batch processing script for SAM3-guided MMPose estimation.

Processes videos from video_meta.json using SAM3 bounding boxes for target IDs.
Supports:
  - Graceful shutdown and resuming from where it left off
  - Configurable row ranges for parallel SLURM jobs
  - Output video generation with pose overlays
  - Pose result caching to HDF5

Usage:
    python batch_sam_pose.py --start-row 0 --end-row 100
    python batch_sam_pose.py --row-indices 45 100 200
    python batch_sam_pose.py --dry-run
"""

import sys
import os
import signal
import time
import argparse
import json
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import subprocess

import cv2
import numpy as np
import h5py
import torch
from tqdm import tqdm
from pycocotools import mask as maskUtils

from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model as init_pose_estimator

# Add parent directory to path for sam3 imports
sys.path.insert(0, str(Path(__file__).parent.parent / "sam3"))

from sam3_crops_utils import (
    load_parsed_csv,
    load_rotation_report,
    get_video_paths,
    load_mask_cache,
    apply_rotation,
    rotate_boxes,
)


# ================== CONFIGURATION ==================
@dataclass
class ModelConfig:
    """Configuration for detection and pose models"""
    detection_config: str = "/orcd/data/satra/002/models/mmdet/dino-5scale_swin-l_8xb2-36e_coco.py"
    detection_checkpoint: str = "/orcd/data/satra/002/models/mmdet/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
    pose_config: str = "/orcd/data/satra/002/models/mmpose/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py"
    pose_checkpoint: str = "/orcd/data/satra/002/models/mmpose/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
    device: str = "cuda:0"


@dataclass 
class ProcessingConfig:
    """Configuration for processing parameters"""
    det_conf_thresh: float = 0.5
    det_min_w: int = 50
    det_min_h: int = 50
    pose_conf: float = 0.25
    box_expand_ratio: float = 0.5  # Expand SAM3 boxes by 50%
    max_frames: Optional[int] = None  # None = process all frames


# Resolve filesystem locations from the repo's single source of truth (config.yaml).
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import PATHS  # noqa: E402


@dataclass
class CacheConfig:
    """Configuration for caching"""
    cache_base_path: str = str(PATHS.cache_for_tracking)
    enable_cache: bool = True
    force_recompute: bool = False


@dataclass
class OutputConfig:
    """Configuration for output"""
    output_base_dir: str = str(PATHS.pipeline_outputs / "sam_pose")
    generate_video: bool = True
    ffmpeg_loglevel: str = "error"


@dataclass
class BatchConfig:
    """Main configuration for batch processing"""
    models: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Video metadata
    video_meta_json: str = "/orcd/data/satra/001/users/brukew/actreg/dataprep/video_meta.json"


# ================== HELPER FUNCTIONS ==================
def build_ffmpeg_writer(output_path: Path, width: int, height: int, fps: float, loglevel: str = "error"):
    """Create an FFmpeg subprocess for writing video frames."""
    cmd = [
        "ffmpeg", "-loglevel", loglevel, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps if fps > 0 else 30), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", 
        "-preset", "veryfast", "-crf", "23",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def expand_box(box: np.ndarray, expand_ratio: float = 0.5, 
               frame_w: Optional[int] = None, frame_h: Optional[int] = None) -> np.ndarray:
    """Expand a bounding box by a ratio, optionally clipping to frame bounds."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    x1_new = x1 - w * expand_ratio / 2
    y1_new = y1 - h * expand_ratio / 2
    x2_new = x2 + w * expand_ratio / 2
    y2_new = y2 + h * expand_ratio / 2
    
    if frame_w is not None and frame_h is not None:
        x1_new = max(0, x1_new)
        y1_new = max(0, y1_new)
        x2_new = min(frame_w, x2_new)
        y2_new = min(frame_h, y2_new)
    
    return np.array([x1_new, y1_new, x2_new, y2_new])


def draw_keypoints(frame_bgr: np.ndarray, res, color: Tuple[int, int, int], 
                   radius: int = 3, conf_thresh: float = 0.3):
    """Draw keypoints from MMPose result onto frame."""
    if not hasattr(res, 'pred_instances') or not hasattr(res.pred_instances, 'keypoints'):
        return

    keypoints_xy = res.pred_instances.keypoints[0]
    keypoint_scores = res.pred_instances.keypoint_scores[0]

    if hasattr(keypoints_xy, 'cpu'):
        keypoints_xy = keypoints_xy.cpu().numpy()
    if hasattr(keypoint_scores, 'cpu'):
        keypoint_scores = keypoint_scores.cpu().numpy()

    for (x, y), score in zip(keypoints_xy, keypoint_scores):
        if score > conf_thresh:
            cv2.circle(frame_bgr, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)


def mmpose_to_pose_dict(pose_res_list: List) -> List[Dict]:
    """Convert MMPose results to dict format for caching."""
    pose_dicts = []
    for res in pose_res_list:
        bbox = res.pred_instances.bboxes[0]
        if hasattr(bbox, 'cpu'):
            bbox = bbox.cpu().numpy()
        
        keypoints_xy = res.pred_instances.keypoints[0]
        keypoint_scores = res.pred_instances.keypoint_scores[0]
        
        if hasattr(keypoints_xy, 'cpu'):
            keypoints_xy = keypoints_xy.cpu().numpy()
        if hasattr(keypoint_scores, 'cpu'):
            keypoint_scores = keypoint_scores.cpu().numpy()
        
        keypoints = np.concatenate([
            keypoints_xy,
            keypoint_scores.reshape(-1, 1)
        ], axis=1)
        
        pose_dicts.append({
            'keypoints': keypoints,
            'bbox': bbox,
            'metadata': {}
        })
    
    return pose_dicts


def decode_masks_for_frame(cache: Dict, frame_idx: int, rotation: int, 
                           frame_w: int, frame_h: int, base_w: int, base_h: int) -> List[np.ndarray]:
    """Decode RLE masks for a frame and apply rotation/scaling."""
    masks_out = []
    if cache is None or 'rles' not in cache or frame_idx not in cache['rles']:
        return masks_out
    
    rles = cache['rles'][frame_idx]
    for rle in rles:
        rle_dict = {"counts": rle.tobytes(), "size": [base_h, base_w]}
        m = maskUtils.decode(rle_dict)
        if m.ndim == 3:
            m = m[:, :, 0]
        if rotation:
            m = apply_rotation(m, rotation)
        if m.shape != (frame_h, frame_w):
            m = cv2.resize(m.astype(np.uint8), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        masks_out.append(m.astype(bool))
    
    return masks_out


def child_ids_for_time(intervals: List[Dict], t: float) -> List[int]:
    """Get child IDs that are active at time t."""
    ids = []
    for iv in intervals:
        start = iv.get('start_sec', 0)
        end = iv.get('end_sec') or float('inf')
        if start <= t < end:
            try:
                ids.append(int(iv['id']))
            except Exception:
                pass
    return ids


# ================== POSE CACHE MANAGEMENT ==================
def get_sam3_pose_cache_path(video_basename: str, config: BatchConfig) -> Path:
    """Generate cache path with SAM3 indicator."""
    cache_dir = Path(config.cache.cache_base_path) / "pose_sam3" / video_basename
    det_name = Path(config.models.detection_config).stem
    pose_name = Path(config.models.pose_config).stem
    filename = f"{det_name}_{config.processing.det_conf_thresh}_{pose_name}_sam3guided.h5"
    return cache_dir / filename


def save_pose_cache(poses_dict: Dict[int, List[Dict]], cache_path: Path, config: BatchConfig):
    """Save pose results to HDF5 cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(cache_path, 'w') as f:
        f.attrs['detection_config'] = config.models.detection_config
        f.attrs['pose_config'] = config.models.pose_config
        f.attrs['sam3_guided'] = True
        
        for frame_idx, pose_list in poses_dict.items():
            frame_group = f.create_group(f'frame_{frame_idx}')
            
            for pose_idx, pose_result in enumerate(pose_list):
                pose_group = frame_group.create_group(f'pose_{pose_idx}')
                pose_group.create_dataset('keypoints', data=pose_result['keypoints'], 
                                         compression='gzip', compression_opts=4)
                pose_group.create_dataset('bbox', data=pose_result['bbox'], 
                                         compression='gzip', compression_opts=4)


def load_pose_cache(cache_path: Path) -> Optional[Dict[int, List[Dict]]]:
    """Load pose results from HDF5 cache."""
    if not cache_path.exists():
        return None
    
    poses = {}
    try:
        with h5py.File(cache_path, 'r') as f:
            for frame_key in f.keys():
                frame_idx = int(frame_key.split('_')[1])
                frame_group = f[frame_key]
                
                frame_poses = []
                for pose_key in sorted(frame_group.keys()):
                    pose_group = frame_group[pose_key]
                    frame_poses.append({
                        'keypoints': pose_group['keypoints'][:],
                        'bbox': pose_group['bbox'][:]
                    })
                poses[frame_idx] = frame_poses
        return poses
    except Exception as e:
        print(f"Error loading pose cache: {e}")
        return None


# ================== BATCH PROCESSOR ==================
class BatchPoseProcessor:
    """Batch processor for SAM3-guided MMPose estimation."""

    def __init__(self, config: BatchConfig, exp_id: Optional[str] = None, 
                 start_row: int = 0, end_row: Optional[int] = None,
                 row_indices: Optional[List[int]] = None):
        self.config = config
        self.exp_id = exp_id
        self.start_row = start_row
        self.end_row = end_row
        self.row_indices = row_indices
        
        self.interrupted = False
        self.current_video = None
        self.completed_videos: Set[int] = set()
        
        # Load video metadata
        with open(config.video_meta_json, 'r') as f:
            video_meta_data = json.load(f)
        self.video_meta_lookup = {rec['row_idx']: rec for rec in video_meta_data['records']}
        self.all_row_indices = [rec['row_idx'] for rec in video_meta_data['records']]
        
        # Load CSV data for intervals
        self.videos_data = load_parsed_csv()
        self.rotation_data = load_rotation_report()
        
        # Build output directory name
        parent_dir = "sam_pose"
        if self.exp_id:
            parent_dir += f"_{self.exp_id}"
        if self.start_row > 0 or self.end_row is not None:
            row_suffix = f"_rows{self.start_row}-{self.end_row if self.end_row is not None else 'end'}"
            parent_dir += row_suffix
        
        self.output_dir = Path(config.output.output_base_dir) / parent_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.output_dir / "processing_progress.json"
        self._load_progress()
        
        # Models (lazy init)
        self.detector = None
        self.pose_model = None

    def _init_models(self):
        """Initialize detection and pose models."""
        if self.detector is not None:
            return
        
        print("Initializing models...")
        self.detector = init_detector(
            self.config.models.detection_config,
            self.config.models.detection_checkpoint,
            device=self.config.models.device
        )
        self.pose_model = init_pose_estimator(
            self.config.models.pose_config,
            self.config.models.pose_checkpoint,
            device=self.config.models.device
        )
        print("Models initialized.")

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on SIGINT/SIGTERM."""
        print(f"\n\nBatch processing interrupted! Saving progress...")
        self.interrupted = True
        
        if self.current_video is not None and self.current_video in self.completed_videos:
            self.completed_videos.remove(self.current_video)
        
        self._save_progress()
        
        print(f"Progress saved to: {self.progress_file}")
        print(f"Completed videos: {len(self.completed_videos)}")
        if self.current_video:
            print(f"Interrupted during row_idx: {self.current_video}")
            print("Resume by running the script again with the same parameters.")
        
        sys.exit(0)

    def _load_progress(self):
        """Load processing progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    self.completed_videos = set(progress_data.get('completed_row_indices', []))
                print(f"Resumed from previous session. {len(self.completed_videos)} videos already completed.")
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")
                self.completed_videos = set()
        else:
            self.completed_videos = set()

    def _save_progress(self):
        """Save processing progress to file."""
        progress_data = {
            'completed_row_indices': list(self.completed_videos),
            'last_updated': datetime.now().isoformat(),
            'config': {
                'video_meta_json': self.config.video_meta_json,
                'start_row': self.start_row,
                'end_row': self.end_row,
                'exp_id': self.exp_id,
            }
        }
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")

    def _get_rows_to_process(self) -> List[int]:
        """Get list of row indices to process based on configuration."""
        if self.row_indices:
            # Process specific rows
            rows = [r for r in self.row_indices if r in self.video_meta_lookup]
        else:
            # Process row range
            rows = self.all_row_indices[self.start_row:self.end_row]
        
        return rows

    def _process_single_video(self, row_idx: int) -> bool:
        """Process a single video by row_idx."""
        video_meta = self.video_meta_lookup.get(row_idx)
        if not video_meta:
            print(f"  ❌ Row {row_idx} not found in video metadata")
            return False
        
        original_path = Path(video_meta.get('original_path', ''))
        if not original_path.exists():
            print(f"  ❌ Video file not found: {original_path}")
            return False
        
        row = self.videos_data[row_idx - 1]
        intervals = row.get('intervals', [])
        
        # Load SAM3 mask cache
        _, sam3_path, cache_path = get_video_paths(row_idx, self.videos_data)
        cache = load_mask_cache(
            cache_path, 
            video_basename=sam3_path.stem if sam3_path else None, 
            prompt="person", 
            model_name="facebook-sam3"
        )
        
        cache_width = cache_height = None
        if cache and cache.get('attrs'):
            cache_width = cache['attrs'].get('width')
            cache_height = cache['attrs'].get('height')
        base_w = cache_width or 0
        base_h = cache_height or 0
        
        # Setup pose caching
        video_basename = original_path.stem
        pose_cache_path = get_sam3_pose_cache_path(video_basename, self.config)
        poses_to_save = {}
        cached_poses = None
        
        if self.config.cache.enable_cache and not self.config.cache.force_recompute:
            cached_poses = load_pose_cache(pose_cache_path)
            if cached_poses:
                print(f"  Loaded {len(cached_poses)} cached pose frames")
        
        # Open video
        cap = cv2.VideoCapture(str(original_path))
        if not cap.isOpened():
            print(f"  ❌ Could not open video: {original_path}")
            return False
        
        fps = video_meta.get('fps', cap.get(cv2.CAP_PROP_FPS))
        if not fps or fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(frame_count, self.config.processing.max_frames) if self.config.processing.max_frames else frame_count
        
        # Get rotation metadata
        rotation_meta = video_meta.get('rotation_meta', 0)
        if rotation_meta:
            rotation_meta = rotation_meta * -1
        
        # Setup output video
        output_path = self.output_dir / "videos" / f"row{row_idx}_{video_basename}_pose.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        
        try:
            for idx in tqdm(range(total), desc=f"row{row_idx}", leave=False):
                if self.interrupted:
                    break
                
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_w, frame_h = frame_rgb.shape[1], frame_rgb.shape[0]
                
                base_width = cache_width or frame_w
                base_height = cache_height or frame_h
                auto_rotated_guess = bool(rotation_meta) and frame_w == base_height and frame_h == base_width
                rotate_frame = bool(rotation_meta) and not auto_rotated_guess
                rotate_boxes_flag = bool(rotation_meta)
                
                if rotate_frame:
                    frame_rgb = apply_rotation(frame_rgb, rotation_meta)
                    frame_w, frame_h = frame_rgb.shape[1], frame_rgb.shape[0]
                
                # Get SAM3 cache boxes for this frame
                boxes = obj_ids = scores = None
                key = idx
                if cache and key in cache.get('boxes', {}):
                    boxes = cache['boxes'][key]
                    obj_ids = cache['obj_ids'][key]
                    scores = cache['scores'][key]
                
                # Rotate/scale boxes
                boxes_rot = None
                if boxes is not None:
                    boxes_rot = boxes.copy()
                    expected_w, expected_h = base_width, base_height
                    if rotate_boxes_flag and rotation_meta:
                        boxes_rot = rotate_boxes(boxes_rot, rotation_meta, base_width, base_height)
                        if rotation_meta in (-90, 90, -270, 270):
                            expected_w, expected_h = base_height, base_width
                        elif rotation_meta in (-180, 180):
                            expected_w, expected_h = base_width, base_height
                    if expected_w and expected_h and (frame_w != expected_w or frame_h != expected_h):
                        scale_x = frame_w / expected_w
                        scale_y = frame_h / expected_h
                        boxes_rot[:, [0, 2]] *= scale_x
                        boxes_rot[:, [1, 3]] *= scale_y
                
                frame_time = idx / fps if fps else 0.0
                target_set = set(int(t) for t in child_ids_for_time(intervals, frame_time) if t is not None)
                
                # Initialize writer if needed
                if writer is None and self.config.output.generate_video:
                    writer = build_ffmpeg_writer(output_path, frame_w, frame_h, fps, 
                                                 loglevel=self.config.output.ffmpeg_loglevel)
                
                # No boxes or no targets - just write frame as-is
                if boxes_rot is None or obj_ids is None:
                    if writer:
                        writer.stdin.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).tobytes())
                    continue
                
                keep = [i for i, oid in enumerate(obj_ids) if int(oid) in target_set]
                if not keep:
                    if writer:
                        writer.stdin.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).tobytes())
                    continue
                
                frame_vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Check if we have cached poses for this frame
                if cached_poses and idx in cached_poses:
                    for cached_pose in cached_poses[idx]:
                        keypoints = cached_pose['keypoints']
                        for (x, y, score) in keypoints:
                            if score > self.config.processing.pose_conf:
                                cv2.circle(frame_vis, (int(x), int(y)), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                else:
                    # Compute poses - build union box over all target boxes and expand
                    xs1 = [boxes_rot[ki][0] for ki in keep]
                    ys1 = [boxes_rot[ki][1] for ki in keep]
                    xs2 = [boxes_rot[ki][2] for ki in keep]
                    ys2 = [boxes_rot[ki][3] for ki in keep]
                    union_box = [min(xs1), min(ys1), max(xs2), max(ys2)]
                    
                    expanded_box = expand_box(
                        np.array(union_box), 
                        expand_ratio=self.config.processing.box_expand_ratio, 
                        frame_w=frame_w, 
                        frame_h=frame_h
                    )
                    
                    # Run pose estimation
                    pose_res = inference_topdown(self.pose_model, frame_vis, expanded_box.reshape(1, 4))
                    
                    for res in pose_res:
                        draw_keypoints(frame_vis, res, (0, 255, 0), conf_thresh=self.config.processing.pose_conf)
                    
                    if self.config.cache.enable_cache:
                        poses_to_save[idx] = mmpose_to_pose_dict(pose_res)
                
                if writer:
                    writer.stdin.write(frame_vis.astype(np.uint8).tobytes())
                
                # Periodic GPU cleanup
                if idx % 200 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Finalize
            if writer is not None:
                writer.stdin.close()
                writer.wait()
            
            cap.release()
            
            # Save pose cache
            if self.config.cache.enable_cache and poses_to_save:
                save_pose_cache(poses_to_save, pose_cache_path, self.config)
                print(f"  Saved pose cache: {pose_cache_path}")
            
            if self.config.output.generate_video:
                print(f"  Saved video: {output_path}")
            
            return True
            
        except Exception as e:
            import traceback
            print(f"  ❌ Error processing row {row_idx}: {e}")
            traceback.print_exc()
            
            if writer is not None:
                try:
                    writer.stdin.close()
                    writer.wait()
                except:
                    pass
            cap.release()
            return False

    def process_all(self):
        """Process all configured videos."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("=" * 60)
        print("SAM3-Guided MMPose Batch Processing")
        print("=" * 60)
        print(f"Video metadata: {self.config.video_meta_json}")
        print(f"Output directory: {self.output_dir}")
        print(f"Cache directory: {self.config.cache.cache_base_path}")
        print()
        
        # Initialize models
        self._init_models()
        
        # Get rows to process
        all_rows = self._get_rows_to_process()
        print(f"Total videos in range: {len(all_rows)}")
        
        # Filter already completed
        remaining_rows = [r for r in all_rows if r not in self.completed_videos]
        print(f"Remaining to process: {len(remaining_rows)}")
        
        if not remaining_rows:
            print("All videos have already been processed!")
            return
        
        start_time = time.time()
        processed_count = 0
        failed_count = 0
        
        for i, row_idx in enumerate(remaining_rows):
            if self.interrupted:
                break
            
            self.current_video = row_idx
            video_meta = self.video_meta_lookup.get(row_idx, {})
            filename = video_meta.get('FileName', 'unknown')
            
            print(f"\n[{i+1}/{len(remaining_rows)}] Processing row {row_idx}: {filename}")
            
            success = self._process_single_video(row_idx)
            
            if success:
                self.completed_videos.add(row_idx)
                processed_count += 1
                print(f"  ✅ Completed")
            else:
                failed_count += 1
            
            # Save progress periodically
            if (i + 1) % 5 == 0:
                self._save_progress()
        
        # Final save
        self._save_progress()
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Videos processed this session: {processed_count}")
        print(f"Videos failed this session: {failed_count}")
        print(f"Total completed videos: {len(self.completed_videos)}")
        print(f"Progress file: {self.progress_file}")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch SAM3-guided MMPose estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process rows 0-100
  python batch_sam_pose.py --start-row 0 --end-row 100

  # Process specific rows
  python batch_sam_pose.py --row-indices 45 100 200 316

  # Process all with experiment ID
  python batch_sam_pose.py --exp-id my_experiment

  # Dry run (check configuration without processing)
  python batch_sam_pose.py --dry-run
        """
    )
    
    parser.add_argument('--start-row', type=int, default=0,
                       help='Start row index (0-based, inclusive)')
    parser.add_argument('--end-row', type=int, default=None,
                       help='End row index (exclusive, None = process to end)')
    parser.add_argument('--row-indices', type=int, nargs='+', default=None,
                       help='Specific row indices to process (overrides start/end)')
    parser.add_argument('--exp-id', type=str, default=None,
                       help='Experiment identifier for output directory')
    parser.add_argument('--output-dir', type=str, 
                       default=str(PATHS.pipeline_outputs / "sam_pose"),
                       help='Base output directory')
    parser.add_argument('--cache-dir', type=str,
                       default=str(PATHS.cache_for_tracking),
                       help='Base cache directory')
    parser.add_argument('--video-meta', type=str,
                       default='/orcd/data/satra/001/users/brukew/actreg/dataprep/video_meta.json',
                       help='Path to video_meta.json')
    parser.add_argument('--no-video', action='store_true',
                       help='Skip video generation (cache poses only)')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Recompute even if cache exists')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames per video (None = all)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for inference')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually processing')
    
    args = parser.parse_args()
    
    # Build configuration
    config = BatchConfig()
    config.output.output_base_dir = args.output_dir
    config.cache.cache_base_path = args.cache_dir
    config.video_meta_json = args.video_meta
    config.output.generate_video = not args.no_video
    config.cache.force_recompute = args.force_recompute
    config.processing.max_frames = args.max_frames
    config.models.device = args.device
    
    # Validate inputs
    if not Path(config.video_meta_json).exists():
        print(f"Error: Video metadata file not found: {config.video_meta_json}")
        sys.exit(1)
    
    # Create processor
    processor = BatchPoseProcessor(
        config=config,
        exp_id=args.exp_id,
        start_row=args.start_row,
        end_row=args.end_row,
        row_indices=args.row_indices,
    )
    
    if args.dry_run:
        print("DRY RUN - Would process the following:")
        rows = processor._get_rows_to_process()
        remaining = [r for r in rows if r not in processor.completed_videos]
        print(f"  Total rows in range: {len(rows)}")
        print(f"  Already completed: {len(processor.completed_videos)}")
        print(f"  Remaining: {len(remaining)}")
        if remaining:
            print(f"  First 10 remaining rows: {remaining[:10]}")
        return
    
    processor.process_all()


if __name__ == "__main__":
    main()

