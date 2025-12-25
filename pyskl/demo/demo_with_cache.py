#!/usr/bin/env python
"""
Demo script for skeleton-based action recognition using cached poses.
This script loads pre-computed pose estimations from cache and runs action recognition.
"""

import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import subprocess
import torch
import sys
from scipy.optimize import linear_sum_assignment

# Add path to motion_tracking for cache_manager
sys.path.insert(0, '/orcd/data/satra/001/users/brukew/sailsprep/motion_tracking')
from cache_manager import CacheManager

from pyskl.apis import inference_recognizer, init_recognizer


FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo with cached poses')
    parser.add_argument('video', help='input video file path')
    parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default='https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--cache-base-path',
        type=str,
        default='/orcd/data/satra/001/users/brukew/cache_for_tracking',
        help='Base path for cache storage')
    parser.add_argument(
        '--detection-config',
        default='projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py',
        help='Detection config used for caching')
    parser.add_argument(
        '--pose-config',
        default='configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py',
        help='Pose config used for caching')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.4,
        help='Detection threshold used for caching')
    args = parser.parse_args()
    return args


def frame_extraction(video_path, short_side):
    """Extract frames given video_path."""
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')

    print(f'[DEBUG] Opening video: {video_path}')
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print('[ERROR] Could not open video')
        return [], []

    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'[DEBUG] Total frames in video: {total_frames}')

    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None

    while flag:
        if cnt % 30 == 0:  # Print every 30 frames
            print(f'[DEBUG] Extracting frame {cnt}/{total_frames}')

        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
            print(f'[DEBUG] Resize dimensions: {w}x{h} -> {new_w}x{new_h}')

        frame = mmcv.imresize(frame, (new_w, new_h))
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    vid.release()
    print(f'[DEBUG] Extracted {cnt} frames total')
    return frame_paths, frames


def convert_wholebody_to_coco17(wholebody_kpts):
    """
    Convert wholebody keypoints (133 points) to COCO-17 format.

    Wholebody format: 17 body + 6 feet + 68 face + 42 hands = 133 keypoints
    COCO-17 uses only the first 17 body keypoints.

    Args:
        wholebody_kpts: (133, 3) array with x, y, confidence

    Returns:
        coco17_kpts: (17, 3) array with x, y, confidence
    """
    return wholebody_kpts[:17, :]


def pose_tracking(pose_results, max_tracks=2, thre=30):
    """Track poses across frames using Hungarian matching."""
    def dist_ske(ske1, ske2):
        dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
        diff = np.abs(ske1[:, 2] - ske2[:, 2])
        return np.sum(np.maximum(dist, diff))

    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)

    if num_joints is None:
        return None, None

    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]


def main():
    args = parse_args()

    print('[DEBUG] Starting frame extraction...')
    frame_paths, original_frames = frame_extraction(args.video, args.short_side)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape
    print(f'[DEBUG] Extracted {num_frame} frames')

    # Construct output path for cache lookup (same format as motion_resnet.py)
    # We need to convert the input video path to the cache key format
    video_name = osp.basename(args.video)
    video_dir = osp.dirname(args.video)

    # Extract relative path components for cache key
    # This is a simplified version - adjust based on your actual path structure
    path_parts = args.video.split('/')
    if 'Videos_from_external_standardized' in path_parts:
        idx = path_parts.index('Videos_from_external_standardized')
        local_path = '/' + '/'.join(path_parts[idx+1:])
    else:
        local_path = '/' + video_name

    output_path_stem = local_path[:-4].replace("/", "_")  # Remove .mp4 and replace / with _
    cache_output_path = f"/orcd/data/satra/001/users/brukew/motion_tracking_output/resnet/test/{output_path_stem}.mp4"

    print(f'[DEBUG] Cache lookup path: {cache_output_path}')

    # Initialize cache manager
    cache_manager = CacheManager(
        output_video_path=cache_output_path,
        detection_config=args.detection_config,
        pose_config=args.pose_config,
        detection_confidence_threshold=args.det_score_thr,
        nms_type="strict",
        nms_threshold=0.7,
        bbox_min_height=50,
        bbox_min_width=50,
        cache_base_path=args.cache_base_path
    )

    # Load cached poses
    if not cache_manager.check_pose_cache():
        print(f'[ERROR] No cached poses found at: {cache_manager.pose_cache_path}')
        print('[ERROR] Please run motion_resnet.py on this video first to generate cached poses.')
        return

    print('[DEBUG] Loading cached pose results...')
    pose_cache = cache_manager.load_all_poses()
    print(f'[DEBUG] Loaded {len(pose_cache)} frames with cached poses')

    # Get original video dimensions for scaling
    cap = cv2.VideoCapture(args.video)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print('[ERROR] Could not read video to get original dimensions')
        return

    orig_h, orig_w = first_frame.shape[:2]
    resized_h, resized_w = h, w

    # Calculate scaling factors
    scale_x = resized_w / orig_w
    scale_y = resized_h / orig_h
    print(f'[DEBUG] Original video: {orig_w}x{orig_h}, Resized: {resized_w}x{resized_h}')
    print(f'[DEBUG] Scale factors: x={scale_x:.3f}, y={scale_y:.3f}')

    # Convert cached poses to the format expected by the script and scale coordinates
    pose_results = []
    for i in range(num_frame):
        if i in pose_cache:
            frame_poses = []
            for person in pose_cache[i]:
                # Convert wholebody (133) to COCO-17 keypoints
                coco17_kpts = convert_wholebody_to_coco17(person['keypoints']).copy()

                # Scale keypoint coordinates to match resized frames
                coco17_kpts[:, 0] *= scale_x  # Scale x coordinates
                coco17_kpts[:, 1] *= scale_y  # Scale y coordinates

                frame_poses.append(coco17_kpts)
            pose_results.append(frame_poses)
        else:
            pose_results.append([])

    print(f'[DEBUG] Converted poses to COCO-17 format and scaled to resized dimensions')

    # Load action recognition config and model
    print('[DEBUG] Loading action recognition config...')
    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']

    # Check if using GCN
    GCN_flag = 'GCN' in config.model.type
    print(f'[DEBUG] GCN_flag is set to {GCN_flag}')
    GCN_nperson = None
    if GCN_flag:
        format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
        GCN_nperson = format_op.get('num_person', 2)

    print('[DEBUG] Initializing action recognition model...')
    model = init_recognizer(config, args.checkpoint, args.device)
    print('[DEBUG] Action recognition model loaded')

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    # Prepare fake annotation for inference
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    if GCN_flag:
        # Track poses for GCN models
        print(f'[DEBUG] Using GCN mode with max_tracks={GCN_nperson}')
        tracking_inputs = [[pose for pose in poses] for poses in pose_results]
        print(f'[DEBUG] Created tracking inputs with {len(tracking_inputs)} frames')
        print(f'[DEBUG] Starting pose tracking...')
        keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=GCN_nperson)
        print(f'[DEBUG] Pose tracking complete')
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score
        print(f'[DEBUG] Keypoint shape: {keypoint.shape if keypoint is not None else None}')
    else:
        # For PoseC3D models
        num_person = max([len(x) for x in pose_results])
        num_keypoint = 17  # COCO-17 keypoints
        keypoint = np.zeros((num_person, num_frame, num_keypoint, 2), dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frame, num_keypoint), dtype=np.float16)

        for i, poses in enumerate(pose_results):
            for j, pose in enumerate(poses):
                keypoint[j, i] = pose[:, :2]
                keypoint_score[j, i] = pose[:, 2]

        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score

    # Run action recognition
    if fake_anno['keypoint'] is None:
        action_label = 'No person detected'
    else:
        print('[DEBUG] Running action recognition inference...')
        results = inference_recognizer(model, fake_anno)
        action_label = label_map[results[0][0]]
        print(f'[DEBUG] Predicted action: {action_label}')

    # Visualize results (reusing frames from extraction)
    print('[DEBUG] Creating visualization...')
    vis_frames = []
    for i, frame in enumerate(original_frames):
        vis_frame = frame.copy()

        # Draw skeletons if poses exist
        if i < len(pose_results) and len(pose_results[i]) > 0:
            for pose in pose_results[i]:
                # Draw keypoints
                for kpt in pose:
                    x, y, conf = kpt
                    if conf > 0.3:  # Only draw confident keypoints
                        cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Add action label
        cv2.putText(vis_frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)
        vis_frames.append(vis_frame)

    # Write output video using ffmpeg
    print('[DEBUG] Writing output video...')
    temp_dir = './tmp/output_frames'
    os.makedirs(temp_dir, exist_ok=True)

    for i, frame in enumerate(vis_frames):
        frame_path = osp.join(temp_dir, f'frame_{i:06d}.jpg')
        cv2.imwrite(frame_path, frame)

    # Use vf filter to ensure dimensions are divisible by 2
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', '24',
        '-i', osp.join(temp_dir, 'frame_%06d.jpg'),
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # Pad to even dimensions
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        args.out_filename
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    # Clean up
    shutil.rmtree(temp_dir)
    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)

    print(f'[DEBUG] Processing complete! Output saved: {args.out_filename}')
    print(f'[DEBUG] Predicted action: {action_label}')


if __name__ == '__main__':
    main()
