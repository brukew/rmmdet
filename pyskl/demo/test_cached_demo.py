#!/usr/bin/env python
# Test script to load cached poses and run action recognition

import sys
sys.path.insert(0, '/orcd/data/satra/001/users/brukew/sailsprep/motion_tracking')
from cache_manager import CacheManager

# Video path
video_path = "/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized/M.J._Home_Videos_AMES_S9B3J7I4R1/12-16 month videos/20170802_084834.mp4"

# Construct output path the same way motion_resnet.py does
# VID_LOCAL_PATH[:-4].replace("/", "_")
VID_LOCAL_PATH = "/M.J._Home_Videos_AMES_S9B3J7I4R1/12-16 month videos/20170802_084834.mp4"
output_path_stem = VID_LOCAL_PATH[:-4].replace("/", "_")  # Remove .mp4 and replace / with _
output_video_path = f"/orcd/data/satra/001/users/brukew/motion_tracking_output/resnet/test/{output_path_stem}.mp4"

print(f"Output path for cache: {output_video_path}")

# Initialize cache manager with the EXACT same parameters as motion_resnet.py
cache_manager = CacheManager(
    output_video_path=output_video_path,
    detection_config="projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py",
    pose_config="configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py",
    detection_confidence_threshold=0.4,
    nms_type="strict",
    nms_threshold=0.7,
    bbox_min_height=50,
    bbox_min_width=50,
    cache_base_path="/orcd/data/satra/001/users/brukew/cache_for_tracking"
)

print(f"Video basename: {cache_manager.video_basename}")
print(f"Detection config name: {cache_manager.detection_config_name}")
print(f"Pose config name: {cache_manager.pose_config_name}")
print(f"Looking for pose cache at: {cache_manager.pose_cache_path}")
print(f"\nChecking for pose cache...")

if cache_manager.check_pose_cache():
    print("✓ Pose cache exists!")
    print(f"  Cache file: {cache_manager.pose_cache_path}")

    # Load all cached poses
    pose_cache = cache_manager.load_all_poses()
    print(f"✓ Loaded {len(pose_cache)} frames with cached poses")

    # Show sample frame
    if pose_cache:
        first_frame = min(pose_cache.keys())
        print(f"\nFirst frame with poses: {first_frame}")
        print(f"Number of people detected: {len(pose_cache[first_frame])}")
        if pose_cache[first_frame]:
            first_person = pose_cache[first_frame][0]
            print(f"First person keypoints shape: {first_person['keypoints'].shape}")
            print(f"Keypoint format: x, y, confidence")
            print(f"Sample keypoint (nose): {first_person['keypoints'][0]}")
else:
    print("✗ No pose cache found!")
    print("  Expected cache file:", cache_manager.pose_cache_path)

if cache_manager.check_detection_cache():
    print("\n✓ Detection cache exists!")
else:
    print("\n✗ No detection cache found!")
