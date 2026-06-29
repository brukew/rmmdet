#!/usr/bin/env python3
"""
Data Chapter Figures and Tables.

Generates:
- Fig 3.1: Video quality distribution (6 metrics) for 298 RMM videos
- Fig 3.2: Video metadata distribution (timepoint, location, people, context)
- Fig 3.3: Frame progression (full frame -> SAM3 crop -> pose skeleton)
- Table 3.1: Class distribution JSON by CV fold for 4-class and 5-class

Usage:
    python fig_3_data_chapter.py
    python fig_3_data_chapter.py --fig 3.1
    python fig_3_data_chapter.py --fig 3.2
    python fig_3_data_chapter.py --fig 3.3
    python fig_3_data_chapter.py --table 3.1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Resolve filesystem locations from the repo's single source of truth (config.yaml).
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import PATHS  # noqa: E402

# Data paths
ANNOTATIONS_EXCEL = Path("/orcd/data/satra/002/datasets/SAILS/data4analysis/Video Rating Data/SAILS_RATINGS_ALL_8.8.25.xlsx")
RMM_SEGMENTS_CSV = ACTREG_ROOT / "dataprep" / "rmm_segments.csv"
SPLITS_4CLASS_DIR = ACTREG_ROOT / "dataprep" / "tal" / "splits_cv_4class"
SPLITS_5CLASS_DIR = ACTREG_ROOT / "dataprep" / "tal" / "splits_cv_5class"
ORIGINAL_VIDEO_BASE = Path("/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized")
SAM3_OUTPUT_BASE = PATHS.rmm_sam_numbered
POSE_CACHE_BASE = PATHS.cache_for_tracking
VIDEO_META_JSON = ACTREG_ROOT / "dataprep" / "video_meta.json"

# Output paths
OUTPUT_DIR = ACTREG_ROOT / "insights" / "figs"
TABLES_DIR = ACTREG_ROOT / "insights" / "tables"

# COCO skeleton for pose visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Color palette
COLORS = {
    "primary": "#2196F3",
    "secondary": "#FF9800",
    "accent": "#4CAF50",
    "light": "#E3F2FD"
}


def load_annotations() -> pd.DataFrame:
    """Load the annotations Excel file."""
    print("Loading annotations from Excel...")
    df = pd.read_excel(ANNOTATIONS_EXCEL)
    print(f"  Loaded {len(df)} rows")
    return df


def load_rmm_segments() -> pd.DataFrame:
    """Load RMM segments CSV."""
    print("Loading RMM segments...")
    df = pd.read_csv(RMM_SEGMENTS_CSV)
    print(f"  Loaded {len(df)} segments")
    return df


def get_rmm_video_filenames(rmm_df: pd.DataFrame) -> set:
    """Extract unique video filenames from RMM segments."""
    # The 'filename' column contains just the filename (e.g., '4-10-2020.mov')
    filenames = set(rmm_df['filename'].str.lower().unique())
    print(f"  Found {len(filenames)} unique RMM video filenames")
    return filenames


def filter_annotations_to_rmm(annot_df: pd.DataFrame, rmm_filenames: set) -> pd.DataFrame:
    """Filter annotations to only include RMM videos."""
    # Match by filename (case-insensitive)
    annot_df = annot_df.copy()
    annot_df['filename_lower'] = annot_df['FileName'].str.lower()
    filtered = annot_df[annot_df['filename_lower'].isin(rmm_filenames)]
    print(f"  Filtered to {len(filtered)} annotation rows for RMM videos")
    return filtered


# =============================================================================
# Fig 3.1: Video Quality Distribution
# =============================================================================

def plot_fig_3_1(annot_df: pd.DataFrame, rmm_filenames: set, output_path: Path = None):
    """
    Generate Fig 3.1: Video quality distribution for 298 RMM videos.
    
    Creates a 2x3 grid of histograms for the 6 video quality metrics.
    """
    print("\n=== Generating Fig 3.1: Video Quality Distribution ===")
    
    # Filter to RMM videos
    filtered = filter_annotations_to_rmm(annot_df, rmm_filenames)
    
    # Quality columns
    quality_cols = [
        'Video_Quality_Child_Face_Visibility',
        'Video_Quality_Child_Body_Visibility',
        'Video_Quality_Child_Hand_Visibility',
        'Video_Quality_Lighting',
        'Video_Quality_Resolution',
        'Video_Quality_Motion'
    ]
    
    # Nice labels for plots
    labels = [
        'Face Visibility',
        'Body Visibility',
        'Hand Visibility',
        'Lighting',
        'Resolution',
        'Motion Stability'
    ]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, (col, label) in enumerate(zip(quality_cols, labels)):
        ax = axes[idx]
        data = filtered[col].dropna()
        
        # Create histogram
        sns.histplot(
            data, 
            bins=10, 
            kde=False,
            color=COLORS["primary"],
            alpha=0.7,
            ax=ax,
            edgecolor='white',
            linewidth=0.5
        )
        
        ax.set_xlabel('Rating (1-10)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        
        # Add mean and median annotations
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.1f}')
        ax.legend(fontsize=8, loc='upper left')
    
    plt.suptitle('Video Quality Distribution (298 RMM Videos)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = OUTPUT_DIR / "fig_3_1_video_quality_distribution.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    # Also save PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()


# =============================================================================
# Fig 3.2: Video Metadata Distribution
# =============================================================================

def plot_fig_3_2(annot_df: pd.DataFrame, rmm_filenames: set, output_path: Path = None):
    """
    Generate Fig 3.2: Video metadata distribution.
    
    Creates a 1x3 grid showing:
    - (A) Timepoint distribution
    - (B) Location distribution
    - (C) Number of people distribution
    """
    print("\n=== Generating Fig 3.2: Video Metadata Distribution ===")
    
    # Filter to RMM videos
    filtered = filter_annotations_to_rmm(annot_df, rmm_filenames)
    
    # Create figure - 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # (A) Timepoint distribution
    ax = axes[0]
    timepoint_counts = filtered['time_point'].value_counts().sort_index()
    
    # Clean up timepoint labels
    timepoint_labels = [tp.replace('_', ' ').replace('month', 'mo') for tp in timepoint_counts.index]
    
    bars = ax.bar(range(len(timepoint_counts)), timepoint_counts.values, 
                  color=COLORS["primary"], alpha=0.9, edgecolor='white')
    ax.set_xticks(range(len(timepoint_counts)))
    ax.set_xticklabels(timepoint_labels, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Timepoint', fontsize=11)
    ax.set_ylabel('Number of Videos', fontsize=11)
    ax.set_title('(A) Distribution by Timepoint', fontsize=12, fontweight='bold')
    
    # (B) Location distribution
    ax = axes[1]
    location_counts = filtered['Location'].value_counts()
    
    bars = ax.bar(range(len(location_counts)), location_counts.values,
                  color=COLORS["secondary"], alpha=0.9, edgecolor='white')
    ax.set_xticks(range(len(location_counts)))
    ax.set_xticklabels(location_counts.index, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Location', fontsize=11)
    ax.set_ylabel('Number of Videos', fontsize=11)
    ax.set_title('(B) Distribution by Location', fontsize=12, fontweight='bold')
    
    # (C) Number of people distribution
    ax = axes[2]
    # Combine adults and children, handling NaN values
    filtered_copy = filtered.copy()
    filtered_copy['#_adults'] = pd.to_numeric(filtered_copy['#_adults'], errors='coerce').fillna(0)
    filtered_copy['#_children'] = pd.to_numeric(filtered_copy['#_children'], errors='coerce').fillna(0)
    filtered_copy['total_people'] = filtered_copy['#_adults'] + filtered_copy['#_children']
    
    people_counts = filtered_copy['total_people'].value_counts().sort_index()
    
    bars = ax.bar(range(len(people_counts)), people_counts.values,
                  color=COLORS["accent"], alpha=0.9, edgecolor='white')
    ax.set_xticks(range(len(people_counts)))
    ax.set_xticklabels([int(x) for x in people_counts.index], fontsize=10)
    ax.set_xlabel('Number of People in Video', fontsize=11)
    ax.set_ylabel('Number of Videos', fontsize=11)
    ax.set_title('(C) Distribution by Number of People', fontsize=12, fontweight='bold')
    
    plt.suptitle('Video Metadata Distribution (298 RMM Videos)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = OUTPUT_DIR / "fig_3_2_video_metadata_distribution.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()


# =============================================================================
# Fig 3.3: Frame to Pose Progression
# =============================================================================

def find_video_with_good_pose(rmm_df: pd.DataFrame, min_confidence: float = 0.5, prefer_centered: bool = True) -> Optional[Dict]:
    """
    Find an RMM video that has SAM3 output and good pose keypoints.
    
    Args:
        rmm_df: DataFrame with RMM segments
        min_confidence: Minimum keypoint confidence threshold
        prefer_centered: If True, prefer frames where child is centered
    
    Returns video metadata if found, None otherwise.
    """
    # Load video metadata
    if not VIDEO_META_JSON.exists():
        print(f"  Warning: Video meta JSON not found: {VIDEO_META_JSON}")
        return None
    
    with open(VIDEO_META_JSON, 'r') as f:
        video_meta = json.load(f)
    
    # Get unique video files from RMM segments
    rmm_videos = set(rmm_df['filename'].unique())
    
    # Pose cache filename
    pose_cache_filename = "dino-5scale_swin-l_8xb2-36e_coco_0.5_td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288_sam3guided.h5"
    
    # Collect all valid candidates
    candidates = []
    
    # Check each video for pose cache availability
    for record in video_meta.get('records', []):
        filename = record.get('FileName', '')
        
        # Check if this is an RMM video
        if filename not in rmm_videos:
            continue
        
        # Get video stem for pose cache lookup
        video_stem = Path(filename).stem
        
        # Try to find pose cache by video stem
        pose_path = POSE_CACHE_BASE / "pose_sam3" / video_stem / pose_cache_filename
        
        if not pose_path.exists():
            continue
        
        # Get frame dimensions from record (after rotation)
        cache_width = record.get('cache_width', 1920)
        cache_height = record.get('cache_height', 1080)
        
        # Check if pose has good keypoints
        try:
            with h5py.File(pose_path, 'r') as f:
                # Structure is: frame_X/pose_0/keypoints
                frame_keys = sorted([k for k in f.keys() if k.startswith('frame_')],
                                   key=lambda x: int(x.split('_')[1]))
                
                # Search through frames for one with all keypoints visible
                for frame_key in frame_keys[50:300]:  # Skip first 50 frames, check next 250
                    if 'pose_0' not in f[frame_key]:
                        continue
                    
                    pose_group = f[frame_key]['pose_0']
                    if 'keypoints' not in pose_group:
                        continue
                    
                    kpts = pose_group['keypoints'][:]
                    bbox = pose_group['bbox'][:] if 'bbox' in pose_group else None
                    
                    # Check first 17 keypoints (COCO body)
                    if kpts.shape[0] >= 17:
                        body_kpts = kpts[:17, :]
                        confidences = body_kpts[:, 2]
                        
                        if np.all(confidences > min_confidence):
                            frame_idx = int(frame_key.split('_')[1])
                            
                            # Calculate how centered the child is and check bbox validity
                            if bbox is not None:
                                bbox_width = bbox[2] - bbox[0]
                                bbox_height = bbox[3] - bbox[1]
                                bbox_area = bbox_width * bbox_height
                                frame_area = cache_width * cache_height
                                
                                # Skip if bbox covers too much of the frame (not a real crop)
                                coverage = bbox_area / frame_area
                                if coverage > 0.85:  # Skip if bbox is >85% of frame
                                    continue
                                
                                # Skip if bbox is too small (child too far away)
                                if coverage < 0.05:  # Skip if bbox is <5% of frame
                                    continue
                                
                                bbox_center_x = (bbox[0] + bbox[2]) / 2
                                bbox_center_y = (bbox[1] + bbox[3]) / 2
                                
                                # Distance from center (normalized)
                                center_dist_x = abs(bbox_center_x - cache_width / 2) / (cache_width / 2)
                                center_dist_y = abs(bbox_center_y - cache_height / 2) / (cache_height / 2)
                                center_score = 1.0 - (center_dist_x + center_dist_y) / 2
                                
                                # Prefer child taking up 20-50% of frame
                                if 0.15 < coverage < 0.5:
                                    size_score = 1.0
                                elif coverage < 0.15:
                                    size_score = coverage / 0.15
                                else:
                                    size_score = max(0, 1.0 - (coverage - 0.5) / 0.35)
                            else:
                                center_score = 0.5
                                size_score = 0.5
                            
                            candidate = {
                                'filename': filename,
                                'video_stem': video_stem,
                                'frame_idx': frame_idx,
                                'frame_key': frame_key,
                                'pose_cache_path': str(pose_path),
                                'video_path': record.get('original_path', ''),
                                'bbox': bbox.tolist() if bbox is not None else None,
                                'record': record,
                                'center_score': center_score,
                                'size_score': size_score,
                                'combined_score': center_score * 0.7 + size_score * 0.3
                            }
                            candidates.append(candidate)
                            
                            # If we have enough candidates from this video, move on
                            if len([c for c in candidates if c['video_stem'] == video_stem]) >= 3:
                                break
                                
        except Exception as e:
            print(f"  Error reading pose cache {pose_path}: {e}")
            continue
    
    if not candidates:
        return None
    
    # Sort by combined score (higher is better)
    candidates.sort(key=lambda x: x['combined_score'], reverse=True)
    
    print(f"  Found {len(candidates)} candidate frames")
    print(f"  Top candidate: {candidates[0]['filename']} frame {candidates[0]['frame_idx']} (score: {candidates[0]['combined_score']:.2f})")
    
    return candidates[0]


def draw_skeleton(img: np.ndarray, keypoints: np.ndarray, confidence_thresh: float = 0.3) -> np.ndarray:
    """
    Draw pose skeleton on image.
    
    Args:
        img: Input image (BGR)
        keypoints: Array of shape [17, 3] with (x, y, confidence)
        confidence_thresh: Minimum confidence to draw
        
    Returns:
        Image with skeleton overlay
    """
    img = img.copy()
    
    # Colors for different body parts
    colors = {
        'head': (255, 100, 100),      # Blue-ish
        'arms': (100, 255, 100),      # Green-ish  
        'torso': (100, 100, 255),     # Red-ish
        'legs': (255, 255, 100)       # Cyan-ish
    }
    
    # Draw skeleton connections
    for i, (start_idx, end_idx) in enumerate(COCO_SKELETON):
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue
            
        start_kpt = keypoints[start_idx]
        end_kpt = keypoints[end_idx]
        
        if start_kpt[2] < confidence_thresh or end_kpt[2] < confidence_thresh:
            continue
        
        start_pos = (int(start_kpt[0]), int(start_kpt[1]))
        end_pos = (int(end_kpt[0]), int(end_kpt[1]))
        
        # Choose color based on connection type
        if i < 4:
            color = colors['head']
        elif i < 9:
            color = colors['arms']
        elif i < 12:
            color = colors['torso']
        else:
            color = colors['legs']
        
        cv2.line(img, start_pos, end_pos, color, 2, cv2.LINE_AA)
    
    # Draw keypoints
    for idx, kpt in enumerate(keypoints):
        if kpt[2] < confidence_thresh:
            continue
        
        pos = (int(kpt[0]), int(kpt[1]))
        cv2.circle(img, pos, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, pos, 4, (0, 0, 0), 1, cv2.LINE_AA)
    
    return img


def plot_fig_3_3(rmm_df: pd.DataFrame, output_path: Path = None):
    """
    Generate Fig 3.3: Frame progression (full frame -> SAM3 crop -> pose skeleton).
    
    Shows the preprocessing pipeline visually.
    """
    print("\n=== Generating Fig 3.3: Frame to Pose Progression ===")
    
    # Find a video with good pose data
    print("  Searching for video with good pose keypoints...")
    video_info = find_video_with_good_pose(rmm_df, min_confidence=0.4, prefer_centered=True)
    
    if video_info is None:
        print("  ERROR: Could not find a video with good pose data")
        print("  Creating placeholder figure...")
        
        # Create placeholder
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (ax, title) in enumerate(zip(axes, ['(A) Original Frame', '(B) SAM3 Crop', '(C) Pose Skeleton'])):
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle('Frame to Pose Progression', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path is None:
            output_path = OUTPUT_DIR / "fig_3_3_frame_pose_progression.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved placeholder: {output_path}")
        plt.close()
        return
    
    print(f"  Selected video: {video_info['filename']}")
    print(f"  Using frame {video_info['frame_idx']} ({video_info['frame_key']})")
    
    # Load original video frame
    video_path = Path(video_info['video_path'])
    
    if not video_path.exists():
        print(f"  ERROR: Video not found: {video_path}")
        return
    
    # Read the frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_info['frame_idx'])
    ret, original_frame = cap.read()
    cap.release()
    
    if not ret:
        print("  ERROR: Could not read frame from video")
        return
    
    # Handle rotation FIRST - the pose keypoints and bbox are in the rotated coordinate space
    record = video_info.get('record', {})
    rotation = record.get('rotation_meta', 0)
    if rotation == -90:
        original_frame = cv2.rotate(original_frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 90:
        original_frame = cv2.rotate(original_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        original_frame = cv2.rotate(original_frame, cv2.ROTATE_180)
    
    print(f"  Frame shape after rotation: {original_frame.shape}")
    
    # Load pose keypoints (these are already in the rotated coordinate space)
    with h5py.File(video_info['pose_cache_path'], 'r') as f:
        pose_group = f[video_info['frame_key']]['pose_0']
        keypoints = pose_group['keypoints'][:17, :]  # First 17 are COCO body keypoints
        bbox = pose_group['bbox'][:] if 'bbox' in pose_group else None
    
    # Get frame dimensions
    h, w = original_frame.shape[:2]
    
    # Get bounding box (already in rotated coordinate space)
    if bbox is not None:
        # bbox format from pose cache: [x1, y1, x2, y2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Add padding around the bbox for better visualization
        padding = 30
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
    else:
        # Estimate bbox from keypoints
        valid_kpts = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_kpts) > 0:
            x_min, y_min = valid_kpts[:, :2].min(axis=0)
            x_max, y_max = valid_kpts[:, :2].max(axis=0)
            padding = 50
            x1, y1 = int(max(0, x_min - padding)), int(max(0, y_min - padding))
            x2, y2 = int(min(w, x_max + padding)), int(min(h, y_max + padding))
        else:
            x1, y1 = 0, 0
            x2, y2 = w, h
    
    print(f"  Bounding box: [{x1}, {y1}, {x2}, {y2}]")
    
    # Create SAM3 crop
    cropped_frame = original_frame[y1:y2, x1:x2].copy()
    
    # Adjust keypoints for crop
    cropped_keypoints = keypoints.copy()
    cropped_keypoints[:, 0] -= x1
    cropped_keypoints[:, 1] -= y1
    
    # Draw skeleton on cropped frame
    skeleton_frame = draw_skeleton(cropped_frame.copy(), cropped_keypoints, confidence_thresh=0.3)
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    skeleton_rgb = cv2.cvtColor(skeleton_frame, cv2.COLOR_BGR2RGB)
    
    # Create figure with extra horizontal space for arrows
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.15)  # Add space between subplots for arrows
    
    # (A) Original frame with bounding box
    axes[0].imshow(original_rgb)
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                          fill=False, edgecolor='lime', linewidth=2, linestyle='--')
    axes[0].add_patch(rect)
    axes[0].set_title('(A) Original Frame', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # (B) SAM3 Crop
    axes[1].imshow(cropped_rgb)
    axes[1].set_title('(B) SAM3 Crop', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # (C) Pose Skeleton
    axes[2].imshow(skeleton_rgb)
    axes[2].set_title('(C) Pose Skeleton', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add arrows between panels using figure-level annotations
    from matplotlib.patches import FancyArrowPatch
    
    # Get positions of axes in figure coordinates
    fig.canvas.draw()
    
    # Arrow between (A) and (B)
    # Get right edge of axes[0] and left edge of axes[1]
    ax0_bbox = axes[0].get_position()
    ax1_bbox = axes[1].get_position()
    ax2_bbox = axes[2].get_position()
    
    # Arrow 1: A -> B (at vertical center of the plots)
    arrow1_start = (ax0_bbox.x1 + 0.01, ax0_bbox.y0 + ax0_bbox.height / 2)
    arrow1_end = (ax1_bbox.x0 - 0.01, ax1_bbox.y0 + ax1_bbox.height / 2)
    
    arrow1 = FancyArrowPatch(
        arrow1_start, arrow1_end,
        transform=fig.transFigure,
        arrowstyle='->,head_width=0.4,head_length=0.3',
        color='#333333',
        linewidth=3,
        mutation_scale=15
    )
    fig.patches.append(arrow1)
    
    # Arrow 2: B -> C
    arrow2_start = (ax1_bbox.x1 + 0.01, ax1_bbox.y0 + ax1_bbox.height / 2)
    arrow2_end = (ax2_bbox.x0 - 0.01, ax2_bbox.y0 + ax2_bbox.height / 2)
    
    arrow2 = FancyArrowPatch(
        arrow2_start, arrow2_end,
        transform=fig.transFigure,
        arrowstyle='->,head_width=0.4,head_length=0.3',
        color='#333333',
        linewidth=3,
        mutation_scale=15
    )
    fig.patches.append(arrow2)
    
    plt.suptitle('Frame to Pose Progression', fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    if output_path is None:
        output_path = OUTPUT_DIR / "fig_3_3_frame_pose_progression.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()


# =============================================================================
# Fig 3.4: Skeleton to Heatmap Progression (PoseC3D Input)
# =============================================================================

def generate_pose_heatmap(keypoints: np.ndarray, img_shape: Tuple[int, int], 
                          sigma: float = 6.0, use_score: bool = True) -> np.ndarray:
    """
    Generate pseudo heatmaps based on joint coordinates and confidence.
    
    This replicates the GeneratePoseTarget pipeline from PoseC3D.
    
    Args:
        keypoints: Array of shape (num_keypoints, 3) with (x, y, confidence)
        img_shape: Tuple of (height, width) for output heatmap
        sigma: Gaussian sigma for heatmap generation
        use_score: Whether to use confidence as max value
    
    Returns:
        Heatmap array of shape (num_keypoints, height, width)
    """
    img_h, img_w = img_shape
    num_kp = keypoints.shape[0]
    
    # Initialize heatmap array
    heatmap = np.zeros((num_kp, img_h, img_w), dtype=np.float32)
    
    for i in range(num_kp):
        kpt = keypoints[i]
        mu_x, mu_y = kpt[0], kpt[1]
        max_value = kpt[2] if use_score else 1.0
        
        if max_value < 1e-3:
            continue
        
        # Calculate bounds for the Gaussian patch
        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
        
        x = np.arange(st_x, ed_x, 1, np.float32)
        y = np.arange(st_y, ed_y, 1, np.float32)
        
        if not (len(x) and len(y)):
            continue
        
        y = y[:, None]
        
        # Generate Gaussian patch
        patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
        patch = patch * max_value
        
        heatmap[i, st_y:ed_y, st_x:ed_x] = np.maximum(
            heatmap[i, st_y:ed_y, st_x:ed_x], patch
        )
    
    return heatmap


def plot_fig_3_4(rmm_df: pd.DataFrame, output_path: Path = None):
    """
    Generate Fig 3.4: Skeleton to Heatmap progression showing PoseC3D input.
    
    Shows: Frame with skeleton overlay -> Pose heatmap (PoseC3D input)
    """
    print("\n=== Generating Fig 3.4: Skeleton to Heatmap (PoseC3D Input) ===")
    
    # Find a video with good pose data (reuse same selection as Fig 3.3)
    print("  Searching for video with good pose keypoints...")
    video_info = find_video_with_good_pose(rmm_df, min_confidence=0.4, prefer_centered=True)
    
    if video_info is None:
        print("  ERROR: Could not find a video with good pose data")
        return
    
    print(f"  Selected video: {video_info['filename']}")
    print(f"  Using frame {video_info['frame_idx']} ({video_info['frame_key']})")
    
    # Load original video frame
    video_path = Path(video_info['video_path'])
    
    if not video_path.exists():
        print(f"  ERROR: Video not found: {video_path}")
        return
    
    # Read the frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_info['frame_idx'])
    ret, original_frame = cap.read()
    cap.release()
    
    if not ret:
        print("  ERROR: Could not read frame from video")
        return
    
    # Handle rotation
    record = video_info.get('record', {})
    rotation = record.get('rotation_meta', 0)
    if rotation == -90:
        original_frame = cv2.rotate(original_frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 90:
        original_frame = cv2.rotate(original_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        original_frame = cv2.rotate(original_frame, cv2.ROTATE_180)
    
    # Load pose keypoints
    with h5py.File(video_info['pose_cache_path'], 'r') as f:
        pose_group = f[video_info['frame_key']]['pose_0']
        keypoints = pose_group['keypoints'][:17, :]  # First 17 are COCO body keypoints
        bbox = pose_group['bbox'][:] if 'bbox' in pose_group else None
    
    # Get frame dimensions
    h, w = original_frame.shape[:2]
    
    # Get bounding box
    if bbox is not None:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        padding = 30
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
    else:
        valid_kpts = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_kpts) > 0:
            x_min, y_min = valid_kpts[:, :2].min(axis=0)
            x_max, y_max = valid_kpts[:, :2].max(axis=0)
            padding = 50
            x1, y1 = int(max(0, x_min - padding)), int(max(0, y_min - padding))
            x2, y2 = int(min(w, x_max + padding)), int(min(h, y_max + padding))
        else:
            x1, y1 = 0, 0
            x2, y2 = w, h
    
    # Create SAM3 crop
    cropped_frame = original_frame[y1:y2, x1:x2].copy()
    crop_h, crop_w = cropped_frame.shape[:2]
    
    # Adjust keypoints for crop
    cropped_keypoints = keypoints.copy()
    cropped_keypoints[:, 0] -= x1
    cropped_keypoints[:, 1] -= y1
    
    # Draw skeleton on cropped frame
    skeleton_frame = draw_skeleton(cropped_frame.copy(), cropped_keypoints, confidence_thresh=0.3)
    
    # Generate heatmap (using PoseC3D's GeneratePoseTarget logic)
    # PoseC3D typically uses sigma=0.6 but that's for normalized coordinates
    # For pixel coordinates, we scale sigma proportionally
    heatmap_sigma = max(crop_h, crop_w) * 0.02  # ~2% of image size
    
    heatmaps = generate_pose_heatmap(
        cropped_keypoints, 
        img_shape=(crop_h, crop_w), 
        sigma=heatmap_sigma,
        use_score=True
    )
    
    # Combine heatmaps into a single visualization
    # Option 1: Max across all keypoint channels
    combined_heatmap = np.max(heatmaps, axis=0)
    
    # Normalize for visualization
    if combined_heatmap.max() > 0:
        combined_heatmap = combined_heatmap / combined_heatmap.max()
    
    # Create colored heatmap using a colormap
    from matplotlib import cm
    heatmap_colored = cm.hot(combined_heatmap)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Also create per-channel visualization (stacked keypoint heatmaps)
    # Color each keypoint channel differently
    keypoint_colors = plt.cm.hsv(np.linspace(0, 0.9, 17))[:, :3]
    
    per_channel_vis = np.zeros((crop_h, crop_w, 3), dtype=np.float32)
    for i in range(17):
        if cropped_keypoints[i, 2] > 0.3:  # Only if keypoint is visible
            channel = heatmaps[i]
            if channel.max() > 0:
                channel_norm = channel / channel.max()
                for c in range(3):
                    per_channel_vis[:, :, c] += channel_norm * keypoint_colors[i, c]
    
    # Clip and normalize
    per_channel_vis = np.clip(per_channel_vis, 0, 1)
    per_channel_vis = (per_channel_vis * 255).astype(np.uint8)
    
    # Convert BGR to RGB for matplotlib
    skeleton_rgb = cv2.cvtColor(skeleton_frame, cv2.COLOR_BGR2RGB)
    
    # Create figure with arrows
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.12)
    
    # (A) Frame with skeleton overlay
    axes[0].imshow(skeleton_rgb)
    axes[0].set_title('(A) Frame with Skeleton', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # (B) Pose heatmap (PoseC3D input)
    axes[1].imshow(per_channel_vis)
    axes[1].set_title('(B) Pose Heatmap (PoseC3D Input)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Add arrow between panels
    from matplotlib.patches import FancyArrowPatch
    
    fig.canvas.draw()
    ax0_bbox = axes[0].get_position()
    ax1_bbox = axes[1].get_position()
    
    arrow_start = (ax0_bbox.x1 + 0.01, ax0_bbox.y0 + ax0_bbox.height / 2)
    arrow_end = (ax1_bbox.x0 - 0.01, ax1_bbox.y0 + ax1_bbox.height / 2)
    
    arrow = FancyArrowPatch(
        arrow_start, arrow_end,
        transform=fig.transFigure,
        arrowstyle='->,head_width=0.4,head_length=0.3',
        color='#333333',
        linewidth=3,
        mutation_scale=15
    )
    fig.patches.append(arrow)
    
    plt.suptitle('Skeleton to Heatmap Progression', fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    if output_path is None:
        output_path = OUTPUT_DIR / "fig_3_4_skeleton_to_heatmap.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()


# =============================================================================
# Table 3.1: Class Distribution by CV Fold
# =============================================================================

def generate_table_3_1(output_path: Path = None):
    """
    Generate Table 3.1: Class distribution by CV fold for 4-class and 5-class.
    
    Creates a JSON file with class counts broken down by fold and split.
    """
    print("\n=== Generating Table 3.1: Class Distribution by CV Fold ===")
    
    # Class label mappings
    label_map_4class = {
        -1: "background",
        0: "hands_flapping",
        1: "jumping",
        2: "rocking",
        3: "spinning"
    }
    
    label_map_5class = {
        -1: "background",
        0: "hands_flapping",
        1: "jumping",
        2: "one_hand_flap",
        3: "rocking",
        4: "spinning"
    }
    
    results = {
        "4class": {},
        "5class": {}
    }
    
    # Process 4-class splits
    print("  Processing 4-class splits...")
    for fold in range(3):
        fold_key = f"fold_{fold}"
        results["4class"][fold_key] = {}
        
        for split in ["train", "val"]:
            csv_path = SPLITS_4CLASS_DIR / f"fold_{fold}_{split}_windows.csv"
            
            if not csv_path.exists():
                print(f"    Warning: {csv_path} not found")
                continue
            
            df = pd.read_csv(csv_path)
            
            # Count by primary_label
            counts = df['primary_label'].value_counts().to_dict()
            
            # Convert to named classes
            named_counts = {}
            for label_id, count in counts.items():
                class_name = label_map_4class.get(label_id, f"unknown_{label_id}")
                named_counts[class_name] = int(count)
            
            results["4class"][fold_key][split] = named_counts
            print(f"    Fold {fold} {split}: {sum(named_counts.values())} windows")
    
    # Process 5-class splits
    print("  Processing 5-class splits...")
    for fold in range(3):
        fold_key = f"fold_{fold}"
        results["5class"][fold_key] = {}
        
        for split in ["train", "val"]:
            csv_path = SPLITS_5CLASS_DIR / f"fold_{fold}_{split}_windows.csv"
            
            if not csv_path.exists():
                print(f"    Warning: {csv_path} not found")
                continue
            
            df = pd.read_csv(csv_path)
            
            # Count by primary_label
            counts = df['primary_label'].value_counts().to_dict()
            
            # Convert to named classes
            named_counts = {}
            for label_id, count in counts.items():
                class_name = label_map_5class.get(label_id, f"unknown_{label_id}")
                named_counts[class_name] = int(count)
            
            results["5class"][fold_key][split] = named_counts
            print(f"    Fold {fold} {split}: {sum(named_counts.values())} windows")
    
    # Add summary statistics
    for task in ["4class", "5class"]:
        totals = {"train": {}, "val": {}}
        
        for fold_key in results[task]:
            for split in ["train", "val"]:
                if split in results[task][fold_key]:
                    for class_name, count in results[task][fold_key][split].items():
                        if class_name not in totals[split]:
                            totals[split][class_name] = []
                        totals[split][class_name].append(count)
        
        # Compute mean and std per class
        results[task]["summary"] = {}
        for split in ["train", "val"]:
            results[task]["summary"][split] = {}
            for class_name, counts in totals[split].items():
                results[task]["summary"][split][class_name] = {
                    "mean": round(np.mean(counts), 1),
                    "std": round(np.std(counts), 1),
                    "total": int(sum(counts))
                }
    
    # Save JSON
    if output_path is None:
        output_path = TABLES_DIR / "table_3_1_class_distribution_by_fold.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved: {output_path}")
    
    # Print summary
    print("\n  === 4-Class Summary ===")
    for class_name, stats in results["4class"]["summary"]["train"].items():
        print(f"    {class_name}: {stats['mean']:.0f} ± {stats['std']:.0f} (total: {stats['total']})")
    
    print("\n  === 5-Class Summary ===")
    for class_name, stats in results["5class"]["summary"]["train"].items():
        print(f"    {class_name}: {stats['mean']:.0f} ± {stats['std']:.0f} (total: {stats['total']})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Data Chapter figures and tables")
    parser.add_argument("--fig", type=str, choices=["3.1", "3.2", "3.3", "3.4", "all"], 
                        default="all", help="Which figure to generate")
    parser.add_argument("--table", type=str, choices=["3.1", "all"],
                        default="all", help="Which table to generate")
    
    args = parser.parse_args()
    
    # Load common data
    annot_df = None
    rmm_df = None
    rmm_filenames = None
    
    if args.fig in ["3.1", "3.2", "all"] or args.table in ["3.1", "all"]:
        annot_df = load_annotations()
        rmm_df = load_rmm_segments()
        rmm_filenames = get_rmm_video_filenames(rmm_df)
    
    if args.fig in ["3.3", "3.4", "all"]:
        if rmm_df is None:
            rmm_df = load_rmm_segments()
    
    # Generate requested outputs
    if args.fig == "3.1" or args.fig == "all":
        plot_fig_3_1(annot_df, rmm_filenames)
    
    if args.fig == "3.2" or args.fig == "all":
        plot_fig_3_2(annot_df, rmm_filenames)
    
    if args.fig == "3.3" or args.fig == "all":
        if rmm_df is None:
            rmm_df = load_rmm_segments()
        plot_fig_3_3(rmm_df)
    
    if args.fig == "3.4" or args.fig == "all":
        if rmm_df is None:
            rmm_df = load_rmm_segments()
        plot_fig_3_4(rmm_df)
    
    if args.table == "3.1" or args.table == "all":
        generate_table_3_1()
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
