#!/usr/bin/env python3
"""
Tests for extract_vjepa_features.py

Run with:
    pytest tests/test_extract_vjepa_features.py -v
    
Or for quick local test (no pytest required):
    python tests/test_extract_vjepa_features.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from extract_vjepa_features import (
    CropConfig,
    CropHelper,
    VideoFeatureDataset,
    child_ids_for_time,
    load_parsed_sam3_csv,
    normalize_video_key,
    rotate_boxes,
)


# =============================================================================
# Quick sanity checks (no pytest required)
# =============================================================================

def test_normalize_video_key():
    """Test normalize_video_key function."""
    # Basic path
    path = "A.B._Home_Videos_AMES_L5V8G4A3K7/12-16 month videos/4-10-2020.mov"
    result = normalize_video_key(path)
    assert result == path, f"Expected unchanged path, got: {result}"
    
    # Backslash conversion
    path = r"A.B._Home_Videos\12-16 month videos\video.mov"
    result = normalize_video_key(path)
    assert "\\" not in result, f"Backslash not removed: {result}"
    assert "/" in result, f"Forward slash not added: {result}"
    
    # Leading slash removal
    path = "/A.B._Home_Videos/test.mov"
    result = normalize_video_key(path)
    assert not result.startswith("/"), f"Leading slash not removed: {result}"
    
    # Volume prefix removal
    path = "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/A.B./video.mov"
    result = normalize_video_key(path)
    assert "Volumes" not in result, f"Volume prefix not removed: {result}"
    
    print("   ✓ normalize_video_key works correctly")


def test_child_ids_for_time():
    """Test child_ids_for_time function."""
    # Single interval
    row = {"intervals": [{"id": 1, "start_sec": 0, "end_sec": 10}]}
    assert child_ids_for_time(row, 5.0) == [1], "Should find id 1 at time 5"
    assert child_ids_for_time(row, 0.0) == [1], "Should find id 1 at time 0"
    assert child_ids_for_time(row, 10.0) == [], "Should not find at end (exclusive)"
    
    # Multiple intervals
    row = {
        "intervals": [
            {"id": 1, "start_sec": 0, "end_sec": 10},
            {"id": 2, "start_sec": 5, "end_sec": 15},
        ]
    }
    result = child_ids_for_time(row, 7.0)
    assert 1 in result and 2 in result, "Both intervals active at time 7"
    
    result = child_ids_for_time(row, 12.0)
    assert 1 not in result and 2 in result, "Only interval 2 active at time 12"
    
    # No intervals
    row = {"intervals": []}
    assert child_ids_for_time(row, 5.0) == [], "Empty intervals should return empty"
    
    # Open-ended interval
    row = {"intervals": [{"id": 1, "start_sec": 0, "end_sec": None}]}
    assert child_ids_for_time(row, 1000.0) == [1], "Open-ended should match any time"
    
    # Time before interval
    row = {"intervals": [{"id": 1, "start_sec": 10, "end_sec": 20}]}
    assert child_ids_for_time(row, 5.0) == [], "Time before interval should return empty"
    
    print("   ✓ child_ids_for_time works correctly")


def test_rotate_boxes():
    """Test rotate_boxes function."""
    boxes = np.array([[10, 20, 30, 40]])
    
    # No rotation
    result = rotate_boxes(boxes, 0, 100, 100)
    np.testing.assert_array_equal(result, boxes)
    
    # 90 degree rotation
    width, height = 100, 80
    result = rotate_boxes(boxes.copy(), 90, width, height)
    expected = np.array([[80 - 40, 10, 80 - 20, 30]])
    np.testing.assert_array_equal(result, expected)
    
    # 180 degree rotation
    result = rotate_boxes(boxes.copy(), 180, width, height)
    expected = np.array([[100 - 30, 80 - 40, 100 - 10, 80 - 20]])
    np.testing.assert_array_equal(result, expected)
    
    # -90 degree rotation
    result = rotate_boxes(boxes.copy(), -90, width, height)
    expected = np.array([[20, 100 - 30, 40, 100 - 10]])
    np.testing.assert_array_equal(result, expected)
    
    print("   ✓ rotate_boxes works correctly")


def test_load_parsed_sam3_csv():
    """Test load_parsed_sam3_csv function."""
    # Missing file
    result = load_parsed_sam3_csv(Path("/nonexistent/path.csv"))
    assert result == {}, "Missing file should return empty dict"
    
    # Valid CSV (create temp file)
    # Use proper CSV quoting: double quotes inside are escaped as ""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('FileName,child_sam3_ids\n')
        f.write('video1.mov,"[{""id"": 1, ""start_sec"": 0, ""end_sec"": 10}]"\n')
        f.write('video2.mov,"[{""id"": 2, ""start_sec"": 5, ""end_sec"": 15}]"\n')
        temp_path = Path(f.name)
    
    try:
        result = load_parsed_sam3_csv(temp_path)
        assert "video1.mov" in result, f"Should contain video1.mov, got keys: {list(result.keys())}"
        assert "video2.mov" in result, f"Should contain video2.mov, got keys: {list(result.keys())}"
        assert len(result["video1.mov"]["intervals"]) == 1, f"Expected 1 interval, got: {result['video1.mov']['intervals']}"
        assert result["video1.mov"]["intervals"][0]["id"] == 1
    finally:
        temp_path.unlink()
    
    print("   ✓ load_parsed_sam3_csv works correctly")


def test_crop_config():
    """Test CropConfig dataclass."""
    # Default values
    cfg = CropConfig()
    assert cfg.enabled is False
    assert cfg.padding == 20
    assert cfg.mask_model == "facebook-sam3"
    assert cfg.mask_prompt == "person"
    
    # Custom values
    cfg = CropConfig(enabled=True, padding=50, mask_model="custom-model")
    assert cfg.enabled is True
    assert cfg.padding == 50
    assert cfg.mask_model == "custom-model"
    
    print("   ✓ CropConfig works correctly")


def test_crop_helper_disabled():
    """Test CropHelper with cropping disabled."""
    cfg = CropConfig(enabled=False)
    helper = CropHelper(cfg)
    
    frames = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
    indices = np.arange(16)
    
    result_frames, meta = helper.apply_crop(
        frames=frames,
        video_file="test.mov",
        frame_indices=indices,
        fps=30.0,
        video_path=Path("/tmp/test.mov"),
    )
    
    np.testing.assert_array_equal(result_frames, frames)
    assert meta["crop_applied"] is False
    
    print("   ✓ CropHelper (disabled) works correctly")


def test_crop_helper_no_sam3_row():
    """Test CropHelper when no SAM3 data available."""
    cfg = CropConfig(enabled=True)
    helper = CropHelper(cfg)
    helper.sam3_rows = {}  # Empty SAM3 data
    
    frames = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
    indices = np.arange(16)
    
    result_frames, meta = helper.apply_crop(
        frames=frames,
        video_file="unknown_video.mov",
        frame_indices=indices,
        fps=30.0,
        video_path=Path("/tmp/unknown_video.mov"),
    )
    
    np.testing.assert_array_equal(result_frames, frames)
    assert meta["crop_applied"] is False
    assert meta["crop_reason"] == "no_sam3_row"
    
    print("   ✓ CropHelper (no SAM3 row) works correctly")


def test_crop_helper_invalid_fps():
    """Test CropHelper with invalid FPS."""
    cfg = CropConfig(enabled=True)
    helper = CropHelper(cfg)
    helper.sam3_rows = {"test.mov": {"intervals": []}}
    
    frames = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
    indices = np.arange(16)
    
    result_frames, meta = helper.apply_crop(
        frames=frames,
        video_file="test.mov",
        frame_indices=indices,
        fps=0.0,  # Invalid FPS
        video_path=Path("/tmp/test.mov"),
    )
    
    np.testing.assert_array_equal(result_frames, frames)
    assert meta["crop_applied"] is False
    assert meta["crop_reason"] == "fps_missing"
    
    print("   ✓ CropHelper (invalid FPS) works correctly")


def test_video_feature_dataset_initialization():
    """Test VideoFeatureDataset initialization."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        video_df = pd.DataFrame({
            "video_file": ["folder/video1.mov", "folder/video2.mov"],
            "child_id": ["A1B2C3D4E5", "F6G7H8I9J0"],
            "filename_stem": ["video1", "video2"],
        })
        
        dataset = VideoFeatureDataset(
            video_df=video_df,
            videos_root=tmp_path,
            snippet_size=16,
            snippet_stride=16,
        )
        
        assert len(dataset) == 2
        assert dataset.records[0]["feature_name"] == "A1B2C3D4E5_video1"
        assert dataset.records[1]["feature_name"] == "F6G7H8I9J0_video2"
    
    print("   ✓ VideoFeatureDataset initialization works correctly")


def test_video_feature_dataset_missing_video():
    """Test VideoFeatureDataset handling of missing video files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        video_df = pd.DataFrame({
            "video_file": ["nonexistent.mov"],
            "child_id": ["A1B2C3D4E5"],
            "filename_stem": ["nonexistent"],
        })
        
        dataset = VideoFeatureDataset(
            video_df=video_df,
            videos_root=tmp_path,
            snippet_size=16,
            snippet_stride=16,
        )
        
        result = dataset[0]
        assert result is None, "Missing video should return None"
    
    print("   ✓ VideoFeatureDataset (missing video) works correctly")


def run_all_tests():
    """Run all quick sanity checks."""
    print("=" * 60)
    print("Running quick sanity checks for extract_vjepa_features.py...")
    print("=" * 60)
    
    print("\n1. Testing normalize_video_key...")
    test_normalize_video_key()
    
    print("\n2. Testing child_ids_for_time...")
    test_child_ids_for_time()
    
    print("\n3. Testing rotate_boxes...")
    test_rotate_boxes()
    
    print("\n4. Testing load_parsed_sam3_csv...")
    test_load_parsed_sam3_csv()
    
    print("\n5. Testing CropConfig...")
    test_crop_config()
    
    print("\n6. Testing CropHelper (disabled)...")
    test_crop_helper_disabled()
    
    print("\n7. Testing CropHelper (no SAM3 row)...")
    test_crop_helper_no_sam3_row()
    
    print("\n8. Testing CropHelper (invalid FPS)...")
    test_crop_helper_invalid_fps()
    
    print("\n9. Testing VideoFeatureDataset initialization...")
    test_video_feature_dataset_initialization()
    
    print("\n10. Testing VideoFeatureDataset (missing video)...")
    test_video_feature_dataset_missing_video()
    
    print("\n" + "=" * 60)
    print("All 10 tests passed! ✓")
    print("=" * 60)
    print("\nFor full pytest suite, install pytest and run:")
    print("  pytest tests/test_extract_vjepa_features.py -v")


# =============================================================================
# Pytest test classes (only used when pytest is available)
# =============================================================================

try:
    import pytest
    from unittest.mock import MagicMock, patch
    
    class TestNormalizeVideoKeyPytest:
        """Pytest tests for normalize_video_key function."""
        
        def test_basic_path(self):
            path = "A.B._Home_Videos_AMES_L5V8G4A3K7/12-16 month videos/4-10-2020.mov"
            result = normalize_video_key(path)
            assert result == path
        
        def test_backslash_conversion(self):
            path = r"A.B._Home_Videos\12-16 month videos\video.mov"
            result = normalize_video_key(path)
            assert "\\" not in result
            assert "/" in result
        
        def test_leading_slash_removal(self):
            path = "/A.B._Home_Videos/test.mov"
            result = normalize_video_key(path)
            assert not result.startswith("/")
        
        def test_volume_prefix_removal(self):
            path = "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/A.B./video.mov"
            result = normalize_video_key(path)
            assert "Volumes" not in result
            assert result == "A.B./video.mov"
    
    class TestChildIdsForTimePytest:
        """Pytest tests for child_ids_for_time function."""
        
        def test_single_interval(self):
            row = {"intervals": [{"id": 1, "start_sec": 0, "end_sec": 10}]}
            assert child_ids_for_time(row, 5.0) == [1]
            assert child_ids_for_time(row, 0.0) == [1]
            assert child_ids_for_time(row, 10.0) == []
        
        def test_multiple_intervals(self):
            row = {
                "intervals": [
                    {"id": 1, "start_sec": 0, "end_sec": 10},
                    {"id": 2, "start_sec": 5, "end_sec": 15},
                ]
            }
            result = child_ids_for_time(row, 7.0)
            assert 1 in result
            assert 2 in result
            
            result = child_ids_for_time(row, 12.0)
            assert 1 not in result
            assert 2 in result
        
        def test_no_intervals(self):
            row = {"intervals": []}
            assert child_ids_for_time(row, 5.0) == []
        
        def test_open_ended_interval(self):
            row = {"intervals": [{"id": 1, "start_sec": 0, "end_sec": None}]}
            assert child_ids_for_time(row, 1000.0) == [1]
    
    class TestRotateBoxesPytest:
        """Pytest tests for rotate_boxes function."""
        
        def test_no_rotation(self):
            boxes = np.array([[10, 20, 30, 40]])
            result = rotate_boxes(boxes, 0, 100, 100)
            np.testing.assert_array_equal(result, boxes)
        
        def test_90_degree_rotation(self):
            boxes = np.array([[10, 20, 30, 40]])
            width, height = 100, 80
            result = rotate_boxes(boxes, 90, width, height)
            expected = np.array([[80 - 40, 10, 80 - 20, 30]])
            np.testing.assert_array_equal(result, expected)
        
        def test_180_degree_rotation(self):
            boxes = np.array([[10, 20, 30, 40]])
            width, height = 100, 80
            result = rotate_boxes(boxes, 180, width, height)
            expected = np.array([[100 - 30, 80 - 40, 100 - 10, 80 - 20]])
            np.testing.assert_array_equal(result, expected)
    
    class TestLoadParsedSam3CsvPytest:
        """Pytest tests for load_parsed_sam3_csv function."""
        
        def test_missing_file(self):
            result = load_parsed_sam3_csv(Path("/nonexistent/path.csv"))
            assert result == {}
        
        def test_valid_csv(self, tmp_path):
            # Use proper CSV quoting: double quotes inside are escaped as ""
            csv_content = 'FileName,child_sam3_ids\nvideo1.mov,"[{""id"": 1, ""start_sec"": 0, ""end_sec"": 10}]"\n'
            csv_path = tmp_path / "test.csv"
            csv_path.write_text(csv_content)
            
            result = load_parsed_sam3_csv(csv_path)
            assert "video1.mov" in result
            assert len(result["video1.mov"]["intervals"]) == 1
    
    class TestCropHelperPytest:
        """Pytest tests for CropHelper class."""
        
        def test_disabled_cropping(self):
            cfg = CropConfig(enabled=False)
            helper = CropHelper(cfg)
            
            frames = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
            indices = np.arange(16)
            
            result_frames, meta = helper.apply_crop(
                frames=frames,
                video_file="test.mov",
                frame_indices=indices,
                fps=30.0,
                video_path=Path("/tmp/test.mov"),
            )
            
            np.testing.assert_array_equal(result_frames, frames)
            assert meta["crop_applied"] is False
        
        def test_no_sam3_row(self):
            cfg = CropConfig(enabled=True)
            helper = CropHelper(cfg)
            helper.sam3_rows = {}
            
            frames = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
            result_frames, meta = helper.apply_crop(
                frames=frames,
                video_file="unknown_video.mov",
                frame_indices=np.arange(16),
                fps=30.0,
                video_path=Path("/tmp/unknown_video.mov"),
            )
            
            np.testing.assert_array_equal(result_frames, frames)
            assert meta["crop_reason"] == "no_sam3_row"
    
    class TestVideoFeatureDatasetPytest:
        """Pytest tests for VideoFeatureDataset class."""
        
        def test_initialization(self, tmp_path):
            video_df = pd.DataFrame({
                "video_file": ["folder/video1.mov", "folder/video2.mov"],
                "child_id": ["A1B2C3D4E5", "F6G7H8I9J0"],
                "filename_stem": ["video1", "video2"],
            })
            
            dataset = VideoFeatureDataset(
                video_df=video_df,
                videos_root=tmp_path,
                snippet_size=16,
                snippet_stride=16,
            )
            
            assert len(dataset) == 2
            assert dataset.records[0]["feature_name"] == "A1B2C3D4E5_video1"
        
        def test_missing_video(self, tmp_path):
            video_df = pd.DataFrame({
                "video_file": ["nonexistent.mov"],
                "child_id": ["A1B2C3D4E5"],
                "filename_stem": ["nonexistent"],
            })
            
            dataset = VideoFeatureDataset(
                video_df=video_df,
                videos_root=tmp_path,
                snippet_size=16,
                snippet_stride=16,
            )
            
            result = dataset[0]
            assert result is None

except ImportError:
    # pytest not available, skip pytest classes
    pass


if __name__ == "__main__":
    run_all_tests()
