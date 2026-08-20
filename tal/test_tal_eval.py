#!/usr/bin/env python3
"""
Unit tests and sanity checks for TAL evaluation pipeline.

This module contains:
1. Unit tests for tIoU computation
2. Unit tests for AP computation
3. Unit tests for window→segment postprocessing
4. Oracle sanity check: perfect predictor should get AP ≈ 1.0

Run tests:
    python test_tal_eval.py

Run oracle sanity check on real data:
    python test_tal_eval.py --oracle --splits-root /path/to/splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Import modules under test
from tal_map_eval import (
    GTSegment,
    PredSegment,
    compute_ap_for_class,
    compute_ap_from_pr,
    compute_map,
    compute_tiou,
    load_gt_segments,
)
from window_to_segments import (
    PostprocessParams,
    find_contiguous_runs,
    merge_close_segments,
    smooth_scores,
    window_scores_to_segments,
)

# Resolve paths relative to the repo root so this runs from any clone location.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())


# ============================================================================
# Unit tests for tIoU
# ============================================================================

def test_tiou_perfect_overlap():
    """Perfect overlap should give tIoU = 1.0."""
    tiou = compute_tiou(0.0, 10.0, 0.0, 10.0)
    assert abs(tiou - 1.0) < 1e-6, f"Expected 1.0, got {tiou}"
    print("✓ test_tiou_perfect_overlap")


def test_tiou_no_overlap():
    """No overlap should give tIoU = 0.0."""
    tiou = compute_tiou(0.0, 5.0, 10.0, 15.0)
    assert abs(tiou - 0.0) < 1e-6, f"Expected 0.0, got {tiou}"
    print("✓ test_tiou_no_overlap")


def test_tiou_touching():
    """Touching segments (end == start) should give tIoU = 0.0 (half-open)."""
    tiou = compute_tiou(0.0, 5.0, 5.0, 10.0)
    assert abs(tiou - 0.0) < 1e-6, f"Expected 0.0, got {tiou}"
    print("✓ test_tiou_touching")


def test_tiou_half_overlap():
    """50% overlap should give tIoU = 1/3."""
    # Pred: [0, 10), GT: [5, 15)
    # Intersection: [5, 10) = 5
    # Union: 10 + 10 - 5 = 15
    # tIoU = 5/15 = 1/3
    tiou = compute_tiou(0.0, 10.0, 5.0, 15.0)
    expected = 1.0 / 3.0
    assert abs(tiou - expected) < 1e-6, f"Expected {expected}, got {tiou}"
    print("✓ test_tiou_half_overlap")


def test_tiou_contained():
    """Fully contained segment."""
    # Pred: [2, 8), GT: [0, 10)
    # Intersection: [2, 8) = 6
    # Union: 6 + 10 - 6 = 10
    # tIoU = 6/10 = 0.6
    tiou = compute_tiou(2.0, 8.0, 0.0, 10.0)
    expected = 0.6
    assert abs(tiou - expected) < 1e-6, f"Expected {expected}, got {tiou}"
    print("✓ test_tiou_contained")


# ============================================================================
# Unit tests for AP computation
# ============================================================================

def test_ap_perfect_detector():
    """Perfect detector with 100% recall should get AP = 1.0."""
    # Precision-recall: all ones
    precision = np.array([1.0, 1.0, 1.0])
    recall = np.array([0.33, 0.67, 1.0])
    ap = compute_ap_from_pr(precision, recall)
    assert abs(ap - 1.0) < 1e-6, f"Expected 1.0, got {ap}"
    print("✓ test_ap_perfect_detector")


def test_ap_empty():
    """Empty predictions should give AP = 0."""
    precision = np.array([])
    recall = np.array([])
    ap = compute_ap_from_pr(precision, recall)
    assert abs(ap - 0.0) < 1e-6, f"Expected 0.0, got {ap}"
    print("✓ test_ap_empty")


def test_ap_single_tp():
    """Single TP with full recall."""
    precision = np.array([1.0])
    recall = np.array([1.0])
    ap = compute_ap_from_pr(precision, recall)
    assert abs(ap - 1.0) < 1e-6, f"Expected 1.0, got {ap}"
    print("✓ test_ap_single_tp")


def test_ap_with_fps():
    """AP with some false positives."""
    # 2 TPs out of 4 preds, 2 GTs
    # Pred order: TP, FP, TP, FP
    # Precision: 1/1, 1/2, 2/3, 2/4
    # Recall: 1/2, 1/2, 2/2, 2/2
    precision = np.array([1.0, 0.5, 0.667, 0.5])
    recall = np.array([0.5, 0.5, 1.0, 1.0])
    ap = compute_ap_from_pr(precision, recall)
    # Should be area under interpolated curve
    assert ap > 0.5 and ap < 1.0, f"Expected AP in (0.5, 1.0), got {ap}"
    print("✓ test_ap_with_fps")


def test_compute_ap_for_class_no_gt():
    """No GT segments should return NaN AP."""
    preds = [PredSegment("video1", 0, 0.0, 5.0, 0.9)]
    gt_by_video = {"video1": []}  # No GT
    
    ap, details = compute_ap_for_class(preds, gt_by_video, class_id=0, tiou_threshold=0.5)
    
    assert np.isnan(ap), f"Expected NaN, got {ap}"
    assert details["n_gt"] == 0
    print("✓ test_compute_ap_for_class_no_gt")


def test_compute_ap_for_class_no_preds():
    """No predictions should return AP = 0."""
    preds = []
    gt_by_video = {
        "video1": [GTSegment("seg1", "video1", 0, "hands flapping", 0.0, 5.0)]
    }
    
    ap, details = compute_ap_for_class(preds, gt_by_video, class_id=0, tiou_threshold=0.5)
    
    assert abs(ap - 0.0) < 1e-6, f"Expected 0.0, got {ap}"
    assert details["n_pred"] == 0
    print("✓ test_compute_ap_for_class_no_preds")


# ============================================================================
# Unit tests for postprocessing
# ============================================================================

def test_smooth_scores():
    """Test moving average smoothing."""
    scores = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    smoothed = smooth_scores(scores, k=3)
    
    # Smoothed should be less extreme
    assert smoothed[2] < 1.0, "Middle should be smoothed"
    assert smoothed[0] >= 0.0, "Should not go negative"
    assert len(smoothed) == len(scores), "Length should match"
    print("✓ test_smooth_scores")


def test_find_contiguous_runs():
    """Test finding runs of True values."""
    mask = np.array([False, True, True, False, True, False])
    runs = find_contiguous_runs(mask)
    
    assert runs == [(1, 3), (4, 5)], f"Expected [(1, 3), (4, 5)], got {runs}"
    print("✓ test_find_contiguous_runs")


def test_find_contiguous_runs_empty():
    """Empty mask should return empty list."""
    runs = find_contiguous_runs(np.array([], dtype=bool))
    assert runs == [], f"Expected [], got {runs}"
    print("✓ test_find_contiguous_runs_empty")


def test_find_contiguous_runs_all_true():
    """All True should return single run."""
    mask = np.array([True, True, True])
    runs = find_contiguous_runs(mask)
    assert runs == [(0, 3)], f"Expected [(0, 3)], got {runs}"
    print("✓ test_find_contiguous_runs_all_true")


def test_merge_close_segments():
    """Test merging close segments."""
    # Two segments with 1s gap
    segments = [(0.0, 2.0, 0.9, 2), (3.0, 5.0, 0.8, 2)]
    
    # Gap = 1s, merge_gap = 1.5s -> should merge
    merged = merge_close_segments(segments, gap_sec=1.5, score_reducer="max")
    assert len(merged) == 1, f"Expected 1 merged segment, got {len(merged)}"
    assert merged[0][0] == 0.0 and merged[0][1] == 5.0
    assert merged[0][2] == 0.9  # max score
    print("✓ test_merge_close_segments")


def test_window_scores_to_segments():
    """Test full window→segment pipeline."""
    # Create synthetic window predictions
    df = pd.DataFrame({
        "video_key": ["vid1"] * 5,
        "start_sec": [0.0, 1.0, 2.0, 3.0, 4.0],
        "end_sec": [2.0, 3.0, 4.0, 5.0, 6.0],
        "score_class0": [0.1, 0.8, 0.9, 0.7, 0.1],
        "score_class1": [0.1, 0.1, 0.1, 0.1, 0.1],
    })
    
    params = PostprocessParams(
        smooth_k=1,  # No smoothing
        threshold=0.5,
        merge_gap_sec=1.0,
        class_ids=[0, 1],
    )
    
    segments_df = window_scores_to_segments(df, params)
    
    # Should detect one segment for class 0 covering windows 1-3
    class0_segs = segments_df[segments_df["class_id"] == 0]
    assert len(class0_segs) == 1, f"Expected 1 class0 segment, got {len(class0_segs)}"
    
    seg = class0_segs.iloc[0]
    assert seg["start_sec"] == 1.0, f"Expected start=1.0, got {seg['start_sec']}"
    assert seg["end_sec"] == 5.0, f"Expected end=5.0, got {seg['end_sec']}"
    
    # Should have no segments for class 1 (all below threshold)
    class1_segs = segments_df[segments_df["class_id"] == 1]
    assert len(class1_segs) == 0, f"Expected 0 class1 segments, got {len(class1_segs)}"
    
    print("✓ test_window_scores_to_segments")


# ============================================================================
# Integration test: Perfect oracle
# ============================================================================

def test_perfect_oracle():
    """
    A perfect predictor that outputs GT segments exactly should get AP ≈ 1.0.
    """
    # Create synthetic GT
    gt_by_video = {
        "video1": [
            GTSegment("s1", "video1", 0, "hands flapping", 0.0, 5.0),
            GTSegment("s2", "video1", 0, "hands flapping", 10.0, 15.0),
        ],
        "video2": [
            GTSegment("s3", "video2", 1, "jumping", 0.0, 3.0),
        ],
    }
    
    # Create perfect predictions (same as GT)
    pred_df = pd.DataFrame([
        {"video_key": "video1", "class_id": 0, "start_sec": 0.0, "end_sec": 5.0, "score": 0.99},
        {"video_key": "video1", "class_id": 0, "start_sec": 10.0, "end_sec": 15.0, "score": 0.95},
        {"video_key": "video2", "class_id": 1, "start_sec": 0.0, "end_sec": 3.0, "score": 0.9},
    ])
    
    metrics = compute_map(pred_df, gt_by_video, class_ids=[0, 1], tiou_thresholds=[0.5])
    
    map_05 = metrics["mAP@0.5"]
    assert map_05 > 0.99, f"Perfect oracle should get mAP ≈ 1.0, got {map_05}"
    print("✓ test_perfect_oracle")


# ============================================================================
# Oracle sanity check on real data
# ============================================================================

def run_oracle_sanity_check(splits_root: Path, task: str = "4class"):
    """
    Run oracle sanity check on real GT data.
    
    Creates "perfect" window predictions from GT overlap and verifies
    that the evaluator returns high AP.
    """
    print("\n" + "=" * 60)
    print("Oracle Sanity Check on Real Data")
    print("=" * 60)
    
    # Load real GT
    print(f"\nLoading GT from: {splits_root}")
    gt_by_video = load_gt_segments(splits_root, task, "cv")
    
    n_gt = sum(len(segs) for segs in gt_by_video.values())
    print(f"  Loaded {n_gt} GT segments from {len(gt_by_video)} videos")
    
    if n_gt == 0:
        print("  ERROR: No GT segments found!")
        return False
    
    # Create oracle predictions: exactly match GT segments with high confidence
    pred_records = []
    for video_key, gt_list in gt_by_video.items():
        for gt in gt_list:
            pred_records.append({
                "video_key": video_key,
                "class_id": gt.label_idx,
                "start_sec": gt.start_sec,
                "end_sec": gt.end_sec,
                "score": 0.99,  # High confidence
            })
    
    pred_df = pd.DataFrame(pred_records)
    print(f"  Created {len(pred_df)} oracle predictions")
    
    # Evaluate
    class_ids = [0, 1, 2, 3] if task == "4class" else [0, 1, 2, 3, 4]
    tiou_thresholds = [0.3, 0.5, 0.7]
    
    metrics = compute_map(pred_df, gt_by_video, class_ids, tiou_thresholds)
    
    print("\nOracle Results:")
    all_pass = True
    for thr in tiou_thresholds:
        map_val = metrics[f"mAP@{thr}"]
        status = "✓" if map_val > 0.99 else "✗"
        print(f"  mAP@{thr}: {map_val:.4f} {status}")
        if map_val < 0.99:
            all_pass = False
    
    if all_pass:
        print("\n✓ Oracle sanity check PASSED: All mAP values ≈ 1.0")
    else:
        print("\n✗ Oracle sanity check FAILED: Some mAP values < 1.0")
        print("  This indicates a bug in the evaluation pipeline!")
    
    return all_pass


# ============================================================================
# Main
# ============================================================================

def run_all_unit_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running Unit Tests")
    print("=" * 60)
    print()
    
    # tIoU tests
    print("tIoU tests:")
    test_tiou_perfect_overlap()
    test_tiou_no_overlap()
    test_tiou_touching()
    test_tiou_half_overlap()
    test_tiou_contained()
    print()
    
    # AP tests
    print("AP tests:")
    test_ap_perfect_detector()
    test_ap_empty()
    test_ap_single_tp()
    test_ap_with_fps()
    test_compute_ap_for_class_no_gt()
    test_compute_ap_for_class_no_preds()
    print()
    
    # Postprocessing tests
    print("Postprocessing tests:")
    test_smooth_scores()
    test_find_contiguous_runs()
    test_find_contiguous_runs_empty()
    test_find_contiguous_runs_all_true()
    test_merge_close_segments()
    test_window_scores_to_segments()
    print()
    
    # Integration test
    print("Integration tests:")
    test_perfect_oracle()
    print()
    
    print("=" * 60)
    print("All unit tests PASSED!")
    print("=" * 60)


# =============================================================================
# Fusion Pipeline Smoke Test
# =============================================================================

def run_fusion_smoke_test():
    """
    Smoke test for the fusion pipeline.
    
    Creates synthetic data, runs fusion, and verifies the output
    can be processed by the TAL evaluation pipeline.
    """
    import tempfile
    
    print("\n" + "=" * 60)
    print("Fusion Pipeline Smoke Test")
    print("=" * 60)
    
    try:
        from tal_fusion import FusionConfig, run_oof_fusion
    except ImportError as e:
        print(f"  Skipping fusion smoke test: {e}")
        return True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic window splits CSV
        print("\n1. Creating synthetic data...")
        np.random.seed(42)
        
        n_windows = 30
        n_videos = 3
        
        window_rows = []
        for vid_idx in range(n_videos):
            video_key = f"test_video_{vid_idx}"
            for win_idx in range(n_windows // n_videos):
                window_id = f"{video_key}__t{win_idx * 1000}_{(win_idx + 2) * 1000}"
                primary_label = np.random.choice([-1, 0, 1, 2, 3], p=[0.6, 0.15, 0.1, 0.1, 0.05])
                window_rows.append({
                    "window_id": window_id,
                    "video_key": video_key,
                    "start_sec": float(win_idx),
                    "end_sec": float(win_idx + 2),
                    "primary_label": primary_label,
                    "is_background": 1 if primary_label == -1 else 0,
                })
        
        window_df = pd.DataFrame(window_rows)
        window_csv = tmpdir / "windows.csv"
        window_df.to_csv(window_csv, index=False)
        
        # Create TAL-format predictions for two modalities
        def create_tal_preds(window_ids, seed):
            np.random.seed(seed)
            rows = []
            for wid in window_ids:
                video_key = wid.rsplit("__t", 1)[0]
                start_ms, end_ms = wid.rsplit("__t", 1)[1].split("_")
                
                logits = np.random.randn(5)
                probs = np.exp(logits) / np.exp(logits).sum()
                
                row = {
                    "window_id": wid,
                    "video_key": video_key,
                    "start_sec": float(start_ms) / 1000,
                    "end_sec": float(end_ms) / 1000,
                }
                for i in range(5):
                    row[f"score_class{i}"] = probs[i]
                rows.append(row)
            return pd.DataFrame(rows)
        
        window_ids = window_df["window_id"].tolist()
        
        mod1_df = create_tal_preds(window_ids, seed=100)
        mod1_csv = tmpdir / "mod1.csv"
        mod1_df.to_csv(mod1_csv, index=False)
        
        # Mod2 missing some windows
        mod2_df = create_tal_preds(window_ids[:int(len(window_ids) * 0.8)], seed=200)
        mod2_csv = tmpdir / "mod2.csv"
        mod2_df.to_csv(mod2_csv, index=False)
        
        print(f"   Created {len(window_df)} windows, {n_videos} videos")
        print(f"   Mod1: {len(mod1_df)} windows, Mod2: {len(mod2_df)} windows")
        
        # Run fusion
        print("\n2. Running OOF fusion...")
        config = FusionConfig(
            num_classes=5,
            hidden_dim=8,
            num_epochs=3,
            n_inner_folds=2,
            random_seed=42,
        )
        
        fused_df, stats = run_oof_fusion(
            window_splits_csv=window_csv,
            modality1_csv=mod1_csv,
            modality2_csv=mod2_csv,
            config=config,
        )
        
        print(f"   OOF accuracy: {stats['oof_accuracy']:.4f}")
        
        # Verify output format
        print("\n3. Verifying fused output format...")
        assert len(fused_df) == n_windows, f"Expected {n_windows} rows, got {len(fused_df)}"
        assert "window_id" in fused_df.columns
        assert "video_key" in fused_df.columns
        for i in range(5):
            assert f"score_class{i}" in fused_df.columns
        
        # Verify probabilities sum to 1
        prob_cols = [f"score_class{i}" for i in range(5)]
        prob_sums = fused_df[prob_cols].sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probabilities don't sum to 1"
        
        print("   ✓ Output format valid")
        
        # Run through postprocessing
        print("\n4. Running postprocessing on fused output...")
        params = PostprocessParams(
            smooth_k=1,
            threshold=0.3,
            merge_gap_sec=1.0,
            class_ids=[0, 1, 2, 3],
        )
        
        segments_df = window_scores_to_segments(fused_df, params)
        print(f"   Generated {len(segments_df)} segments")
        print("   ✓ Postprocessing successful")
        
        print("\n" + "=" * 60)
        print("✓ Fusion pipeline smoke test PASSED!")
        print("=" * 60)
        
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Test TAL evaluation pipeline.")
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Run oracle sanity check on real data.",
    )
    parser.add_argument(
        "--fusion",
        action="store_true",
        help="Run fusion pipeline smoke test.",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=_REPO_ROOT / "dataprep/splits",
        help="Root directory for split CSVs (for oracle check).",
    )
    parser.add_argument(
        "--task",
        choices=["4class", "5class"],
        default="4class",
        help="Task type for oracle check.",
    )
    
    args = parser.parse_args()
    
    # Always run unit tests
    try:
        run_all_unit_tests()
    except AssertionError as e:
        print(f"\n✗ Unit test FAILED: {e}")
        sys.exit(1)
    
    # Optionally run oracle check
    if args.oracle:
        success = run_oracle_sanity_check(args.splits_root, args.task)
        if not success:
            sys.exit(1)
    
    # Optionally run fusion smoke test
    if args.fusion:
        success = run_fusion_smoke_test()
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()

