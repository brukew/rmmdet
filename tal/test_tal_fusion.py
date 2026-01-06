#!/usr/bin/env python3
"""
Unit tests for TAL fusion module.

Tests cover:
1. TAL-format CSV schema validation
2. Grouped split determinism (seeded)
3. OOF predictions coverage (one fused row per input window)
4. Missing modality handling
5. Feature computation (log-probs)

Run tests:
    python test_tal_fusion.py

Run with verbose output:
    python test_tal_fusion.py -v
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Import modules under test
from tal_fusion import (
    FusionConfig,
    FusionDataset,
    LogProbMLPFusion,
    compute_class_weights,
    load_tal_format_preds,
    load_window_metadata,
    merge_modalities,
    run_oof_fusion,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def create_synthetic_window_splits_csv(n_windows: int = 100, n_videos: int = 5) -> pd.DataFrame:
    """
    Create synthetic window splits CSV data.
    
    Returns:
        DataFrame mimicking fold_*_val_windows.csv structure
    """
    np.random.seed(42)
    
    rows = []
    windows_per_video = n_windows // n_videos
    
    for vid_idx in range(n_videos):
        video_key = f"video_{vid_idx}"
        for win_idx in range(windows_per_video):
            window_id = f"{video_key}__t{win_idx * 1000}_{(win_idx + 2) * 1000}"
            start_sec = float(win_idx)
            end_sec = float(win_idx + 2)
            
            # Assign random label (-1 = background, 0-3 = RMM classes)
            primary_label = np.random.choice([-1, 0, 1, 2, 3], p=[0.7, 0.1, 0.08, 0.07, 0.05])
            is_background = 1 if primary_label == -1 else 0
            
            rows.append({
                "window_id": window_id,
                "video_key": video_key,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "primary_label": primary_label,
                "is_background": is_background,
            })
    
    return pd.DataFrame(rows)


def create_synthetic_tal_preds_csv(window_ids: list, num_classes: int = 5) -> pd.DataFrame:
    """
    Create synthetic TAL-format predictions.
    
    Args:
        window_ids: List of window IDs to include
        num_classes: Number of classes
    
    Returns:
        DataFrame in TAL format
    """
    np.random.seed(123)
    
    rows = []
    for wid in window_ids:
        # Parse video_key from window_id
        video_key = wid.rsplit("__t", 1)[0]
        start_ms, end_ms = wid.rsplit("__t", 1)[1].split("_")
        start_sec = float(start_ms) / 1000
        end_sec = float(end_ms) / 1000
        
        # Generate random softmax scores
        logits = np.random.randn(num_classes)
        probs = np.exp(logits) / np.exp(logits).sum()
        
        row = {
            "window_id": wid,
            "video_key": video_key,
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
        for i in range(num_classes):
            row[f"score_class{i}"] = probs[i]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# Unit Tests: Schema Validation
# =============================================================================

def test_tal_format_schema():
    """Test that TAL-format CSV loading validates expected columns."""
    # Create valid TAL-format CSV
    df = create_synthetic_tal_preds_csv(["vid1__t0_2000", "vid1__t1000_3000"], num_classes=5)
    
    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        loaded = load_tal_format_preds(Path(f.name), num_classes=5, prefix="test")
    
    # Check renamed columns exist
    assert "window_id" in loaded.columns
    for i in range(5):
        assert f"test_score_class{i}" in loaded.columns
    
    print("✓ test_tal_format_schema")


def test_window_metadata_label_mapping():
    """Test that primary_label is correctly mapped to label_id."""
    df = create_synthetic_window_splits_csv(n_windows=10, n_videos=1)
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        meta = load_window_metadata(Path(f.name), num_classes=5)
    
    # Check label_id mapping
    # primary_label -1 -> label_id 4 (background)
    # primary_label 0-3 -> label_id 0-3
    for idx, row in df.iterrows():
        expected = 4 if row["primary_label"] == -1 else row["primary_label"]
        actual = meta.iloc[idx]["label_id"]
        assert actual == expected, f"Row {idx}: expected {expected}, got {actual}"
    
    print("✓ test_window_metadata_label_mapping")


# =============================================================================
# Unit Tests: Grouped Splits
# =============================================================================

def test_grouped_split_determinism():
    """Test that grouped splits are deterministic with same seed."""
    from sklearn.model_selection import GroupKFold
    
    # Create data with multiple windows per video
    n_windows = 50
    n_videos = 5
    
    groups = np.repeat(np.arange(n_videos), n_windows // n_videos)
    labels = np.random.randint(0, 5, size=n_windows)
    
    # Run GroupKFold twice with same seed
    kfold1 = GroupKFold(n_splits=3)
    splits1 = list(kfold1.split(np.arange(n_windows), labels, groups))
    
    kfold2 = GroupKFold(n_splits=3)
    splits2 = list(kfold2.split(np.arange(n_windows), labels, groups))
    
    # Splits should be identical
    for (train1, val1), (train2, val2) in zip(splits1, splits2):
        assert np.array_equal(train1, train2)
        assert np.array_equal(val1, val2)
    
    print("✓ test_grouped_split_determinism")


def test_grouped_split_no_video_leakage():
    """Test that grouped splits don't leak windows from same video."""
    from sklearn.model_selection import GroupKFold
    
    n_windows = 50
    n_videos = 5
    
    groups = np.repeat(np.arange(n_videos), n_windows // n_videos)
    labels = np.random.randint(0, 5, size=n_windows)
    
    kfold = GroupKFold(n_splits=3)
    
    for train_idx, val_idx in kfold.split(np.arange(n_windows), labels, groups):
        train_videos = set(groups[train_idx])
        val_videos = set(groups[val_idx])
        
        # No overlap between train and val videos
        assert len(train_videos & val_videos) == 0, "Video leakage detected!"
    
    print("✓ test_grouped_split_no_video_leakage")


# =============================================================================
# Unit Tests: Missing Modality Handling
# =============================================================================

def test_merge_modalities_missing_handling():
    """Test that missing modality predictions are filled correctly."""
    # Create window metadata
    window_meta = pd.DataFrame({
        "window_id": ["w1", "w2", "w3"],
        "video_key": ["v1", "v1", "v2"],
        "start_sec": [0.0, 1.0, 0.0],
        "end_sec": [2.0, 3.0, 2.0],
        "label_id": [0, 4, 1],
    })
    
    # Mod1 has all windows
    mod1_preds = pd.DataFrame({
        "window_id": ["w1", "w2", "w3"],
        "mod1_score_class0": [0.8, 0.1, 0.2],
        "mod1_score_class1": [0.1, 0.1, 0.6],
        "mod1_score_class2": [0.05, 0.1, 0.1],
        "mod1_score_class3": [0.03, 0.1, 0.05],
        "mod1_score_class4": [0.02, 0.6, 0.05],
    })
    
    # Mod2 is missing w2
    mod2_preds = pd.DataFrame({
        "window_id": ["w1", "w3"],
        "mod2_score_class0": [0.7, 0.3],
        "mod2_score_class1": [0.15, 0.5],
        "mod2_score_class2": [0.08, 0.1],
        "mod2_score_class3": [0.04, 0.05],
        "mod2_score_class4": [0.03, 0.05],
    })
    
    merged = merge_modalities(window_meta, mod1_preds, mod2_preds, num_classes=5)
    
    # Check all windows present
    assert len(merged) == 3
    
    # Check missing indicators
    assert merged[merged["window_id"] == "w1"]["mod2_missing"].values[0] == 0
    assert merged[merged["window_id"] == "w2"]["mod2_missing"].values[0] == 1
    assert merged[merged["window_id"] == "w3"]["mod2_missing"].values[0] == 0
    
    # Check missing values filled with uniform
    w2_row = merged[merged["window_id"] == "w2"].iloc[0]
    uniform = 1.0 / 5
    for i in range(5):
        assert abs(w2_row[f"mod2_score_class{i}"] - uniform) < 1e-6
    
    print("✓ test_merge_modalities_missing_handling")


def test_missing_modality_valid_probs():
    """Test that missing modality results in valid probability distribution."""
    # Create data where one modality is completely missing
    window_meta = pd.DataFrame({
        "window_id": ["w1"],
        "video_key": ["v1"],
        "start_sec": [0.0],
        "end_sec": [2.0],
        "label_id": [0],
    })
    
    mod1_preds = pd.DataFrame({
        "window_id": ["w1"],
        "mod1_score_class0": [0.8],
        "mod1_score_class1": [0.1],
        "mod1_score_class2": [0.05],
        "mod1_score_class3": [0.03],
        "mod1_score_class4": [0.02],
    })
    
    # Empty mod2
    mod2_preds = pd.DataFrame({
        "window_id": [],
        "mod2_score_class0": [],
        "mod2_score_class1": [],
        "mod2_score_class2": [],
        "mod2_score_class3": [],
        "mod2_score_class4": [],
    })
    
    merged = merge_modalities(window_meta, mod1_preds, mod2_preds, num_classes=5)
    
    # Check probabilities sum to 1
    row = merged.iloc[0]
    mod2_sum = sum(row[f"mod2_score_class{i}"] for i in range(5))
    assert abs(mod2_sum - 1.0) < 1e-6, f"Probabilities don't sum to 1: {mod2_sum}"
    
    print("✓ test_missing_modality_valid_probs")


# =============================================================================
# Unit Tests: Log-Prob Features
# =============================================================================

def test_log_prob_computation():
    """Test that log-prob features are computed correctly."""
    df = pd.DataFrame({
        "window_id": ["w1"],
        "video_key": ["v1"],
        "mod1_score_class0": [0.5],
        "mod1_score_class1": [0.3],
        "mod1_score_class2": [0.2],
        "mod2_score_class0": [0.4],
        "mod2_score_class1": [0.4],
        "mod2_score_class2": [0.2],
        "mod1_missing": [0.0],
        "mod2_missing": [0.0],
        "label_id": [0],
    })
    
    dataset = FusionDataset(df, num_classes=3, log_eps=1e-10)
    log_p1, log_p2, miss1, miss2, label, wid = dataset[0]
    
    # Check log-prob values
    expected_log_p1 = np.log(np.array([0.5, 0.3, 0.2]) + 1e-10)
    assert np.allclose(log_p1.numpy(), expected_log_p1, atol=1e-6)
    
    expected_log_p2 = np.log(np.array([0.4, 0.4, 0.2]) + 1e-10)
    assert np.allclose(log_p2.numpy(), expected_log_p2, atol=1e-6)
    
    print("✓ test_log_prob_computation")


# =============================================================================
# Unit Tests: MLP Model
# =============================================================================

def test_mlp_forward_pass():
    """Test MLP forward pass shape and validity."""
    model = LogProbMLPFusion(num_classes=5, hidden_dim=16, dropout=0.1)
    
    batch_size = 4
    log_p1 = torch.randn(batch_size, 5)
    log_p2 = torch.randn(batch_size, 5)
    miss1 = torch.zeros(batch_size, 1)
    miss2 = torch.zeros(batch_size, 1)
    
    output = model(log_p1, log_p2, miss1, miss2)
    
    assert output.shape == (batch_size, 5), f"Expected (4, 5), got {output.shape}"
    assert torch.isfinite(output).all(), "Output contains NaN/Inf"
    
    print("✓ test_mlp_forward_pass")


def test_mlp_parameter_count():
    """Test MLP parameter count matches expected."""
    model = LogProbMLPFusion(num_classes=5, hidden_dim=16, dropout=0.1)
    
    # Input: 2*5 + 2 = 12
    # fc1: 12 * 16 + 16 = 208
    # fc2: 16 * 5 + 5 = 85
    # Total: 293
    expected_params = (12 * 16 + 16) + (16 * 5 + 5)
    
    assert model.num_parameters == expected_params, \
        f"Expected {expected_params} params, got {model.num_parameters}"
    
    print("✓ test_mlp_parameter_count")


# =============================================================================
# Unit Tests: Class Weights
# =============================================================================

def test_class_weights_computation():
    """Test class weight computation for imbalanced data."""
    labels = np.array([0, 0, 0, 0, 1, 1, 2, 4, 4, 4, 4, 4])
    weights = compute_class_weights(labels, num_classes=5)
    
    # Check weights are positive
    assert (weights > 0).all()
    
    # Check rare classes have higher weight
    # Class 2 appears once, class 4 appears 5 times
    assert weights[2] > weights[4], "Rare class should have higher weight"
    
    print("✓ test_class_weights_computation")


def test_class_weights_missing_class():
    """Test class weight computation handles missing classes."""
    labels = np.array([0, 0, 1, 1])  # Missing classes 2, 3, 4
    weights = compute_class_weights(labels, num_classes=5)
    
    assert len(weights) == 5
    assert (weights > 0).all(), "All weights should be positive"
    
    print("✓ test_class_weights_missing_class")


# =============================================================================
# Integration Test: OOF Pipeline
# =============================================================================

def test_oof_fusion_coverage():
    """Test that OOF fusion produces exactly one prediction per input window."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic data
        window_df = create_synthetic_window_splits_csv(n_windows=50, n_videos=5)
        window_csv = tmpdir / "windows.csv"
        window_df.to_csv(window_csv, index=False)
        
        window_ids = window_df["window_id"].tolist()
        
        # Create mod1 with all windows
        mod1_df = create_synthetic_tal_preds_csv(window_ids, num_classes=5)
        mod1_csv = tmpdir / "mod1.csv"
        mod1_df.to_csv(mod1_csv, index=False)
        
        # Create mod2 with only 80% of windows
        mod2_ids = window_ids[:int(len(window_ids) * 0.8)]
        mod2_df = create_synthetic_tal_preds_csv(mod2_ids, num_classes=5)
        mod2_csv = tmpdir / "mod2.csv"
        mod2_df.to_csv(mod2_csv, index=False)
        
        # Run OOF fusion with reduced epochs for speed
        config = FusionConfig(
            num_classes=5,
            hidden_dim=8,
            num_epochs=5,
            n_inner_folds=3,
            random_seed=42,
        )
        
        output_df, stats = run_oof_fusion(
            window_splits_csv=window_csv,
            modality1_csv=mod1_csv,
            modality2_csv=mod2_csv,
            config=config,
        )
        
        # Check coverage: one row per input window
        assert len(output_df) == len(window_df), \
            f"Expected {len(window_df)} rows, got {len(output_df)}"
        
        # Check all window IDs present
        output_ids = set(output_df["window_id"])
        expected_ids = set(window_df["window_id"])
        assert output_ids == expected_ids, "Window ID mismatch"
        
        # Check probabilities are valid
        for i in range(5):
            col = f"score_class{i}"
            assert (output_df[col] >= 0).all(), f"{col} has negative values"
            assert (output_df[col] <= 1).all(), f"{col} has values > 1"
        
        # Check probs sum to ~1
        prob_sums = output_df[[f"score_class{i}" for i in range(5)]].sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probabilities don't sum to 1"
        
        print("✓ test_oof_fusion_coverage")


def test_oof_fusion_determinism():
    """Test that OOF fusion is deterministic with same seed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create data
        window_df = create_synthetic_window_splits_csv(n_windows=30, n_videos=3)
        window_csv = tmpdir / "windows.csv"
        window_df.to_csv(window_csv, index=False)
        
        window_ids = window_df["window_id"].tolist()
        mod1_df = create_synthetic_tal_preds_csv(window_ids, num_classes=5)
        mod1_csv = tmpdir / "mod1.csv"
        mod1_df.to_csv(mod1_csv, index=False)
        
        mod2_df = create_synthetic_tal_preds_csv(window_ids, num_classes=5)
        mod2_csv = tmpdir / "mod2.csv"
        mod2_df.to_csv(mod2_csv, index=False)
        
        config = FusionConfig(
            num_classes=5,
            hidden_dim=8,
            num_epochs=3,
            n_inner_folds=2,
            random_seed=42,
        )
        
        # Run twice
        output1, _ = run_oof_fusion(window_csv, mod1_csv, mod2_csv, config)
        output2, _ = run_oof_fusion(window_csv, mod1_csv, mod2_csv, config)
        
        # Compare probabilities
        for i in range(5):
            col = f"score_class{i}"
            assert np.allclose(output1[col].values, output2[col].values, atol=1e-6), \
                f"Non-deterministic: {col} differs between runs"
        
        print("✓ test_oof_fusion_determinism")


# =============================================================================
# Main
# =============================================================================

def run_all_tests(verbose: bool = True):
    """Run all unit tests."""
    if verbose:
        print("=" * 60)
        print("Running TAL Fusion Unit Tests")
        print("=" * 60)
        print()
    
    # Schema tests
    if verbose:
        print("Schema validation tests:")
    test_tal_format_schema()
    test_window_metadata_label_mapping()
    if verbose:
        print()
    
    # Grouped split tests
    if verbose:
        print("Grouped split tests:")
    test_grouped_split_determinism()
    test_grouped_split_no_video_leakage()
    if verbose:
        print()
    
    # Missing modality tests
    if verbose:
        print("Missing modality tests:")
    test_merge_modalities_missing_handling()
    test_missing_modality_valid_probs()
    if verbose:
        print()
    
    # Feature computation tests
    if verbose:
        print("Feature computation tests:")
    test_log_prob_computation()
    if verbose:
        print()
    
    # MLP tests
    if verbose:
        print("MLP model tests:")
    test_mlp_forward_pass()
    test_mlp_parameter_count()
    if verbose:
        print()
    
    # Class weight tests
    if verbose:
        print("Class weight tests:")
    test_class_weights_computation()
    test_class_weights_missing_class()
    if verbose:
        print()
    
    # Integration tests
    if verbose:
        print("Integration tests:")
    test_oof_fusion_coverage()
    test_oof_fusion_determinism()
    if verbose:
        print()
    
    if verbose:
        print("=" * 60)
        print("All TAL fusion tests PASSED!")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run TAL fusion tests.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    try:
        run_all_tests(verbose=verbose)
    except AssertionError as e:
        print(f"\n✗ Test FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



