#!/usr/bin/env python3
"""
TAL Window-Level Late Fusion via Out-of-Fold MLP Training.

This module implements late fusion of two modalities (e.g., V-JEPA + PoseC3D) for
Temporal Action Localization. The fusion is learned on val-window predictions
using inner cross-validation (out-of-fold) to avoid train/test leakage.

Key features:
- Log-probability (PoE-style) feature space for fusion
- Grouped splits by video_key to avoid leakage across overlapping windows
- Missing modality handling with uniform distribution + indicator flag
- Outputs TAL-format fused window prediction CSVs

Example usage:
    from tal_fusion import run_oof_fusion
    
    fused_df = run_oof_fusion(
        window_splits_csv="/path/to/fold_0_val_windows.csv",
        modality1_csv="/path/to/vjepa/tal_format_preds.csv",
        modality2_csv="/path/to/posec3d/tal_format_preds.csv",
        num_classes=5,
        random_seed=42,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Epsilon for log-prob computation to avoid log(0)
LOG_EPS = 1e-10


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FusionConfig:
    """Configuration for TAL fusion training."""
    
    num_classes: int = 5
    """Number of classes (including background as class 4)."""
    
    hidden_dim: int = 16
    """Hidden dimension for MLP."""
    
    dropout: float = 0.1
    """Dropout rate."""
    
    num_epochs: int = 100
    """Number of training epochs."""
    
    lr: float = 0.01
    """Learning rate."""
    
    batch_size: int = 64
    """Batch size."""
    
    n_inner_folds: int = 5
    """Number of inner CV folds for OOF prediction."""
    
    random_seed: int = 42
    """Random seed for reproducibility."""
    
    log_eps: float = 1e-10
    """Epsilon for log-prob computation."""


# =============================================================================
# MLP Fusion Model
# =============================================================================

class LogProbMLPFusion(nn.Module):
    """
    Late fusion MLP operating on log-probability features.
    
    Architecture:
        - Input: concat(log_prob_mod1, log_prob_mod2, missing_flags) 
                 Shape: (2 * num_classes + 2,)
        - Hidden: Linear -> ReLU -> Dropout
        - Output: Linear (num_classes)
    
    The model learns to combine log-probabilities from two modalities,
    effectively implementing a learnable product-of-experts fusion.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        hidden_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Input: 2 * num_classes log-probs + 2 missing flags
        input_dim = 2 * num_classes + 2
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(
        self,
        log_prob1: torch.Tensor,
        log_prob2: torch.Tensor,
        missing1: torch.Tensor,
        missing2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            log_prob1: (batch, num_classes) log-probs from modality 1
            log_prob2: (batch, num_classes) log-probs from modality 2
            missing1: (batch, 1) missing indicator for modality 1
            missing2: (batch, 1) missing indicator for modality 2
        
        Returns:
            logits: (batch, num_classes) fused logits
        """
        # Concatenate all features
        x = torch.cat([log_prob1, log_prob2, missing1, missing2], dim=-1)
        
        # MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Dataset
# =============================================================================

class FusionDataset(Dataset):
    """
    Dataset for TAL fusion training.
    
    Expects a DataFrame with columns:
        - window_id
        - video_key
        - label_id (0-4 for 5-class, where 4 = background)
        - mod1_score_class0 ... mod1_score_class{N-1}
        - mod2_score_class0 ... mod2_score_class{N-1}
        - mod1_missing (0 or 1)
        - mod2_missing (0 or 1)
    """
    
    def __init__(self, df: pd.DataFrame, num_classes: int, log_eps: float = 1e-10):
        self.df = df.reset_index(drop=True)
        self.num_classes = num_classes
        self.log_eps = log_eps
        
        # Column names
        self.mod1_cols = [f"mod1_score_class{i}" for i in range(num_classes)]
        self.mod2_cols = [f"mod2_score_class{i}" for i in range(num_classes)]
        
        # Validate columns
        required = self.mod1_cols + self.mod2_cols + ["mod1_missing", "mod2_missing", "label_id"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        row = self.df.iloc[idx]
        
        # Get probabilities
        prob1 = np.array([row[c] for c in self.mod1_cols], dtype=np.float32)
        prob2 = np.array([row[c] for c in self.mod2_cols], dtype=np.float32)
        
        # Convert to log-probs
        log_prob1 = np.log(prob1 + self.log_eps)
        log_prob2 = np.log(prob2 + self.log_eps)
        
        # Missing flags
        missing1 = np.array([row["mod1_missing"]], dtype=np.float32)
        missing2 = np.array([row["mod2_missing"]], dtype=np.float32)
        
        # Label
        label = int(row["label_id"])
        
        return (
            torch.from_numpy(log_prob1),
            torch.from_numpy(log_prob2),
            torch.from_numpy(missing1),
            torch.from_numpy(missing2),
            label,
            str(row["window_id"]),
        )


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_window_metadata(
    window_splits_csv: Path,
    num_classes: int = 5,
) -> pd.DataFrame:
    """
    Load window metadata from splits CSV.
    
    Args:
        window_splits_csv: Path to fold_*_val_windows.csv
        num_classes: Number of classes (5 = 4 RMM + background)
    
    Returns:
        DataFrame with window_id, video_key, label_id columns
    """
    df = pd.read_csv(window_splits_csv)
    
    # Map primary_label to label_id
    # primary_label: -1 = background, 0-3 = RMM classes
    # label_id: 0-3 = RMM classes, 4 = background (for 5-class)
    df["label_id"] = df["primary_label"].apply(
        lambda x: num_classes - 1 if x == -1 else x
    )
    
    # Keep only needed columns
    result = df[["window_id", "video_key", "start_sec", "end_sec", "label_id"]].copy()
    
    logger.info(
        "Loaded %d windows from %s (%d videos)",
        len(result), window_splits_csv, result["video_key"].nunique()
    )
    
    return result


def load_tal_format_preds(
    csv_path: Path,
    num_classes: int = 5,
    prefix: str = "mod1",
) -> pd.DataFrame:
    """
    Load TAL-format prediction CSV and rename score columns.
    
    Args:
        csv_path: Path to tal_format_preds.csv
        num_classes: Number of classes
        prefix: Prefix for score columns (e.g., "mod1", "mod2")
    
    Returns:
        DataFrame with window_id and prefixed score columns
    """
    df = pd.read_csv(csv_path)
    
    # Rename score columns
    rename_map = {}
    for i in range(num_classes):
        old_col = f"score_class{i}"
        new_col = f"{prefix}_score_class{i}"
        if old_col in df.columns:
            rename_map[old_col] = new_col
    
    df = df.rename(columns=rename_map)
    
    # Keep only window_id and score columns
    keep_cols = ["window_id"] + [f"{prefix}_score_class{i}" for i in range(num_classes)]
    keep_cols = [c for c in keep_cols if c in df.columns]
    
    result = df[keep_cols].copy()
    
    logger.info("Loaded %d predictions from %s", len(result), csv_path)
    
    return result


def merge_modalities(
    window_meta: pd.DataFrame,
    mod1_preds: pd.DataFrame,
    mod2_preds: pd.DataFrame,
    num_classes: int = 5,
) -> pd.DataFrame:
    """
    Merge window metadata with predictions from both modalities.
    
    Missing predictions are filled with uniform distribution.
    
    Args:
        window_meta: DataFrame with window_id, video_key, label_id
        mod1_preds: Predictions from modality 1 (prefixed mod1_*)
        mod2_preds: Predictions from modality 2 (prefixed mod2_*)
        num_classes: Number of classes
    
    Returns:
        Merged DataFrame with all columns + missing indicators
    """
    # Start with window metadata (this defines the full set of windows)
    merged = window_meta.copy()
    
    # Left join modality 1
    merged = merged.merge(mod1_preds, on="window_id", how="left")
    
    # Left join modality 2
    merged = merged.merge(mod2_preds, on="window_id", how="left")
    
    # Create missing indicators
    mod1_first_col = f"mod1_score_class0"
    mod2_first_col = f"mod2_score_class0"
    
    merged["mod1_missing"] = merged[mod1_first_col].isna().astype(float)
    merged["mod2_missing"] = merged[mod2_first_col].isna().astype(float)
    
    # Fill missing values with uniform distribution
    uniform_prob = 1.0 / num_classes
    
    for i in range(num_classes):
        mod1_col = f"mod1_score_class{i}"
        mod2_col = f"mod2_score_class{i}"
        
        merged[mod1_col] = merged[mod1_col].fillna(uniform_prob)
        merged[mod2_col] = merged[mod2_col].fillna(uniform_prob)
    
    # Log statistics
    n_missing_mod1 = int(merged["mod1_missing"].sum())
    n_missing_mod2 = int(merged["mod2_missing"].sum())
    
    logger.info(
        "Merged %d windows: %d missing mod1 (%.1f%%), %d missing mod2 (%.1f%%)",
        len(merged),
        n_missing_mod1, 100 * n_missing_mod1 / len(merged),
        n_missing_mod2, 100 * n_missing_mod2 / len(merged),
    )
    
    return merged


# =============================================================================
# Training and Evaluation
# =============================================================================

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for balanced loss.
    
    Args:
        labels: Array of label indices
        num_classes: Number of classes
    
    Returns:
        Tensor of class weights
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # Avoid division by zero
    
    # Inverse frequency, normalized
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    
    return torch.tensor(weights, dtype=torch.float32)


def train_fusion_model(
    model: LogProbMLPFusion,
    train_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    num_epochs: int = 100,
    lr: float = 0.01,
) -> List[float]:
    """
    Train the fusion MLP.
    
    Args:
        model: Fusion model
        train_loader: Training data loader
        class_weights: Class weights for balanced loss
        device: Torch device
        num_epochs: Number of epochs
        lr: Learning rate
    
    Returns:
        List of per-epoch loss values
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    losses = []
    model.train()
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        for log_p1, log_p2, miss1, miss2, labels, _ in train_loader:
            log_p1 = log_p1.to(device)
            log_p2 = log_p2.to(device)
            miss1 = miss1.to(device)
            miss2 = miss2.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(log_p1, log_p2, miss1, miss2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
    
    return losses


def evaluate_fusion_model(
    model: LogProbMLPFusion,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    Evaluate fusion model and return predictions.
    
    Args:
        model: Trained fusion model
        loader: Data loader
        device: Torch device
    
    Returns:
        Dict with window_ids, labels, preds, and probs
    """
    model.eval()
    
    all_window_ids = []
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for log_p1, log_p2, miss1, miss2, labels, window_ids in loader:
            log_p1 = log_p1.to(device)
            log_p2 = log_p2.to(device)
            miss1 = miss1.to(device)
            miss2 = miss2.to(device)
            
            logits = model(log_p1, log_p2, miss1, miss2)
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            
            all_window_ids.extend(window_ids)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.append(probs.cpu().numpy())
    
    return {
        "window_ids": all_window_ids,
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.vstack(all_probs),
    }


# =============================================================================
# Out-of-Fold Fusion Pipeline
# =============================================================================

def run_oof_fusion(
    window_splits_csv: Path,
    modality1_csv: Path,
    modality2_csv: Path,
    config: Optional[FusionConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run out-of-fold fusion to produce fused TAL-format predictions.
    
    This function:
    1. Loads window metadata and predictions from both modalities
    2. Merges them, handling missing predictions
    3. Runs grouped inner-CV to train fusion MLP and get OOF predictions
    4. Returns a TAL-format DataFrame with fused probabilities
    
    Args:
        window_splits_csv: Path to fold_*_val_windows.csv
        modality1_csv: Path to modality 1 tal_format_preds.csv
        modality2_csv: Path to modality 2 tal_format_preds.csv
        config: Fusion configuration (default: FusionConfig())
        device: Torch device (default: auto-detect)
    
    Returns:
        Tuple of:
        - DataFrame in TAL format with fused probabilities
        - Dict with training statistics
    """
    if config is None:
        config = FusionConfig()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seed
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    logger.info("=" * 60)
    logger.info("TAL Out-of-Fold Fusion")
    logger.info("=" * 60)
    logger.info("Modality 1: %s", modality1_csv)
    logger.info("Modality 2: %s", modality2_csv)
    logger.info("Config: %s", config)
    
    # Load data
    window_meta = load_window_metadata(window_splits_csv, config.num_classes)
    mod1_preds = load_tal_format_preds(modality1_csv, config.num_classes, prefix="mod1")
    mod2_preds = load_tal_format_preds(modality2_csv, config.num_classes, prefix="mod2")
    
    # Merge
    merged_df = merge_modalities(window_meta, mod1_preds, mod2_preds, config.num_classes)
    
    # Prepare for OOF training
    groups = merged_df["video_key"].values
    labels = merged_df["label_id"].values
    
    group_kfold = GroupKFold(n_splits=config.n_inner_folds)
    
    # Storage for OOF predictions
    oof_probs = np.zeros((len(merged_df), config.num_classes))
    oof_preds = np.zeros(len(merged_df), dtype=int)
    fold_losses = []
    
    logger.info("\nRunning %d-fold grouped inner CV...", config.n_inner_folds)
    
    for fold_idx, (train_idx, val_idx) in enumerate(group_kfold.split(merged_df, labels, groups)):
        train_df = merged_df.iloc[train_idx].reset_index(drop=True)
        val_df = merged_df.iloc[val_idx].reset_index(drop=True)
        
        n_train_videos = train_df["video_key"].nunique()
        n_val_videos = val_df["video_key"].nunique()
        
        logger.info(
            "  Fold %d/%d: train=%d windows (%d videos), val=%d windows (%d videos)",
            fold_idx + 1, config.n_inner_folds,
            len(train_df), n_train_videos,
            len(val_df), n_val_videos,
        )
        
        # Create datasets and loaders
        train_dataset = FusionDataset(train_df, config.num_classes, config.log_eps)
        val_dataset = FusionDataset(val_df, config.num_classes, config.log_eps)
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        
        # Compute class weights on training set
        train_labels = train_df["label_id"].values
        class_weights = compute_class_weights(train_labels, config.num_classes)
        
        # Create and train model
        model = LogProbMLPFusion(
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        ).to(device)
        
        losses = train_fusion_model(
            model=model,
            train_loader=train_loader,
            class_weights=class_weights,
            device=device,
            num_epochs=config.num_epochs,
            lr=config.lr,
        )
        fold_losses.append(losses[-1])  # Final loss
        
        # Evaluate on validation set (OOF predictions)
        eval_results = evaluate_fusion_model(model, val_loader, device)
        
        # Store OOF predictions at original indices
        oof_probs[val_idx] = eval_results["probs"]
        oof_preds[val_idx] = eval_results["preds"]
        
        # Compute fold accuracy
        fold_acc = (eval_results["preds"] == eval_results["labels"]).mean()
        logger.info("    Final loss: %.4f, Val accuracy: %.4f", losses[-1], fold_acc)
    
    # Build output TAL-format DataFrame
    output_df = pd.DataFrame({
        "window_id": merged_df["window_id"],
        "start_sec": merged_df["start_sec"],
        "end_sec": merged_df["end_sec"],
        "video_key": merged_df["video_key"],
    })
    
    for i in range(config.num_classes):
        output_df[f"score_class{i}"] = oof_probs[:, i]
    
    # Compute overall OOF accuracy
    oof_acc = (oof_preds == labels).mean()
    
    stats = {
        "n_windows": len(merged_df),
        "n_videos": merged_df["video_key"].nunique(),
        "n_inner_folds": config.n_inner_folds,
        "oof_accuracy": float(oof_acc),
        "mean_final_loss": float(np.mean(fold_losses)),
        "config": {
            "num_classes": config.num_classes,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "num_epochs": config.num_epochs,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "random_seed": config.random_seed,
        },
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("OOF Fusion Complete")
    logger.info("=" * 60)
    logger.info("Total windows: %d", stats["n_windows"])
    logger.info("OOF accuracy: %.4f", stats["oof_accuracy"])
    logger.info("Mean final loss: %.4f", stats["mean_final_loss"])
    
    return output_df, stats


def save_fused_predictions(
    output_df: pd.DataFrame,
    stats: Dict,
    out_dir: Path,
) -> None:
    """
    Save fused predictions and statistics.
    
    Args:
        output_df: TAL-format DataFrame with fused probabilities
        stats: Training statistics dict
        out_dir: Output directory
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save TAL-format predictions
    csv_path = out_dir / "tal_format_preds.csv"
    output_df.to_csv(csv_path, index=False)
    logger.info("Saved fused predictions to: %s", csv_path)
    
    # Save statistics
    stats_path = out_dir / "fusion_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved fusion stats to: %s", stats_path)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for TAL fusion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run TAL window-level late fusion via OOF MLP training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--window-splits-csv",
        type=Path,
        required=True,
        help="Path to fold_*_val_windows.csv with window metadata and labels.",
    )
    parser.add_argument(
        "--modality1-csv",
        type=Path,
        required=True,
        help="Path to modality 1 tal_format_preds.csv.",
    )
    parser.add_argument(
        "--modality2-csv",
        type=Path,
        required=True,
        help="Path to modality 2 tal_format_preds.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for fused predictions.",
    )
    
    # Config options
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-inner-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Build config
    config = FusionConfig(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_inner_folds=args.n_inner_folds,
        random_seed=args.random_seed,
    )
    
    # Run fusion
    output_df, stats = run_oof_fusion(
        window_splits_csv=args.window_splits_csv,
        modality1_csv=args.modality1_csv,
        modality2_csv=args.modality2_csv,
        config=config,
    )
    
    # Save outputs
    save_fused_predictions(output_df, stats, args.out_dir)
    
    print(f"\nDone! Fused predictions saved to: {args.out_dir}")


if __name__ == "__main__":
    main()



