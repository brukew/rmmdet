#!/usr/bin/env python
"""
Late Fusion Training: V-JEPA2 (RGB) + PoseC3D (Skeleton)

Trains a learned scalar α for fusing logits from two modalities:
    z_fused = σ(α) * z_rgb + (1 - σ(α)) * z_pose

Where:
    - z_rgb: V-JEPA2 logits (from clip_level_preds.csv)
    - z_pose: PoseC3D logits (from predictions_clip.csv)
    - α: Single learnable parameter (sigmoid-constrained)

The encoders are frozen; only α is trained with class-balanced cross-entropy loss.

Usage:
    python train_fusion_cv.py --help
    python train_fusion_cv.py --num-folds 3 --output-dir runs/4class_cv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Resolve the repo root (directory containing paths.py) so defaults work from any clone.
try:
    _REPO = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())
except StopIteration:
    _REPO = Path(__file__).resolve().parent.parent

DEFAULT_VJEPA_ROOT = _REPO / "v-jepa" / "runs" / "vjepa2_rmm_cv" / "f64_lr1e-5_bs1_acc8_ep20_crop_4cls"
# Use non-weighted PoseC3D which has full val set coverage (205 clips vs 52 for weighted)
DEFAULT_POSEC3D_ROOT = _REPO / "pyskl" / "work_dirs" / "posec3d" / "cv" / "4class_conf04"
# STGCN++ 4-stream fusion (best performing skeleton model)
DEFAULT_STGCN_ROOT = _REPO / "pyskl" / "work_dirs" / "stgcnpp" / "cv" / "4class_conf04_4stream"
DEFAULT_OUTPUT_DIR = _REPO / "fusion" / "runs" / "4class_cv"
DEFAULT_ISSUE_CLIPS_CSV = _REPO / "dataprep" / "issue_clips.csv"

CLASS_NAMES_4CLASS = ["hands flapping", "jumping", "rocking", "spinning"]
CLASS_NAMES_5CLASS = ["hands flapping", "jumping", "one hand flap", "rocking", "spinning"]


def load_issue_clips(csv_path: Optional[Path] = None) -> set:
    """Load set of segment IDs to exclude from training/evaluation."""
    if csv_path is None:
        csv_path = DEFAULT_ISSUE_CLIPS_CSV
    if not csv_path.exists():
        logger.warning("Issue clips CSV not found at %s; no clips will be excluded.", csv_path)
        return set()
    
    df = pd.read_csv(csv_path)
    issue_clips = set(df["segment_id"].astype(str).tolist())
    logger.info("Loaded %d issue clips to exclude from %s", len(issue_clips), csv_path)
    return issue_clips


# =============================================================================
# Data Loading
# =============================================================================

def load_vjepa_predictions(fold_dir: Path) -> pd.DataFrame:
    """
    Load V-JEPA2 clip-level predictions.
    
    Expected columns: segment_id, label_id, pred_top1, score_class0, score_class1, ...
    """
    csv_path = fold_dir / "clip_level_preds.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"V-JEPA2 predictions not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check for per-class scores
    score_cols = [c for c in df.columns if c.startswith("score_class")]
    if not score_cols:
        raise ValueError(
            f"V-JEPA2 predictions missing per-class scores. "
            f"Please re-run evaluation with updated finetune_sails_vjepa2_cv_crop.py. "
            f"Available columns: {list(df.columns)}"
        )
    
    logger.info("Loaded V-JEPA2 predictions from %s: %d clips, %d classes", csv_path, len(df), len(score_cols))
    return df


def load_skeleton_predictions(fold_dir: Path, model_name: str = "skeleton") -> pd.DataFrame:
    """
    Load skeleton-based model (PoseC3D or STGCN++) clip-level predictions.
    
    Handles different folder structures:
        - PoseC3D: fold_dir/eval_val/predictions_clip.csv
        - STGCN++: fold_dir/predictions_clip.csv
    
    Expected columns: segment_id, true_label, pred_label, score_class0, score_class1, ...
    """
    # Try different possible paths (STGCN++ vs PoseC3D folder structure)
    possible_paths = [
        fold_dir / "predictions_clip.csv",  # STGCN++ structure
        fold_dir / "eval_val" / "predictions_clip.csv",  # PoseC3D structure
    ]
    
    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break
    
    if csv_path is None:
        raise FileNotFoundError(
            f"{model_name} predictions not found. Tried: {[str(p) for p in possible_paths]}"
        )
    
    df = pd.read_csv(csv_path)
    
    # Check for per-class scores
    score_cols = [c for c in df.columns if c.startswith("score_class")]
    if not score_cols:
        raise ValueError(f"{model_name} predictions missing per-class scores. Available columns: {list(df.columns)}")
    
    logger.info("Loaded %s predictions from %s: %d clips, %d classes", model_name, csv_path, len(df), len(score_cols))
    return df


# Backwards compatibility alias
def load_posec3d_predictions(fold_dir: Path) -> pd.DataFrame:
    """Load PoseC3D predictions (alias for load_skeleton_predictions)."""
    return load_skeleton_predictions(fold_dir, model_name="PoseC3D")


def merge_predictions(
    vjepa_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    num_classes: int,
    exclude_clips: Optional[set] = None,
) -> pd.DataFrame:
    """
    Merge V-JEPA2 and PoseC3D predictions on segment_id.
    
    Returns DataFrame with columns:
        - segment_id, label_id
        - vjepa_score_class0, vjepa_score_class1, ...
        - pose_score_class0, pose_score_class1, ...
        - video_id (if available)
    """
    # Rename score columns to distinguish modalities
    vjepa_cols = {f"score_class{i}": f"vjepa_score_class{i}" for i in range(num_classes)}
    pose_cols = {f"score_class{i}": f"pose_score_class{i}" for i in range(num_classes)}
    
    vjepa_renamed = vjepa_df.rename(columns=vjepa_cols)
    pose_renamed = pose_df.rename(columns=pose_cols)
    
    # Standardize label column names
    if "label_id" not in vjepa_renamed.columns and "true_label" in pose_renamed.columns:
        # Use PoseC3D's true_label as ground truth
        pass
    
    # Select columns to merge
    vjepa_keep = ["segment_id", "label_id", "video_id"] + list(vjepa_cols.values())
    vjepa_keep = [c for c in vjepa_keep if c in vjepa_renamed.columns]
    
    pose_keep = ["segment_id", "true_label"] + list(pose_cols.values())
    pose_keep = [c for c in pose_keep if c in pose_renamed.columns]
    
    merged = pd.merge(
        vjepa_renamed[vjepa_keep],
        pose_renamed[pose_keep],
        on="segment_id",
        how="inner",
    )
    
    # Use label_id from V-JEPA2 if available, else use true_label from PoseC3D
    if "label_id" not in merged.columns and "true_label" in merged.columns:
        merged["label_id"] = merged["true_label"]
    
    # Filter out issue clips
    if exclude_clips:
        before_count = len(merged)
        merged = merged[~merged["segment_id"].astype(str).isin(exclude_clips)]
        excluded = before_count - len(merged)
        if excluded > 0:
            logger.info("Excluded %d issue clips from merged predictions", excluded)
    
    logger.info(
        "Merged predictions: %d clips (V-JEPA: %d, PoseC3D: %d, intersection: %d)",
        len(merged), len(vjepa_df), len(pose_df), len(merged)
    )
    
    if len(merged) < len(vjepa_df) * 0.95:
        logger.warning(
            "Significant mismatch in predictions! Only %.1f%% overlap.",
            100 * len(merged) / len(vjepa_df)
        )
    
    return merged


def merge_three_predictions(
    vjepa_df: pd.DataFrame,
    posec3d_df: pd.DataFrame,
    stgcn_df: pd.DataFrame,
    num_classes: int,
    exclude_clips: Optional[set] = None,
) -> pd.DataFrame:
    """
    Merge V-JEPA2, PoseC3D, and STGCN++ predictions on segment_id.
    
    Returns DataFrame with columns:
        - segment_id, label_id
        - vjepa_score_class0, vjepa_score_class1, ...
        - posec3d_score_class0, posec3d_score_class1, ...
        - stgcn_score_class0, stgcn_score_class1, ...
        - video_id (if available)
    """
    # Rename score columns to distinguish modalities
    vjepa_cols = {f"score_class{i}": f"vjepa_score_class{i}" for i in range(num_classes)}
    posec3d_cols = {f"score_class{i}": f"posec3d_score_class{i}" for i in range(num_classes)}
    stgcn_cols = {f"score_class{i}": f"stgcn_score_class{i}" for i in range(num_classes)}
    
    vjepa_renamed = vjepa_df.rename(columns=vjepa_cols)
    posec3d_renamed = posec3d_df.rename(columns=posec3d_cols)
    stgcn_renamed = stgcn_df.rename(columns=stgcn_cols)
    
    # Select columns to merge
    vjepa_keep = ["segment_id", "label_id", "video_id"] + list(vjepa_cols.values())
    vjepa_keep = [c for c in vjepa_keep if c in vjepa_renamed.columns]
    
    posec3d_keep = ["segment_id", "true_label"] + list(posec3d_cols.values())
    posec3d_keep = [c for c in posec3d_keep if c in posec3d_renamed.columns]
    
    stgcn_keep = ["segment_id"] + list(stgcn_cols.values())
    stgcn_keep = [c for c in stgcn_keep if c in stgcn_renamed.columns]
    
    # Merge all three
    merged = pd.merge(
        vjepa_renamed[vjepa_keep],
        posec3d_renamed[posec3d_keep],
        on="segment_id",
        how="inner",
    )
    merged = pd.merge(
        merged,
        stgcn_renamed[stgcn_keep],
        on="segment_id",
        how="inner",
    )
    
    # Use label_id from V-JEPA2 if available, else use true_label from PoseC3D
    if "label_id" not in merged.columns and "true_label" in merged.columns:
        merged["label_id"] = merged["true_label"]
    
    # Filter out issue clips
    if exclude_clips:
        before_count = len(merged)
        merged = merged[~merged["segment_id"].astype(str).isin(exclude_clips)]
        excluded = before_count - len(merged)
        if excluded > 0:
            logger.info("Excluded %d issue clips from merged predictions", excluded)
    
    logger.info(
        "Merged 3-way predictions: %d clips (V-JEPA: %d, PoseC3D: %d, STGCN++: %d, intersection: %d)",
        len(merged), len(vjepa_df), len(posec3d_df), len(stgcn_df), len(merged)
    )
    
    if len(merged) < len(vjepa_df) * 0.90:
        logger.warning(
            "Significant mismatch in 3-way predictions! Only %.1f%% overlap.",
            100 * len(merged) / len(vjepa_df)
        )
    
    return merged


# =============================================================================
# Fusion Model
# =============================================================================

class ScalarLogitFusion(nn.Module):
    """
    Late fusion with a single learned scalar α.
    
    z_fused = σ(α) * z_rgb + (1 - σ(α)) * z_pose
    
    Where:
        - α is initialized to 0 (so σ(0) = 0.5, equal weighting)
        - Sigmoid constrains α weight to [0, 1]
    """
    
    def __init__(self, init_alpha: float = 0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
    
    def forward(self, z_rgb: torch.Tensor, z_pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_rgb: (batch, num_classes) - V-JEPA2 logits/probs
            z_pose: (batch, num_classes) - PoseC3D logits/probs
        
        Returns:
            z_fused: (batch, num_classes) - Fused logits/probs
        """
        alpha = torch.sigmoid(self.alpha)
        return alpha * z_rgb + (1 - alpha) * z_pose
    
    @property
    def alpha_value(self) -> float:
        """Return the effective α weight (after sigmoid)."""
        with torch.no_grad():
            return torch.sigmoid(self.alpha).item()


class PerClassLogitFusion(nn.Module):
    """
    Late fusion with per-class learned α vector.
    
    z_fused[c] = σ(α[c]) * z_rgb[c] + (1 - σ(α[c])) * z_pose[c]
    
    Where:
        - α is a vector of size num_classes
        - Each class has its own fusion weight
        - Allows different modality preferences per action type
    """
    
    def __init__(self, num_classes: int = 4, init_alpha: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = nn.Parameter(torch.full((num_classes,), init_alpha))
    
    def forward(self, z_rgb: torch.Tensor, z_pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_rgb: (batch, num_classes) - V-JEPA2 logits/probs
            z_pose: (batch, num_classes) - skeleton model logits/probs
        
        Returns:
            z_fused: (batch, num_classes) - Fused logits/probs
        """
        alpha = torch.sigmoid(self.alpha)  # [num_classes]
        return alpha * z_rgb + (1 - alpha) * z_pose
    
    @property
    def alpha_value(self) -> float:
        """Return mean α weight (for compatibility with scalar version)."""
        with torch.no_grad():
            return torch.sigmoid(self.alpha).mean().item()
    
    @property
    def alpha_per_class(self) -> List[float]:
        """Return per-class α weights (after sigmoid)."""
        with torch.no_grad():
            return torch.sigmoid(self.alpha).tolist()


class MLPLogitFusion(nn.Module):
    """
    Late fusion with a small MLP.
    
    z_fused = MLP(concat(z_rgb, z_pose))
    
    Architecture:
        - Input: concatenated logits [z_rgb; z_pose] (2 * num_classes)
        - Hidden: Linear -> ReLU (hidden_dim)
        - Output: Linear (num_classes)
    
    Can learn non-linear interactions between modalities.
    """
    
    def __init__(self, num_classes: int = 4, hidden_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(num_classes * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, z_rgb: torch.Tensor, z_pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_rgb: (batch, num_classes) - V-JEPA2 logits/probs
            z_pose: (batch, num_classes) - skeleton model logits/probs
        
        Returns:
            z_fused: (batch, num_classes) - Fused logits/probs
        """
        x = torch.cat([z_rgb, z_pose], dim=-1)  # [batch, 2 * num_classes]
        x = F.relu(self.fc1(x))                  # [batch, hidden_dim]
        x = self.dropout(x)
        return self.fc2(x)                       # [batch, num_classes]
    
    @property
    def alpha_value(self) -> float:
        """MLP doesn't have a single α; return 0.5 as placeholder."""
        return 0.5
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ThreeWayMLPFusion(nn.Module):
    """
    Late fusion with a small MLP for 3 modalities.
    
    z_fused = MLP(concat(z_rgb, z_posec3d, z_stgcn))
    
    Architecture:
        - Input: concatenated logits [z_rgb; z_posec3d; z_stgcn] (3 * num_classes)
        - Hidden: Linear -> ReLU (hidden_dim)
        - Output: Linear (num_classes)
    
    Can learn non-linear interactions between all three modalities.
    """
    
    def __init__(self, num_classes: int = 4, hidden_dim: int = 24, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 3 modalities -> hidden -> output
        self.fc1 = nn.Linear(num_classes * 3, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(
        self, 
        z_rgb: torch.Tensor, 
        z_posec3d: torch.Tensor,
        z_stgcn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_rgb: (batch, num_classes) - V-JEPA2 logits/probs
            z_posec3d: (batch, num_classes) - PoseC3D logits/probs
            z_stgcn: (batch, num_classes) - STGCN++ logits/probs
        
        Returns:
            z_fused: (batch, num_classes) - Fused logits/probs
        """
        x = torch.cat([z_rgb, z_posec3d, z_stgcn], dim=-1)  # [batch, 3 * num_classes]
        x = F.relu(self.fc1(x))                              # [batch, hidden_dim]
        x = self.dropout(x)
        return self.fc2(x)                                   # [batch, num_classes]
    
    @property
    def alpha_value(self) -> str:
        """3-way MLP doesn't have α; return placeholder."""
        return "3-way"
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory function to create fusion model
def create_fusion_model(
    fusion_type: str,
    num_classes: int = 4,
    hidden_dim: int = 16,
    dropout: float = 0.1,
    init_alpha: float = 0.0,
) -> nn.Module:
    """
    Create a fusion model based on the specified type.
    
    Args:
        fusion_type: One of 'scalar', 'per_class', 'mlp', 'three_way'
        num_classes: Number of output classes
        hidden_dim: Hidden dimension for MLP (only used if fusion_type='mlp' or 'three_way')
        dropout: Dropout rate for MLP (only used if fusion_type='mlp' or 'three_way')
        init_alpha: Initial alpha value (only used for scalar/per_class)
    
    Returns:
        Fusion model instance
    """
    if fusion_type == "scalar":
        return ScalarLogitFusion(init_alpha=init_alpha)
    elif fusion_type == "per_class":
        return PerClassLogitFusion(num_classes=num_classes, init_alpha=init_alpha)
    elif fusion_type == "mlp":
        return MLPLogitFusion(num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
    elif fusion_type == "three_way":
        # Use larger hidden dim for 3-way fusion (3 inputs vs 2)
        return ThreeWayMLPFusion(num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}. Choose from: scalar, per_class, mlp, three_way")


# =============================================================================
# Dataset
# =============================================================================

class FusionDataset(Dataset):
    """Dataset for late fusion training from pre-computed predictions."""
    
    def __init__(self, df: pd.DataFrame, num_classes: int):
        self.df = df.reset_index(drop=True)
        self.num_classes = num_classes
        
        # Extract score columns
        self.vjepa_cols = [f"vjepa_score_class{i}" for i in range(num_classes)]
        self.pose_cols = [f"pose_score_class{i}" for i in range(num_classes)]
        
        # Validate columns exist
        for col in self.vjepa_cols + self.pose_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        
        z_vjepa = torch.tensor([row[c] for c in self.vjepa_cols], dtype=torch.float32)
        z_pose = torch.tensor([row[c] for c in self.pose_cols], dtype=torch.float32)
        label = int(row["label_id"])
        segment_id = str(row["segment_id"])
        
        return z_vjepa, z_pose, label, segment_id


class ThreeWayFusionDataset(Dataset):
    """Dataset for 3-way late fusion training from pre-computed predictions."""
    
    def __init__(self, df: pd.DataFrame, num_classes: int):
        self.df = df.reset_index(drop=True)
        self.num_classes = num_classes
        
        # Extract score columns for all 3 modalities
        self.vjepa_cols = [f"vjepa_score_class{i}" for i in range(num_classes)]
        self.posec3d_cols = [f"posec3d_score_class{i}" for i in range(num_classes)]
        self.stgcn_cols = [f"stgcn_score_class{i}" for i in range(num_classes)]
        
        # Validate columns exist
        for col in self.vjepa_cols + self.posec3d_cols + self.stgcn_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        
        z_vjepa = torch.tensor([row[c] for c in self.vjepa_cols], dtype=torch.float32)
        z_posec3d = torch.tensor([row[c] for c in self.posec3d_cols], dtype=torch.float32)
        z_stgcn = torch.tensor([row[c] for c in self.stgcn_cols], dtype=torch.float32)
        label = int(row["label_id"])
        segment_id = str(row["segment_id"])
        
        return z_vjepa, z_posec3d, z_stgcn, label, segment_id


# =============================================================================
# Training
# =============================================================================

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        count = max(counts.get(i, 0), 1)  # Avoid div-by-zero
        weights.append(total / (num_classes * count))
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info("Class weights: %s", {i: f"{w:.3f}" for i, w in enumerate(weights)})
    return weights_tensor


def train_fusion(
    model: ScalarLogitFusion,
    train_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    num_epochs: int = 100,
    lr: float = 0.01,
    log_interval: int = 10,
) -> List[float]:
    """
    Train the fusion α parameter.
    
    Returns list of loss values per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    losses = []
    model.train()
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        for z_vjepa, z_pose, labels, _ in train_loader:
            z_vjepa = z_vjepa.to(device)
            z_pose = z_pose.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            z_fused = model(z_vjepa, z_pose)
            loss = criterion(z_fused, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
        if epoch % log_interval == 0 or epoch == num_epochs:
            # Handle different model types for logging
            if hasattr(model, "alpha_per_class"):
                alpha_str = ", ".join([f"{a:.3f}" for a in model.alpha_per_class])
                logger.info(
                    "Epoch %3d/%d | Loss: %.4f | α_per_class = [%s]",
                    epoch, num_epochs, avg_loss, alpha_str
                )
            elif hasattr(model, "alpha"):
                logger.info(
                    "Epoch %3d/%d | Loss: %.4f | α = %.4f (raw: %.4f)",
                    epoch, num_epochs, avg_loss, model.alpha_value, model.alpha.item()
                )
            else:
                # MLP model
                logger.info(
                    "Epoch %3d/%d | Loss: %.4f | (MLP)",
                    epoch, num_epochs, avg_loss
                )
    
    return losses


def train_three_way_fusion(
    model: ThreeWayMLPFusion,
    train_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    num_epochs: int = 100,
    lr: float = 0.01,
    log_interval: int = 10,
) -> List[float]:
    """
    Train the 3-way fusion MLP.
    
    Returns list of loss values per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    losses = []
    model.train()
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        for z_vjepa, z_posec3d, z_stgcn, labels, _ in train_loader:
            z_vjepa = z_vjepa.to(device)
            z_posec3d = z_posec3d.to(device)
            z_stgcn = z_stgcn.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            z_fused = model(z_vjepa, z_posec3d, z_stgcn)
            loss = criterion(z_fused, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
        if epoch % log_interval == 0 or epoch == num_epochs:
            logger.info(
                "Epoch %3d/%d | Loss: %.4f | (3-way MLP)",
                epoch, num_epochs, avg_loss
            )
    
    return losses


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_fusion(
    model: ScalarLogitFusion,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Evaluate fusion model on a dataset."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_segment_ids = []
    
    with torch.no_grad():
        for z_vjepa, z_pose, labels, segment_ids in loader:
            z_vjepa = z_vjepa.to(device)
            z_pose = z_pose.to(device)
            
            z_fused = model(z_vjepa, z_pose)
            probs = F.softmax(z_fused, dim=-1)
            preds = probs.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs.cpu())
            all_segment_ids.extend(segment_ids)
    
    all_probs = torch.cat(all_probs, dim=0).numpy()
    
    return {
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": all_probs,
        "segment_ids": all_segment_ids,
    }


def evaluate_three_way_fusion(
    model: ThreeWayMLPFusion,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Evaluate 3-way fusion model on a dataset."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_segment_ids = []
    
    with torch.no_grad():
        for z_vjepa, z_posec3d, z_stgcn, labels, segment_ids in loader:
            z_vjepa = z_vjepa.to(device)
            z_posec3d = z_posec3d.to(device)
            z_stgcn = z_stgcn.to(device)
            
            z_fused = model(z_vjepa, z_posec3d, z_stgcn)
            probs = F.softmax(z_fused, dim=-1)
            preds = probs.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs.cpu())
            all_segment_ids.extend(segment_ids)
    
    all_probs = torch.cat(all_probs, dim=0).numpy()
    
    return {
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": all_probs,
        "segment_ids": all_segment_ids,
    }


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, class_names: List[str]) -> Dict:
    """Compute comprehensive classification metrics."""
    num_classes = len(class_names)
    
    # Basic accuracy
    top1_acc = float((labels == preds).mean())
    
    # Top-2 accuracy
    top2_preds = np.argsort(probs, axis=1)[:, -2:]
    top2_correct = np.array([labels[i] in top2_preds[i] for i in range(len(labels))])
    top2_acc = float(top2_correct.mean())
    
    # Classification report
    report = classification_report(
        labels, preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    
    # Precision/recall/F1
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    
    # Cohen's kappa
    kappa = cohen_kappa_score(labels, preds)
    
    # Aggregate metrics
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    
    return {
        "top1_acc": top1_acc,
        "top2_acc": top2_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "cohen_kappa": kappa,
        "per_class": {
            class_names[i]: {
                "precision": prec[i],
                "recall": rec[i],
                "f1": f1[i],
                "support": int(support[i]),
            }
            for i in range(num_classes)
        },
    }


def average_metrics(results_list: List[Dict]) -> Dict:
    """
    Average metrics across inner CV folds.
    
    Args:
        results_list: List of metric dictionaries from each inner fold
    
    Returns:
        Dictionary with averaged metrics and their standard deviations
    """
    if not results_list:
        return {}
    
    # Get numeric keys from first result (skip nested dicts like "per_class")
    numeric_keys = [
        k for k, v in results_list[0].items()
        if isinstance(v, (int, float)) and k != "alpha"
    ]
    
    averaged = {}
    for key in numeric_keys:
        values = [r[key] for r in results_list if key in r and isinstance(r[key], (int, float))]
        if values:
            averaged[key] = float(np.mean(values))
            averaged[f"{key}_inner_std"] = float(np.std(values))
    
    return averaged


def aggregate_video_predictions(
    df: pd.DataFrame,
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    segment_ids: List[str],
) -> Optional[Dict]:
    """Aggregate clip-level predictions to video-level using mean probabilities."""
    if "video_id" not in df.columns:
        return None
    
    # Create results DataFrame
    results = pd.DataFrame({
        "segment_id": segment_ids,
        "pred": preds,
        "label": labels,
    })
    for i in range(probs.shape[1]):
        results[f"prob_class{i}"] = probs[:, i]
    
    # Merge with video_id
    results = results.merge(
        df[["segment_id", "video_id"]].drop_duplicates(),
        on="segment_id",
        how="left",
    )
    
    if results["video_id"].isna().all():
        return None
    
    # Group by video and aggregate
    video_results = []
    for video_id, group in results.groupby("video_id"):
        if pd.isna(video_id):
            continue
        
        # Average probabilities across clips
        prob_cols = [f"prob_class{i}" for i in range(probs.shape[1])]
        mean_probs = group[prob_cols].mean().values
        video_pred = int(np.argmax(mean_probs))
        video_label = int(group["label"].mode().iloc[0])  # Most common label
        
        video_results.append({
            "video_id": video_id,
            "pred": video_pred,
            "label": video_label,
            "n_clips": len(group),
            **{f"prob_class{i}": mean_probs[i] for i in range(len(mean_probs))},
        })
    
    if not video_results:
        return None
    
    video_df = pd.DataFrame(video_results)
    return {
        "df": video_df,
        "preds": video_df["pred"].values,
        "labels": video_df["label"].values,
        "probs": np.stack([video_df[f"prob_class{i}"].values for i in range(probs.shape[1])], axis=1),
    }


# =============================================================================
# Visualization
# =============================================================================

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: Path,
    normalize: str = "true",
) -> None:
    """Save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        display_labels=class_names,
        normalize=normalize,
        values_format=".2f" if normalize else "d",
        ax=ax,
        colorbar=False,
    )
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", out_path)


def save_loss_curve(losses: List[float], out_path: Path) -> None:
    """Save training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, "b-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Fusion Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved loss curve to %s", out_path)


# =============================================================================
# Main
# =============================================================================

def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger.setLevel(numeric_level)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train late fusion (V-JEPA2 + skeleton model) with learned α."
    )
    parser.add_argument(
        "--vjepa-root",
        type=Path,
        default=DEFAULT_VJEPA_ROOT,
        help="Root directory with V-JEPA2 fold_* subdirectories.",
    )
    parser.add_argument(
        "--skeleton-root",
        type=Path,
        default=None,
        help="Root directory with skeleton model (PoseC3D or STGCN++) fold* subdirectories. "
             "Takes precedence over --posec3d-root if both specified.",
    )
    parser.add_argument(
        "--posec3d-root",
        type=Path,
        default=DEFAULT_POSEC3D_ROOT,
        help="(Deprecated) Root directory with PoseC3D fold* subdirectories. Use --skeleton-root instead.",
    )
    parser.add_argument(
        "--skeleton-model-name",
        type=str,
        default="skeleton",
        help="Name of the skeleton model for logging (e.g., 'PoseC3D', 'STGCN++').",
    )
    parser.add_argument(
        "--stgcn-root",
        type=Path,
        default=DEFAULT_STGCN_ROOT,
        help="Root directory with STGCN++ fold* subdirectories (for 3-way fusion).",
    )
    parser.add_argument(
        "--fusion-type",
        type=str,
        choices=["scalar", "per_class", "mlp", "three_way"],
        default="scalar",
        help="Fusion architecture: 'scalar' (1 param), 'per_class' (num_classes params), "
             "'mlp' (~148 params), 'three_way' (V-JEPA2+PoseC3D+STGCN++ MLP fusion).",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=16,
        help="Hidden dimension for MLP fusion (only used if --fusion-type=mlp).",
    )
    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=0.1,
        help="Dropout rate for MLP fusion (only used if --fusion-type=mlp).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for fusion results.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of classes.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=3,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Training epochs per fold.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for α.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level.",
    )
    parser.add_argument(
        "--issue-clips-csv",
        type=Path,
        default=DEFAULT_ISSUE_CLIPS_CSV,
        help="CSV containing segment_id's to exclude from training/evaluation.",
    )
    return parser.parse_args()


def run_fold(
    fold_idx: int,
    vjepa_root: Path,
    skeleton_root: Path,
    output_dir: Path,
    num_classes: int,
    class_names: List[str],
    num_epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    exclude_clips: Optional[set] = None,
    skeleton_model_name: str = "skeleton",
    fusion_type: str = "scalar",
    mlp_hidden_dim: int = 16,
    mlp_dropout: float = 0.1,
    stgcn_root: Optional[Path] = None,
    posec3d_root: Optional[Path] = None,
) -> Dict:
    """Run fusion training and evaluation for a single fold."""
    logger.info("=" * 80)
    logger.info("Fold %d", fold_idx)
    logger.info("=" * 80)
    
    # Load predictions
    vjepa_fold_dir = vjepa_root / f"fold_{fold_idx}"
    vjepa_df = load_vjepa_predictions(vjepa_fold_dir)
    
    if fusion_type == "three_way":
        # 3-way fusion: V-JEPA2 + PoseC3D + STGCN++
        if posec3d_root is None or stgcn_root is None:
            raise ValueError("3-way fusion requires both --posec3d-root and --stgcn-root")
        
        # Load PoseC3D predictions
        posec3d_fold_dir = posec3d_root / f"fold{fold_idx}"
        if not posec3d_fold_dir.exists():
            posec3d_fold_dir = posec3d_root / f"fold_{fold_idx}"
        posec3d_df = load_skeleton_predictions(posec3d_fold_dir, model_name="PoseC3D")
        
        # Load STGCN++ predictions
        stgcn_fold_dir = stgcn_root / f"fold{fold_idx}"
        if not stgcn_fold_dir.exists():
            stgcn_fold_dir = stgcn_root / f"fold_{fold_idx}"
        stgcn_df = load_skeleton_predictions(stgcn_fold_dir, model_name="STGCN++")
        
        # Merge all three predictions
        merged_df = merge_three_predictions(
            vjepa_df, posec3d_df, stgcn_df, num_classes, exclude_clips=exclude_clips
        )
        
        # Create 3-way dataset
        dataset = ThreeWayFusionDataset(merged_df, num_classes)
    else:
        # 2-way fusion: V-JEPA2 + skeleton
        # Try both fold naming conventions: fold_0 (V-JEPA style) and fold0 (pyskl style)
        skeleton_fold_dir = skeleton_root / f"fold{fold_idx}"
        if not skeleton_fold_dir.exists():
            skeleton_fold_dir = skeleton_root / f"fold_{fold_idx}"
        
        pose_df = load_skeleton_predictions(skeleton_fold_dir, model_name=skeleton_model_name)
        
        # Merge predictions
        merged_df = merge_predictions(vjepa_df, pose_df, num_classes, exclude_clips=exclude_clips)
        
        # Create 2-way dataset
        dataset = FusionDataset(merged_df, num_classes)
    
    # ==========================================================================
    # Inner CV: Train and evaluate fusion on separate subsets of val predictions
    # This prevents overfitting by never evaluating on training data
    # ==========================================================================
    
    inner_n_splits = 5
    inner_kfold = StratifiedKFold(n_splits=inner_n_splits, shuffle=True, random_state=42 + fold_idx)
    
    inner_clip_results = []
    inner_video_results = []
    all_inner_eval_results = []  # Store for final predictions
    all_inner_val_dfs = []
    
    # Log model info once
    temp_model = create_fusion_model(
        fusion_type=fusion_type,
        num_classes=num_classes,
        hidden_dim=mlp_hidden_dim,
        dropout=mlp_dropout,
        init_alpha=0.0,
    )
    num_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    logger.info("Fusion type: %s (%d trainable parameters)", fusion_type, num_params)
    logger.info("Inner CV: %d-fold split on %d validation predictions", inner_n_splits, len(merged_df))
    del temp_model
    
    for inner_idx, (train_idx, val_idx) in enumerate(inner_kfold.split(merged_df, merged_df["label_id"])):
        train_df = merged_df.iloc[train_idx].reset_index(drop=True)
        val_df = merged_df.iloc[val_idx].reset_index(drop=True)
        
        logger.info(
            "  Inner fold %d/%d | Train: %d | Val: %d",
            inner_idx + 1, inner_n_splits, len(train_df), len(val_df)
        )
        
        # Create train/val datasets using the same class as main dataset
        DatasetClass = ThreeWayFusionDataset if fusion_type == "three_way" else FusionDataset
        train_dataset = DatasetClass(train_df, num_classes)
        val_dataset = DatasetClass(val_df, num_classes)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Compute class weights on training set only
        train_labels = train_df["label_id"].values
        class_weights = compute_class_weights(train_labels, num_classes)
        
        # Initialize fresh model for each inner fold
        model = create_fusion_model(
            fusion_type=fusion_type,
            num_classes=num_classes,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout,
            init_alpha=0.0,
        ).to(device)
        
        # Train (reduced logging for inner folds)
        if fusion_type == "three_way":
            losses = train_three_way_fusion(
                model=model,
                train_loader=train_loader,
                class_weights=class_weights,
                device=device,
                num_epochs=num_epochs,
                lr=lr,
                log_interval=num_epochs + 1,  # Suppress inner fold epoch logs
            )
        else:
            losses = train_fusion(
                model=model,
                train_loader=train_loader,
                class_weights=class_weights,
                device=device,
                num_epochs=num_epochs,
                lr=lr,
                log_interval=num_epochs + 1,  # Suppress inner fold epoch logs
            )
        
        # Evaluate on held-out inner val set
        if fusion_type == "three_way":
            eval_results = evaluate_three_way_fusion(model, val_loader, device)
        else:
            eval_results = evaluate_fusion(model, val_loader, device)
        
        # Compute clip-level metrics for this inner fold
        inner_clip_metrics = compute_metrics(
            eval_results["labels"],
            eval_results["preds"],
            eval_results["probs"],
            class_names,
        )
        inner_clip_results.append(inner_clip_metrics)
        
        # Compute video-level metrics for this inner fold
        inner_video_result = aggregate_video_predictions(
            val_df,
            eval_results["preds"],
            eval_results["labels"],
            eval_results["probs"],
            eval_results["segment_ids"],
        )
        if inner_video_result:
            inner_video_metrics = compute_metrics(
                inner_video_result["labels"],
                inner_video_result["preds"],
                inner_video_result["probs"],
                class_names,
            )
            inner_video_results.append(inner_video_metrics)
        
        # Store for final predictions aggregation
        all_inner_eval_results.append(eval_results)
        all_inner_val_dfs.append(val_df)
        
        logger.info(
            "    → Clip Top-1: %.3f | Video Top-1: %.3f",
            inner_clip_metrics["top1_acc"],
            inner_video_metrics["top1_acc"] if inner_video_result else 0.0
        )
    
    # Aggregate metrics across inner folds
    clip_metrics = average_metrics(inner_clip_results)
    clip_metrics["fusion_type"] = fusion_type
    clip_metrics["alpha"] = model.alpha_value  # From last model (just for reference)
    
    video_metrics = average_metrics(inner_video_results) if inner_video_results else None
    
    logger.info(
        "Fold %d Clip-Level (inner CV avg) | Top-1: %.3f | Top-2: %.3f | Macro F1: %.3f | (MLP fusion)",
        fold_idx, clip_metrics["top1_acc"], clip_metrics.get("top2_acc", 0),
        clip_metrics["macro_f1"]
    )
    if video_metrics:
        logger.info(
            "Fold %d Video-Level (inner CV avg) | Top-1: %.3f | Macro F1: %.3f | κ = %.3f",
            fold_idx, video_metrics["top1_acc"], video_metrics["macro_f1"],
            video_metrics["cohen_kappa"]
        )
    
    # Concatenate all inner val predictions for saving (covers full val set)
    all_segment_ids = []
    all_labels = []
    all_preds = []
    all_probs = []
    for er in all_inner_eval_results:
        all_segment_ids.extend(er["segment_ids"])
        all_labels.extend(er["labels"])
        all_preds.extend(er["preds"])
        all_probs.append(er["probs"])
    
    eval_results = {
        "segment_ids": all_segment_ids,
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.vstack(all_probs),
    }
    
    # For video results, concatenate all inner val dfs
    video_results = aggregate_video_predictions(
        pd.concat(all_inner_val_dfs, ignore_index=True),
        eval_results["preds"],
        eval_results["labels"],
        eval_results["probs"],
        eval_results["segment_ids"],
    )
    
    # Save outputs
    fold_out = output_dir / f"fold_{fold_idx}"
    fold_out.mkdir(parents=True, exist_ok=True)
    
    # Save alpha/model info
    alpha_path = fold_out / "alpha.json"
    alpha_info = {"alpha_mean": model.alpha_value, "fusion_type": fusion_type}
    if hasattr(model, "alpha_per_class"):
        alpha_info["alpha_per_class"] = model.alpha_per_class
    elif hasattr(model, "alpha") and model.alpha.numel() == 1:
        alpha_info["alpha_raw"] = model.alpha.item()
    if hasattr(model, "num_parameters"):
        alpha_info["num_parameters"] = model.num_parameters
    with open(alpha_path, "w") as f:
        json.dump(alpha_info, f, indent=2)
    
    # Save predictions
    pred_df = pd.DataFrame({
        "segment_id": eval_results["segment_ids"],
        "label_id": eval_results["labels"],
        "pred_id": eval_results["preds"],
        "label_name": [class_names[l] for l in eval_results["labels"]],
        "pred_name": [class_names[p] for p in eval_results["preds"]],
    })
    for i in range(num_classes):
        pred_df[f"fused_score_class{i}"] = eval_results["probs"][:, i]
    pred_df.to_csv(fold_out / "predictions_clip.csv", index=False)
    
    # Save video predictions
    if video_results:
        video_results["df"]["label_name"] = video_results["df"]["label"].map(lambda x: class_names[x])
        video_results["df"]["pred_name"] = video_results["df"]["pred"].map(lambda x: class_names[x])
        video_results["df"].to_csv(fold_out / "predictions_video.csv", index=False)
    
    # Save metrics
    metrics_out = {
        "clip": clip_metrics,
        "video": video_metrics,
    }
    with open(fold_out / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    
    # Save confusion matrices
    save_confusion_matrix(
        eval_results["labels"],
        eval_results["preds"],
        class_names,
        fold_out / "confusion_matrix_clip.png",
    )
    if video_results:
        save_confusion_matrix(
            video_results["labels"],
            video_results["preds"],
            class_names,
            fold_out / "confusion_matrix_video.png",
        )
    
    # Save loss curve
    save_loss_curve(losses, fold_out / "loss_curve.png")
    
    return {
        "fold": fold_idx,
        "clip": clip_metrics,
        "video": video_metrics,
        "n_samples": len(merged_df),
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    
    # Load issue clips to exclude
    issue_clips = load_issue_clips(args.issue_clips_csv)
    
    # Resolve skeleton root (--skeleton-root takes precedence over --posec3d-root)
    skeleton_root = args.skeleton_root if args.skeleton_root else args.posec3d_root
    skeleton_model_name = args.skeleton_model_name
    
    # Auto-detect model name from path if not specified
    if skeleton_model_name == "skeleton":
        if "stgcn" in str(skeleton_root).lower():
            skeleton_model_name = "STGCN++"
        elif "posec3d" in str(skeleton_root).lower():
            skeleton_model_name = "PoseC3D"
    
    logger.info("Fusion type: %s", args.fusion_type)
    
    if args.num_classes == 5:
        class_names = CLASS_NAMES_5CLASS
    else:
        class_names = CLASS_NAMES_4CLASS[:args.num_classes]
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all folds
    fold_results = []
    for fold_idx in range(args.num_folds):
        result = run_fold(
            fold_idx=fold_idx,
            vjepa_root=args.vjepa_root,
            skeleton_root=skeleton_root,
            output_dir=args.output_dir,
            num_classes=args.num_classes,
            class_names=class_names,
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            exclude_clips=issue_clips,
            skeleton_model_name=skeleton_model_name,
            fusion_type=args.fusion_type,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_dropout=args.mlp_dropout,
            stgcn_root=args.stgcn_root,
            posec3d_root=args.posec3d_root,
        )
        fold_results.append(result)
    
    # Aggregate cross-fold results
    logger.info("=" * 80)
    logger.info("Cross-Fold Summary")
    logger.info("=" * 80)
    
    summary = {
        "fusion_type": args.fusion_type,
        "folds": fold_results,
        "clip": {},
        "video": {},
    }
    
    # Compute means and stds for clip-level metrics
    clip_metrics_keys = ["top1_acc", "top2_acc", "macro_f1", "weighted_f1", "cohen_kappa", "alpha"]
    for key in clip_metrics_keys:
        values = [r["clip"][key] for r in fold_results if r["clip"].get(key) is not None]
        if values:
            # Skip alpha if it's not numeric (e.g., for 3-way fusion)
            if key == "alpha" and not all(isinstance(v, (int, float)) for v in values):
                continue
            try:
                summary["clip"][f"{key}_mean"] = float(np.mean(values))
                summary["clip"][f"{key}_std"] = float(np.std(values))
            except (TypeError, ValueError):
                # Skip non-numeric values
                pass
    
    # Compute means and stds for video-level metrics
    video_metrics_keys = ["top1_acc", "top2_acc", "macro_f1", "weighted_f1", "cohen_kappa"]
    for key in video_metrics_keys:
        values = [r["video"][key] for r in fold_results if r["video"] is not None and r["video"].get(key) is not None]
        if values:
            summary["video"][f"{key}_mean"] = float(np.mean(values))
            summary["video"][f"{key}_std"] = float(np.std(values))
    
    # Print summary
    logger.info("Clip-Level:")
    logger.info("  Top-1 Accuracy: %.3f ± %.3f", summary["clip"].get("top1_acc_mean", 0), summary["clip"].get("top1_acc_std", 0))
    logger.info("  Top-2 Accuracy: %.3f ± %.3f", summary["clip"].get("top2_acc_mean", 0), summary["clip"].get("top2_acc_std", 0))
    logger.info("  Macro F1: %.3f ± %.3f", summary["clip"].get("macro_f1_mean", 0), summary["clip"].get("macro_f1_std", 0))
    logger.info("  Cohen's κ: %.3f ± %.3f", summary["clip"].get("cohen_kappa_mean", 0), summary["clip"].get("cohen_kappa_std", 0))
    logger.info("  α (V-JEPA2 weight): %.3f ± %.3f", summary["clip"].get("alpha_mean", 0), summary["clip"].get("alpha_std", 0))
    
    if summary["video"]:
        logger.info("Video-Level:")
        logger.info("  Top-1 Accuracy: %.3f ± %.3f", summary["video"].get("top1_acc_mean", 0), summary["video"].get("top1_acc_std", 0))
        logger.info("  Macro F1: %.3f ± %.3f", summary["video"].get("macro_f1_mean", 0), summary["video"].get("macro_f1_std", 0))
        logger.info("  Cohen's κ: %.3f ± %.3f", summary["video"].get("cohen_kappa_mean", 0), summary["video"].get("cohen_kappa_std", 0))
    
    # Save summary
    summary_path = args.output_dir / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved CV summary to %s", summary_path)
    
    # Also save as CSV for easy viewing
    summary_rows = []
    for r in fold_results:
        row = {
            "fold": r["fold"],
            "n_samples": r["n_samples"],
            "clip_top1_acc": r["clip"]["top1_acc"],
            "clip_top2_acc": r["clip"]["top2_acc"],
            "clip_macro_f1": r["clip"]["macro_f1"],
            "clip_cohen_kappa": r["clip"]["cohen_kappa"],
            "alpha": r["clip"]["alpha"],
        }
        if r["video"]:
            row["video_top1_acc"] = r["video"]["top1_acc"]
            row["video_macro_f1"] = r["video"]["macro_f1"]
            row["video_cohen_kappa"] = r["video"]["cohen_kappa"]
        summary_rows.append(row)
    
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "cv_summary.csv", index=False)
    logger.info("Saved CV summary CSV to %s", args.output_dir / "cv_summary.csv")


if __name__ == "__main__":
    main()

