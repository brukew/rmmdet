# Model Comparison: SAILS RMM Action Recognition

Comprehensive comparison of all methods evaluated for RMM classification.

---

## Summary Tables

### 4-Class Task (Clip-Level Metrics)

| Rank | Method | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Cohen's κ |
|------|--------|------------|------------|---------------|-----------|
| 🥇 | **3-Way MLP (V-JEPA2 + PoseC3D + STGCN++)** | **84.7% ± 1.9%** | **96.2% ± 1.9%** | **82.1% ± 3.9%** | **0.759 ± 0.02** |
| 🥈 | MLP Fusion (V-JEPA2 + STGCN++) | 83.8% ± 2.2% | 96.5% ± 0.4% | 82.2% ± 2.5% | 0.745 ± 0.03 |
| 🥉 | MLP Fusion (V-JEPA2 + PoseC3D) | 80.3% ± 3.3% | 95.1% ± 2.4% | 78.9% ± 4.4% | 0.690 ± 0.04 |
| 4 | STGCN++ (4-stream) | 77.9% | 95.1% | 72.7% | 0.635 |
| 5 | V-JEPA2 + SAM3 crop | 76.0% | 93.7% | 75.1% | 0.657 |
| 6 | PoseC3D (non-weighted) | 74.6% | 91.3% | 70.4% | 0.580 |
| 7 | Qwen2.5-VL (zero-shot) | — | — | 28.2% | 0.168 |

*Note: MLP fusion results use nested 5-fold CV within validation set for realistic evaluation.*

### 4-Class Task (Video-Level Metrics)

| Rank | Method | Video Top-1 | Video Macro F1 | Cohen's κ |
|------|--------|-------------|----------------|-----------|
| 🥇 | **3-Way MLP (V-JEPA2 + PoseC3D + STGCN++)** | **86.4% ± 2.6%** | **83.1% ± 4.9%** | **0.786 ± 0.04** |
| 🥈 | MLP Fusion (V-JEPA2 + STGCN++) | 84.2% ± 3.7% | 81.3% ± 4.1% | 0.752 ± 0.06 |
| 🥉 | STGCN++ (4-stream) | 82.4% | 75.9% | 0.635 |
| 4 | MLP Fusion (V-JEPA2 + PoseC3D) | 82.5% ± 3.1% | 80.5% ± 3.6% | 0.728 ± 0.04 |
| 5 | V-JEPA2 + SAM3 crop | 78.6% | 76.7% | 0.657 |
| 6 | PoseC3D (non-weighted) | 74.6% | 70.4% | 0.580 |

*Note: MLP fusion results use nested 5-fold CV within validation set for realistic evaluation.*

### 5-Class Task (Clip-Level Metrics)

| Rank | Method | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Cohen's κ |
|------|--------|------------|------------|---------------|-----------|
| 🥇 | **3-Way MLP (V-JEPA2 + PoseC3D + STGCN++)** | **69.6% ± 3.0%** | **88.5% ± 0.4%** | **69.8% ± 1.8%** | **0.594 ± 0.04** |
| 🥈 | MLP Fusion (V-JEPA2 + STGCN++) | 68.2% ± 1.5% | 88.1% ± 2.1% | 68.5% ± 1.9% | 0.578 ± 0.02 |
| 🥉 | MLP Fusion (V-JEPA2 + PoseC3D) | 67.6% ± 3.6% | 87.5% ± 1.1% | 67.8% ± 1.0% | 0.569 ± 0.04 |
| 4 | PoseC3D + Weighted | 67.0% ± 1.6% | 85.5% ± 1.3% | 65.9% ± 2.8% | 0.549 |
| 5 | STGCN++ 4-stream | 66.5% ± 1.5% | 83.3% ± 1.0% | 64.6% ± 1.4% | 0.541 |
| 6 | STGCN++ 4-stream + Weighted | 65.8% ± 3.3% | 81.6% ± 1.5% | 63.7% ± 2.2% | 0.534 |
| 7 | PoseC3D + Weighted Sqrt | 65.3% ± 2.2% | 83.5% ± 1.1% | 63.9% ± 3.2% | 0.526 |
| 8 | PoseC3D + Focal Loss | 63.9% ± 0.7% | 83.0% ± 1.4% | 61.8% ± 3.5% | 0.502 |
| 9 | STGCN++ 4-stream + Focal | 63.7% ± 2.8% | 83.5% ± 0.4% | 63.6% ± 2.9% | 0.501 |
| 10 | V-JEPA2 + SAM3 crop | 62.5% | 84.8% | 63.0% | 0.499 |
| 11 | Qwen2.5-VL (zero-shot) | — | — | 12.2% | 0.008 |

*Note: MLP fusion results use nested 5-fold CV within validation set for realistic evaluation.*

### 5-Class Task (Video-Level Metrics)

| Rank | Method | Video Top-1 | Video Macro F1 | Cohen's κ |
|------|--------|-------------|----------------|-----------|
| 🥇 | **3-Way MLP (V-JEPA2 + PoseC3D + STGCN++)** | **70.7% ± 3.1%** | **70.8% ± 2.5%** | **0.611 ± 0.04** |
| 🥈 | STGCN++ 4-stream + Weighted | 70.6% ± 2.0% | 68.3% ± 1.2% | 0.534 |
| 🥉 | MLP Fusion (V-JEPA2 + STGCN++) | 69.0% ± 2.0% | 69.6% ± 2.8% | 0.589 ± 0.03 |
| 4 | STGCN++ 4-stream | 68.8% ± 3.0% | 66.9% ± 4.4% | 0.541 |
| 5 | STGCN++ 4-stream + Focal | 67.7% ± 1.0% | 67.1% ± 1.5% | 0.501 |
| 6 | PoseC3D + Weighted | 67.0% ± 1.6% | 65.9% ± 2.8% | 0.549 |
| 7 | MLP Fusion (V-JEPA2 + PoseC3D) | 66.9% ± 4.1% | 66.9% ± 3.1% | 0.560 ± 0.05 |
| 8 | PoseC3D + Weighted Sqrt | 65.3% ± 2.2% | 63.9% ± 3.2% | 0.526 |
| 9 | PoseC3D + Focal Loss | 63.9% ± 0.7% | 61.8% ± 3.5% | 0.502 |
| 10 | V-JEPA2 + SAM3 crop | 62.4% | 62.8% | 0.499 |

*Note: MLP fusion results use nested 5-fold CV within validation set for realistic evaluation.*

---

## Method Details

### Method Overview

| Method | Type | Input | Training | Parameters | Speed |
|--------|------|-------|----------|------------|-------|
| **PoseC3D** | Skeleton-CNN | 17 COCO keypoints → heatmaps | Finetuning | ~24M | Fast |
| **STGCN++** | Skeleton-GCN | 17 COCO keypoints → graph | Finetuning | ~1.4M | **Very Fast** |
| **V-JEPA2** | Video encoder | RGB frames (+ optional crop) | Finetuning | ~300M | Slow |
| **3-Way MLP Fusion** 🏆 | Ensemble | V-JEPA2 + PoseC3D + STGCN++ | MLP | **412** | **Instant** |
| **Late Fusion (MLP)** | Ensemble | V-JEPA2 + Skeleton logits | MLP | **212** | **Instant** |
| **Late Fusion (Scalar)** | Ensemble | V-JEPA2 + Skeleton logits | α only | **1** | **Instant** |
| **Late Fusion (Per-class)** | Ensemble | V-JEPA2 + Skeleton logits | α per class | **4** | **Instant** |
| **Qwen2.5-VL** | Zero-shot VLM | RGB frames | None | ~7B | Very Slow |

---

## Late Fusion (V-JEPA2 + PoseC3D)

Combines RGB-based V-JEPA2 predictions with skeleton-based PoseC3D predictions. Three fusion architectures were evaluated:

### Fusion Architectures

| Architecture | Formula | Parameters | Description |
|--------------|---------|------------|-------------|
| **Scalar α** | `z = σ(α)·z_rgb + (1-σ(α))·z_pose` | 1 | Single learned weight |
| **Per-class α** | `z[i] = σ(α[i])·z_rgb[i] + (1-σ(α[i]))·z_pose[i]` | 4 | Per-class weights |
| **MLP** | `z = MLP(concat(z_rgb, z_pose))` | 212 | Non-linear fusion (8→16→4) |

### 4-Class CV Results Comparison

| Fusion Type | Params | Clip Top-1 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Video κ |
|-------------|--------|------------|---------------|-------------|----------------|---------|
| **MLP** | 212 | **84.2% ± 4.8%** | **84.9% ± 4.0%** | **88.4% ± 2.5%** | **87.0% ± 2.0%** | **0.817 ± 0.04** |
| Scalar α | 1 | 82.6% ± 3.0% | 81.7% ± 3.2% | 83.7% ± 4.4% | 80.5% ± 5.2% | 0.729 ± 0.07 |
| Per-class α | 4 | 82.6% ± 3.6% | 81.2% ± 2.7% | 83.1% ± 3.4% | 79.4% ± 5.2% | 0.718 ± 0.07 |

### MLP Fusion (Best) - Per-Fold Results

| Fold | Clips | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Video κ |
|------|-------|------------|------------|---------------|-------------|----------------|---------|
| 0 | 204 | 90.7% | 98.5% | 89.0% | 91.8% | 88.9% | 0.867 |
| 1 | 217 | 82.9% | 97.2% | 82.1% | 87.1% | 85.3% | 0.797 |
| 2 | 205 | 79.0% | 97.1% | 81.9% | 86.1% | 86.0% | 0.788 |
| **Mean** | — | **84.2% ± 4.8%** | **97.9% ± 1.1%** | **84.9% ± 4.0%** | **88.4% ± 2.5%** | **87.0% ± 2.0%** | **0.817 ± 0.04** |

### Per-Class α - Learned Weights

| Fold | α[0] (hands flapping) | α[1] (jumping) | α[2] (rocking) | α[3] (spinning) |
|------|----------------------|----------------|----------------|-----------------|
| 0 | 0.59 | 0.54 | 0.57 | 0.42 |
| 1 | 0.82 | 0.09 | 0.95 | 0.92 |
| 2 | 0.60 | 0.56 | 0.69 | 0.38 |

**Key insight**: For "jumping" class (α ≈ 0.1-0.56), skeleton features are more discriminative. For "rocking" (α ≈ 0.57-0.95), V-JEPA2 RGB features dominate.

### Scalar α - Learned Weights

| Fold | α (V-JEPA2 weight) |
|------|-------------------|
| 0 | 0.59 |
| 1 | 0.85 |
| 2 | 0.55 |
| **Mean** | **0.66 ± 0.13** |

**Output Directories**:
- Scalar: `fusion/runs/4class_cv/`
- Per-class: `fusion/runs/4class_cv_perclass/`
- MLP: `fusion/runs/4class_cv_mlp/`

---

## Late Fusion (V-JEPA2 + STGCN++)

Combines RGB-based V-JEPA2 predictions with STGCN++ 4-stream fusion (best skeleton model). Both Scalar and MLP fusion were evaluated.

### 4-Class CV Results Comparison

| Fusion Type | Params | Clip Top-1 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Video κ |
|-------------|--------|------------|---------------|-------------|----------------|---------|
| **MLP** | 212 | **86.6% ± 1.5%** | **86.3% ± 1.7%** | **88.0% ± 1.5%** | **85.8% ± 1.4%** | **0.811 ± 0.02** |
| Scalar α | 1 | 76.9% ± 3.9% | 75.8% ± 1.9% | 79.0% ± 5.2% | 76.5% ± 6.9% | 0.654 ± 0.10 |

### MLP Fusion - Per-Fold Results

| Fold | Clips | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Video κ |
|------|-------|------------|------------|---------------|-------------|----------------|---------|
| 0 | 204 | 87.3% | 99.5% | 85.1% | 89.8% | 87.6% | 0.832 |
| 1 | 217 | 85.3% | 98.2% | 85.1% | 86.0% | 84.3% | 0.789 |
| 2 | 205 | 85.9% | 98.5% | 85.7% | 88.1% | 86.7% | 0.819 |
| **Mean** | — | **86.6% ± 1.5%** | **98.9% ± 0.8%** | **86.3% ± 1.7%** | **88.0% ± 1.5%** | **85.8% ± 1.4%** | **0.811 ± 0.02** |

### Scalar α Fusion - Per-Fold Results

| Fold | Clips | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Video κ | α |
|------|-------|------------|------------|---------------|-------------|----------------|---------|---|
| 0 | 204 | 82.4% | 95.6% | 77.9% | 84.7% | 82.1% | 0.745 | 0.97 |
| 1 | 217 | 74.2% | 92.6% | 73.3% | 72.0% | 66.8% | 0.517 | 0.97 |
| 2 | 205 | 74.1% | 95.1% | 76.3% | 80.2% | 80.6% | 0.701 | 0.97 |
| **Mean** | — | **76.9% ± 3.9%** | **94.4% ± 1.3%** | **75.8% ± 1.9%** | **79.0% ± 5.2%** | **76.5% ± 6.9%** | **0.65 ± 0.10** | **0.97** |

**Key Observations**:
- **MLP dramatically improves STGCN++ fusion**: +9.7% Clip Top-1, +9.0% Video Top-1
- Scalar α ≈ 0.97 → V-JEPA2 dominates linear fusion
- But **MLP can find non-linear interactions** that scalar misses
- **MLP + STGCN++ achieves best clip-level accuracy** (86.6%)

**Output Directories**:
- MLP: `fusion/runs/4class_cv_stgcn_mlp/`
- Scalar: `fusion/runs/4class_cv_stgcn/`

---

## 3-Way Late Fusion (V-JEPA2 + PoseC3D + STGCN++)

Combines all three models for maximum complementary information.

### Architecture

```
Input: [z_vjepa; z_posec3d; z_stgcn] → (12 features for 4-class)
Hidden: Linear(12 → 24) + ReLU + Dropout(0.1)
Output: Linear(24 → 4)

Parameters: 412
```

### Evaluation Method

Uses **nested 5-fold CV** within each validation fold to prevent overfitting:
- Split validation predictions into 80% train / 20% val for fusion
- Train fusion on inner train, evaluate on inner val
- Repeat 5x and average metrics

### 4-Class CV Per-Fold Results (with nested CV)

| Fold | Clips | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Video κ |
|------|-------|------------|------------|---------------|-------------|----------------|---------|
| 0 | 204 | 86.3% | 97.8% | 85.2% | 89.0% | 85.0% | 0.815 |
| 1 | 217 | 82.9% | 93.6% | 76.5% | 82.7% | 76.4% | 0.730 |
| 2 | 205 | 83.9% | 97.1% | 85.0% | 87.5% | 87.5% | 0.812 |
| **Mean** | — | **84.7% ± 1.9%** | **96.2% ± 1.9%** | **82.1% ± 3.9%** | **86.4% ± 2.6%** | **83.1% ± 4.9%** | **0.786 ± 0.04** |

### Key Findings

- **Best overall performance**: +2-4% over standalone skeleton models
- Combining 3 modalities captures complementary information:
  - V-JEPA2: RGB appearance + temporal patterns
  - PoseC3D: 3D convolutional skeleton heatmaps
  - STGCN++: Graph-based skeletal dynamics
- 412 parameters still very lightweight
- Nested CV provides realistic, non-overfitted estimates

**Output Directory**: `fusion/runs/4class_cv_3way/`

---

## 5-Class Late Fusion (V-JEPA2 + Skeleton)

MLP fusion was evaluated for 5-class task, combining V-JEPA2 with the best skeleton models.

### Model Configurations

| Component | Model | Config | Standalone Performance |
|-----------|-------|--------|------------------------|
| **V-JEPA2** | 5-class SAM3 crop | `f64_lr1e-5_bs1_acc8_ep20_crop` | 62.5% Clip, 62.4% Video |
| **PoseC3D** | 5-class weighted | `5class_conf04_weighted` | 67.0% Clip, 67.0% Video |
| **STGCN++** | 5-class 4-stream | `5class_conf04_4stream` | 66.5% Clip, 68.8% Video |

### 5-Class CV Results Comparison (with nested CV)

| Fusion | Clip Top-1 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Video κ |
|--------|------------|---------------|-------------|----------------|---------|
| **3-Way MLP** | **69.6% ± 3.0%** | **69.8% ± 1.8%** | **70.7% ± 3.1%** | **70.8% ± 2.5%** | **0.611** |
| MLP (V-JEPA2 + STGCN++) | 68.2% ± 1.5% | 68.5% ± 1.9% | 69.0% ± 2.0% | 69.6% ± 2.8% | 0.589 |
| MLP (V-JEPA2 + PoseC3D) | 67.6% ± 3.6% | 67.8% ± 1.0% | 66.9% ± 4.1% | 66.9% ± 3.1% | 0.560 |

### Key Findings for 5-Class Fusion

- **3-Way MLP achieves best 5-class performance**: 70.7% Video Top-1
- Modest improvement (+2-3%) over standalone skeleton models
- STGCN++ fusion better than PoseC3D fusion for 5-class
- Nested CV provides realistic estimates (previously reported ~75% was overfitting)

**Output Directories**:
- 3-Way MLP: `fusion/runs/5class_cv_3way/`
- MLP (PoseC3D): `fusion/runs/5class_cv_posec3d_mlp/`
- MLP (STGCN++): `fusion/runs/5class_cv_stgcn_mlp/`

---

## V-JEPA2

### 4-Class CV with SAM3 Cropping

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-------------|----------------|-----------|
| 0 | 80.6% | 93.5% | 76.7% | 83.0% | 81.7% | 0.724 |
| 1 | 73.4% | 92.2% | 72.5% | 71.3% | 65.3% | 0.523 |
| 2 | 73.9% | 95.4% | 76.2% | 81.6% | 83.2% | 0.723 |
| **Mean** | **76.0%** | **93.7%** | **75.1%** | **78.6%** | **76.7%** | **0.657** |

**Output**: `/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls/`

### 5-Class CV with SAM3 Cropping

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-------------|----------------|-----------|
| 0 | 65.4% | 85.7% | 64.5% | 64.0% | 63.3% | 0.503 |
| 1 | 60.6% | 85.8% | 61.8% | 66.0% | 66.0% | 0.553 |
| 2 | 61.5% | 83.0% | 62.8% | 57.3% | 59.1% | 0.442 |
| **Mean** | **62.5%** | **84.8%** | **63.0%** | **62.4%** | **62.8%** | **0.499** |

---

## PoseC3D

### 4-Class CV (Non-Weighted)

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-------------|----------------|-----------|
| 0 | 76.1% | 91.5% | 74.5% | 76.1% | 74.5% | 0.595 |
| 1 | 71.0% | 90.8% | 62.5% | 71.0% | 62.5% | 0.528 |
| 2 | 76.7% | 91.5% | 74.2% | 76.7% | 74.2% | 0.618 |
| **Mean** | **74.6%** | **91.3%** | **70.4%** | **74.6%** | **70.4%** | **0.580** |

**Output**: `/orcd/data/satra/001/users/brukew/actreg/pyskl/work_dirs/posec3d/cv/4class_conf04/`

### 5-Class CV + Weighted 🏆

**Best PoseC3D configuration for 5-class task.**

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-------------|----------------|-----------|
| 0 | 69.1% | 87.0% | 69.2% | 69.1% | 69.2% | 0.569 |
| 1 | 66.8% | 83.9% | 62.4% | 66.8% | 62.4% | 0.547 |
| 2 | 65.2% | 85.7% | 66.0% | 65.2% | 66.0% | 0.532 |
| **Mean** | **67.0% ± 1.6%** | **85.5% ± 1.3%** | **65.9% ± 2.8%** | **67.0% ± 1.6%** | **65.9% ± 2.8%** | **0.549** |

**Output**: `pyskl/work_dirs/posec3d/cv/5class_conf04_weighted/`

### 5-Class CV + Weighted Sqrt

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-------------|----------------|-----------|
| 0 | 66.7% | 85.0% | 64.5% | 66.7% | 64.5% | 0.538 |
| 1 | 62.2% | 83.0% | 59.9% | 62.2% | 59.9% | 0.491 |
| 2 | 67.1% | 82.4% | 67.5% | 67.1% | 67.5% | 0.549 |
| **Mean** | **65.3% ± 2.2%** | **83.5% ± 1.1%** | **63.9% ± 3.2%** | **65.3% ± 2.2%** | **63.9% ± 3.2%** | **0.526** |

**Output**: `pyskl/work_dirs/posec3d/cv/5class_conf04_weighted_sqrt/`

### 5-Class CV + Focal Loss

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-------------|----------------|-----------|
| 0 | 63.8% | 83.6% | 62.9% | 63.8% | 62.9% | 0.493 |
| 1 | 63.1% | 81.1% | 57.2% | 63.1% | 57.2% | 0.482 |
| 2 | 64.8% | 84.3% | 65.4% | 64.8% | 65.4% | 0.531 |
| **Mean** | **63.9% ± 0.7%** | **83.0% ± 1.4%** | **61.8% ± 3.5%** | **63.9% ± 0.7%** | **61.8% ± 3.5%** | **0.502** |

**Output**: `pyskl/work_dirs/posec3d/cv/5class_conf04_focal/`

---

## STGCN++ (4-Stream Fusion)

### 4-Class CV

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-------------|----------------|-----------|
| 0 | 81.0% | 96.1% | 78.7% | 80.8% | 76.3% | 0.681 |
| 1 | 76.5% | 94.5% | 67.7% | 81.1% | 71.3% | 0.607 |
| 2 | 76.2% | 94.7% | 71.8% | 85.2% | 80.2% | 0.616 |
| **Mean** | **77.9%** | **95.1%** | **72.7%** | **82.4%** | **75.9%** | **0.635** |

**Output**: `/orcd/data/satra/001/users/brukew/actreg/pyskl/work_dirs/stgcnpp/cv/4class_conf04/`

### 5-Class CV (Non-Weighted)

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-------------|----------------|-----------|
| 0 | 68.6% | 84.1% | 66.7% | 73.5% | 71.7% | 0.571 |
| 1 | 65.7% | 83.9% | 63.6% | 66.3% | 61.5% | 0.528 |
| 2 | 65.2% | 82.0% | 63.4% | 66.7% | 67.3% | 0.524 |
| **Mean** | **66.5% ± 1.5%** | **83.3% ± 1.0%** | **64.6% ± 1.4%** | **68.8% ± 3.0%** | **66.9% ± 4.4%** | **0.541** |

**Output**: `pyskl/work_dirs/stgcnpp/cv/5class_conf04_4stream/`

### 5-Class CV + Weighted 🏆

**Best STGCN++ configuration for 5-class task (video-level).**

| Fold | Clips | Videos | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Cohen's κ |
|------|-------|--------|------------|------------|---------------|-------------|----------------|-----------|
| 0 | 207 | 131 | 70.0% | 83.6% | 66.5% | 73.3% | 67.0% | 0.585 |
| 1 | 217 | 122 | 65.4% | 80.2% | 63.4% | 69.7% | 69.8% | 0.522 |
| 2 | 210 | 131 | 61.9% | 81.0% | 61.2% | 68.7% | 68.1% | 0.496 |
| **Mean** | — | — | **65.8% ± 3.3%** | **81.6% ± 1.5%** | **63.7% ± 2.2%** | **70.6% ± 2.0%** | **68.3% ± 1.2%** | **0.534** |

**Output**: `pyskl/work_dirs/stgcnpp/cv/5class_conf04_weighted_4stream/`

### 5-Class CV + Focal Loss

| Fold | Clips | Videos | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Top-1 | Video Macro F1 | Cohen's κ |
|------|-------|--------|------------|------------|---------------|-------------|----------------|-----------|
| 0 | 207 | 131 | 66.7% | 83.6% | 67.3% | 67.9% | 66.6% | 0.541 |
| 1 | 217 | 122 | 64.5% | 83.9% | 63.2% | 68.9% | 69.1% | 0.504 |
| 2 | 210 | 131 | 60.0% | 82.9% | 60.2% | 66.4% | 65.6% | 0.458 |
| **Mean** | — | — | **63.7% ± 2.8%** | **83.5% ± 0.4%** | **63.6% ± 2.9%** | **67.7% ± 1.0%** | **67.1% ± 1.5%** | **0.501** |

**Output**: `pyskl/work_dirs/stgcnpp/cv/5class_conf04_focal_4stream/`

---

## Qwen2.5-VL (Zero-Shot)

Zero-shot prompting with no training. Uses multi-window voting (4 windows × 16 frames).

### 4-Class Results

| Metric | Value |
|--------|-------|
| Macro F1 | 28.2% |
| Cohen's κ | 0.168 |

### 5-Class Results

| Metric | Value |
|--------|-------|
| Macro F1 | 12.2% |
| Cohen's κ | 0.008 |

**Note**: Zero-shot VLM struggles with fine-grained motor movement classification.

---

## Key Observations

### 1. 3-Way Fusion Achieves Best Performance
- **V-JEPA2 + PoseC3D + STGCN++ (3-Way MLP)**: 86.4% Video Top-1, 83.1% F1
- **+2-4% improvement** over standalone skeleton models
- Combining all 3 modalities captures complementary information
- Still only 412 parameters (lightweight)
- Evaluated with nested 5-fold CV for realistic estimates

### 2. MLP Unlocks STGCN++ Potential
- Scalar fusion with STGCN++ gave α ≈ 0.97 (V-JEPA2 dominated, STGCN++ ignored)
- **MLP finds useful non-linear interactions**: +9.7% Clip Top-1, +9.0% Video Top-1
- Shows linear fusion can miss complementary information

### 3. Both Skeleton Models Are Valuable
- PoseC3D MLP: Better video-level (88.4% vs 88.0%)
- STGCN++ MLP: Better clip-level (86.6% vs 84.2%) with lower variance (±1.5% vs ±4.8%)
- Choice depends on whether you optimize for clip or video predictions

### 4. Per-Class α Does Not Improve Over Scalar
- Per-class weighting (4 parameters) performs similarly to scalar (1 parameter)
- Reveals patterns: skeleton helps "jumping", RGB helps "rocking"
- But class-specific weighting doesn't improve overall accuracy

### 5. 5-Class Fusion Provides Modest Improvement
- **3-Way MLP achieves 70.7% Video Top-1** for 5-class task
- Modest improvement (+2-3%) over standalone skeleton models
- STGCN++ fusion better than PoseC3D fusion
- Class weighting still helps standalone skeleton models
- 5-class task remains challenging due to subtle class distinctions

### 6. Zero-Shot VLM Fails
- Qwen2.5-VL cannot distinguish fine-grained motor movements
- Near-chance performance despite sophisticated prompting
- Task requires temporal motion patterns, not scene understanding

---

## Recommendations

### 4-Class Task

| Use Case | Recommended Method |
|----------|-------------------|
| **Best overall accuracy** | 3-Way MLP (V-JEPA2 + PoseC3D + STGCN++) — 86.4% Video Top-1 |
| **Best clip-level accuracy** | 3-Way MLP — 84.7% Clip Top-1 |
| **2-way fusion (simpler)** | MLP Fusion (V-JEPA2 + STGCN++) — 84.2% Video Top-1 |
| **Fast training/inference** | STGCN++ (4-stream) — 82.4% Video Top-1 |
| **No training data** | N/A (zero-shot fails) |

### 5-Class Task

| Use Case | Recommended Method |
|----------|-------------------|
| **Best overall accuracy** | 3-Way MLP (V-JEPA2 + PoseC3D + STGCN++) — 70.7% Video Top-1 |
| **Best clip-level accuracy** | 3-Way MLP — 69.6% Clip Top-1 |
| **2-way fusion (simpler)** | MLP Fusion (V-JEPA2 + STGCN++) — 69.0% Video Top-1 |
| **Skeleton-only** | STGCN++ 4-stream + Weighted — 70.6% Video Top-1 |
| **Fast training/inference** | STGCN++ 4-stream — 68.8% Video Top-1 |

**Note**: 5-class fusion provides modest improvement (+2-3%) over standalone skeleton models. Standalone STGCN++ with class weighting is competitive.

---

## Output Locations

| Method | Output Directory | Logs |
|--------|------------------|------|
| **3-Way MLP (V-JEPA2 + PoseC3D + STGCN++)** 🏆 | `fusion/runs/4class_cv_3way/` | `~/fusion_logs/fusion_3way_cv_*.out` |
| MLP Fusion 4-class (V-JEPA2 + PoseC3D) | `fusion/runs/4class_cv_mlp/` | `~/fusion_logs/fusion_mlp_cv_*.out` |
| MLP Fusion 4-class (V-JEPA2 + STGCN++) | `fusion/runs/4class_cv_stgcn_mlp/` | `~/fusion_logs/fusion_stgcn_mlp_cv_*.out` |
| **3-Way MLP 5-class** | `fusion/runs/5class_cv_3way/` | `~/fusion_logs/fusion_5class_3way_cv_*.out` |
| MLP Fusion 5-class (V-JEPA2 + PoseC3D) | `fusion/runs/5class_cv_posec3d_mlp/` | `~/fusion_logs/fusion_5class_posec3d_cv_*.out` |
| MLP Fusion 5-class (V-JEPA2 + STGCN++) | `fusion/runs/5class_cv_stgcn_mlp/` | `~/fusion_logs/fusion_5class_stgcn_cv_*.out` |
| Late Fusion Scalar (V-JEPA2 + PoseC3D) | `fusion/runs/4class_cv/` | `~/fusion_logs/fusion_cv_*.out` |
| Late Fusion Per-class (V-JEPA2 + PoseC3D) | `fusion/runs/4class_cv_perclass/` | `~/fusion_logs/fusion_perclass_cv_*.out` |
| Late Fusion Scalar (V-JEPA2 + STGCN++) | `fusion/runs/4class_cv_stgcn/` | `~/fusion_logs/fusion_stgcn_cv_*.out` |
| V-JEPA2 (4-class crop) | `v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls/` | `~/vjepa_rmm_logs/` |
| V-JEPA2 (5-class crop) | `v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop/` | `~/vjepa_rmm_logs/` |
| PoseC3D (4-class) | `pyskl/work_dirs/posec3d/cv/4class_conf04/` | — |
| STGCN++ (4-class) | `pyskl/work_dirs/stgcnpp/cv/4class_conf04_4stream/` | — |
| PoseC3D + Weighted (5-class) | `pyskl/work_dirs/posec3d/cv/5class_conf04_weighted/` | `~/pyskl_logs/` |
| PoseC3D + Weighted Sqrt (5-class) | `pyskl/work_dirs/posec3d/cv/5class_conf04_weighted_sqrt/` | `~/pyskl_logs/` |
| PoseC3D + Focal (5-class) | `pyskl/work_dirs/posec3d/cv/5class_conf04_focal/` | `~/pyskl_logs/` |
| STGCN++ 4-stream + Weighted (5-class) | `pyskl/work_dirs/stgcnpp/cv/5class_conf04_weighted_4stream/` | `~/pyskl_logs/` |
| STGCN++ 4-stream + Focal (5-class) | `pyskl/work_dirs/stgcnpp/cv/5class_conf04_focal_4stream/` | `~/pyskl_logs/` |
| STGCN++ 4-stream (5-class) | `pyskl/work_dirs/stgcnpp/cv/5class_conf04_4stream/` | `~/pyskl_logs/` |

*All paths relative to `/orcd/data/satra/001/users/brukew/actreg/` unless otherwise noted.*
*`~` refers to `/orcd/data/satra/001/users/brukew/`*

