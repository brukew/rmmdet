# Overfitting Remedies for SAILS RMM Classification

This document summarizes the overfitting issues observed in PoseC3D and STGCN++ training on the SAILS dataset, and the various remedies we've implemented and tested.

---

## Problem Summary

The SAILS RMM dataset is **very small** (654 clips total, ~470 for training) which leads to severe overfitting:

| Metric | Observed |
|--------|----------|
| Training Accuracy | 100% (reached by epoch 4-8) |
| Validation Accuracy | 65-75% |
| **Train-Val Gap** | **25-35%** |

### Key Issues Identified

1. **Early training saturation** - Models reach 100% train accuracy very quickly
2. **Unstable validation** - Val accuracy swings 8-10% between epochs
3. **Class imbalance** - "hands flapping" dominates (55% of samples)
4. **Small dataset** - Only ~470 training samples across 4 classes

---

## Remedy 1: Class-Weighted Loss

### Implementation

Created `tools/train_weighted.py` that dynamically computes inverse-frequency class weights:

```python
# Inverse frequency weighting formula
weight[class_i] = total_samples / (num_classes * class_count[class_i])

# Example for 4-class SAILS:
# Class 0 (hands flapping): 264 samples → weight = 0.45
# Class 1 (jumping):        116 samples → weight = 1.02
# Class 2 (rocking):         57 samples → weight = 2.07
# Class 3 (spinning):        36 samples → weight = 3.28
```

### Weight Scaling Options

For highly imbalanced datasets, linear weights can be too aggressive. We implemented scaling options:

| Scale | Formula | Use Case |
|-------|---------|----------|
| `linear` | `w = total / (n_classes × count)` | Default |
| `sqrt` | `w = sqrt(linear_weight)` | 5-class (milder) |
| `log` | `w = log1p(linear_weight)` | Very imbalanced |
| `none` | `w = 1.0` | No weighting |

### Results: PoseC3D

| Task | Metric | Unweighted | Weighted | Change |
|------|--------|------------|----------|--------|
| **4-class CV** | Clip Top-1 | 74.6% | **76.4%** | +1.8% ✅ |
| **4-class CV** | Clip Macro F1 | 70.4% | **76.2%** | +5.8% ✅ |
| **4-class CV** | Video Top-1 | 74.6% | **76.6%** | +2.0% ✅ |
| **5-class CV** | Clip Top-1 | **66.8%** | 46.8% | -20% ❌ |
| **5-class CV** | Clip Macro F1 | **65.1%** | 45.9% | -19.2% ❌ |

**Conclusion for PoseC3D:**
- ✅ **4-class**: Weighting helps significantly
- ❌ **5-class**: Linear weighting is too aggressive (use `--weight-scale sqrt`)

### Results: STGCN++

| Task | Metric | Unweighted | Weighted | Change |
|------|--------|------------|----------|--------|
| **4-class CV** | Clip Top-1 | **77.9%** | 75.3% | -2.6% ❌ |
| **4-class CV** | Video Top-1 | **82.4%** | 79.5% | -2.9% ❌ |
| **5-class CV** | Video Macro F1 | 64.8% | **66.9%** | +2.1% ✅ |

**Conclusion for STGCN++:**
- ❌ **4-class**: Weighting hurts (skip it)
- ✅ **5-class**: Slight benefit with sqrt scaling

### Why the Difference?

| Factor | PoseC3D | STGCN++ |
|--------|---------|---------|
| Architecture | 3D CNN (high capacity, ~23M params) | GCN + TCN (lower capacity, ~1-2M params) |
| Regularization need | High (benefits from weighting) | Lower (graph structure provides regularization) |
| Training stability | More stable | Less stable (validation swings) |

---

## Remedy 2: Regularization Techniques

### 2.1 Frozen Backbone (PoseC3D)

Freeze all backbone layers and only train the classifier head:

```python
# configs/posec3d/slowonly_r50_sails_k400p/joint_regularized.py
model = dict(
    backbone=dict(
        frozen_stages=3,  # Freeze all 3 ResNet stages
    ),
)
```

**Impact:**
- Reduces trainable parameters from ~23M to ~2K
- Train-val gap reduced from 25-35% to **7%**
- But slightly lower peak accuracy (69% vs 74%)

### 2.2 Dropout (Both Models)

Add dropout to classifier head:

```python
# PoseC3D: Already has dropout=0.5 in base config
# STGCN++: Add to regularized config
model = dict(
    cls_head=dict(
        dropout=0.5,  # Was 0.0!
    )
)
```

**Impact:**
- Stabilizes validation accuracy
- Reduces overfitting in later epochs

### 2.3 Weight Decay

Increase L2 regularization:

```python
optimizer = dict(
    type='SGD',
    lr=0.00125,
    momentum=0.9,
    weight_decay=0.001,  # Increased from 0.0003-0.0005
)
```

### 2.4 Reduced Training Duration

| Model | Original | Regularized |
|-------|----------|-------------|
| PoseC3D | 12 epochs | 10 epochs |
| STGCN++ | 24 epochs | 16 epochs |

Best validation often occurs at epochs 4-8; training longer just increases overfitting.

### 2.5 Reduced Data Repetition

```python
data = dict(
    train=dict(
        times=5,  # Reduced from 10
    )
)
```

---

## Remedy 3: Regularized Configs

We created regularized config files that combine multiple techniques:

### PoseC3D Regularized

**File:** `configs/posec3d/slowonly_r50_sails_k400p/joint_regularized.py`

```python
_base_ = './joint.py'

model = dict(
    backbone=dict(frozen_stages=3),  # Freeze backbone
)

optimizer = dict(
    type='SGD',
    lr=0.00125,
    momentum=0.9,
    weight_decay=0.001,
)

total_epochs = 10
data = dict(train=dict(times=5))
```

### STGCN++ Regularized

**File:** `configs/stgcn++/stgcnpp_sails_ntu60p/j_regularized.py`

```python
_base_ = './j.py'

model = dict(
    cls_head=dict(dropout=0.5),
)

optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.001,
    nesterov=True,
)

total_epochs = 16
data = dict(train=dict(times=5))
```

---

## Experimental Results

### Regularized Training Results (4-class Single Split, Validation Set)

| Model | Config | Val Top-1 | Val Mean Class Acc | Train-Val Gap |
|-------|--------|-----------|-------------------|---------------|
| PoseC3D | Original | ~74% | ~70% | 25-35% |
| PoseC3D | **Regularized** | 69.2% | 57.6% | **7%** ✅ |
| STGCN++ | Original | ~75% | ~72% | 25-30% |
| STGCN++ | **Regularized** | **80.8%** | **82.6%** | 19% |

### Training Progression (Regularized)

**PoseC3D Regularized:**
```
Epoch 1:  Val 65.4%, Train 68% (gap: 3%)
Epoch 7:  Val 69.2%, Train 73% (gap: 4%)
Epoch 10: Val 69.2%, Train 76% (gap: 7%)
```

**STGCN++ Regularized:**
```
Epoch 1:  Val 71.2%, Train 82% (gap: 11%)
Epoch 4:  Val 76.9%, Train 96% (gap: 19%)
Epoch 11: Val 80.8%, Train 100% (gap: 19%)
Epoch 16: Val 80.8%, Train 100% (gap: 19%)
```

---

## Recommendations Summary

### For PoseC3D

| Scenario | Recommendation |
|----------|----------------|
| **4-class** | Use weighted training (`train_weighted.py`) |
| **5-class** | Use `--weight-scale sqrt` or skip weighting |
| **Severe overfitting** | Use `joint_regularized.py` (frozen backbone) |
| **Better accuracy** | Use original config with weighting |

### For STGCN++

| Scenario | Recommendation |
|----------|----------------|
| **4-class** | Skip weighting, use original config |
| **5-class** | Use `--weight-scale sqrt` |
| **Severe overfitting** | Use `j_regularized.py` (dropout + lower LR) |
| **Best accuracy** | Regularized config outperforms original |

---

## Usage Examples

### Weighted Training

```bash
# 4-class with class weighting (PoseC3D)
python tools/train_weighted.py configs/posec3d/slowonly_r50_sails_k400p/joint.py \
    --ann-file data/sails/single/4class_conf04.pkl \
    --work-dir work_dirs/posec3d/single/4class_weighted \
    --validate --launcher none

# 5-class with sqrt scaling (both models)
python tools/train_weighted.py configs/stgcn++/stgcnpp_sails_ntu60p/j.py \
    --ann-file data/sails/single/5class_conf04.pkl \
    --work-dir work_dirs/stgcnpp/single/5class_weighted \
    --weight-scale sqrt \
    --num-classes 5 \
    --validate --launcher none
```

### Regularized Training

```bash
# PoseC3D with frozen backbone
python tools/train.py configs/posec3d/slowonly_r50_sails_k400p/joint_regularized.py \
    --work-dir work_dirs/posec3d/single/4class_regularized \
    --validate --launcher none \
    --cfg-options \
        data.train.dataset.ann_file=data/sails/single/4class_conf04.pkl \
        data.val.ann_file=data/sails/single/4class_conf04.pkl

# STGCN++ with dropout
python tools/train.py configs/stgcn++/stgcnpp_sails_ntu60p/j_regularized.py \
    --work-dir work_dirs/stgcnpp/single/4class_regularized_j \
    --validate --launcher none \
    --cfg-options \
        data.train.dataset.ann_file=data/sails/single/4class_conf04.pkl \
        data.val.ann_file=data/sails/single/4class_conf04.pkl
```

### SLURM Batch Submission

```bash
# Test regularized configs
sbatch scripts/slurm/test_regularized.sh

# Weighted training (all models)
sbatch scripts/slurm/posec3d/train_4class_cv_weighted.sh
sbatch scripts/slurm/stgcnpp/train_5class_cv_4stream_weighted.sh
```

---

## Remedy 4: Focal Loss

### What is Focal Loss?

Focal Loss was introduced in ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002) (Lin et al., ICCV 2017) to address class imbalance in object detection. It down-weights easy examples and focuses training on hard, misclassified samples.

**Formula:**
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

Where:
- `p_t` = probability of the correct class
- `γ` (gamma) = focusing parameter (typically 2.0)
- `α` (alpha) = class balance weight (optional)

### How It Works

| γ value | Effect |
|---------|--------|
| γ = 0 | Equivalent to standard cross-entropy |
| γ = 1 | Moderate down-weighting of easy examples |
| γ = 2 | Strong focus on hard examples (recommended) |
| γ = 5 | Very aggressive (may ignore easy examples too much) |

**Example impact on loss:**
- Well-classified sample (p_t = 0.9): Loss reduced by 100× with γ=2
- Misclassified sample (p_t = 0.1): Loss barely affected

### Implementation

We added `FocalLoss` to pyskl:

**File:** `pyskl/models/losses/cross_entropy_loss.py`

```python
@LOSSES.register_module()
class FocalLoss(BaseWeightedLoss):
    def __init__(self, gamma=2.0, alpha=None, loss_weight=1.0, class_weight=None):
        # gamma: focusing parameter
        # alpha/class_weight: per-class weights (optional)
        ...
```

### Config File

**File:** `configs/posec3d/slowonly_r50_sails_k400p/joint_focal.py`

```python
model = dict(
    cls_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            gamma=2.0,           # Focusing parameter
            alpha=None,          # Per-class weights (optional)
            loss_weight=1.0)))
```

### Combining Focal Loss with Class Weights

Focal Loss can be combined with per-class weights for a two-pronged approach:

```python
# Option 1: Use alpha parameter directly
model = dict(
    cls_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            gamma=2.0,
            alpha=[0.45, 1.02, 2.07, 3.28],  # Inverse-frequency weights
        )))

# Option 2: Use class_weight (same effect, compatible with train_weighted.py)
model = dict(
    cls_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            gamma=2.0,
            class_weight=[0.45, 1.02, 2.07, 3.28],
        )))
```

### Usage

```bash
# Basic focal loss training
python tools/train.py configs/posec3d/slowonly_r50_sails_k400p/joint_focal.py \
    --work-dir work_dirs/posec3d/single/4class_focal \
    --validate --launcher none

# Focal loss with class weights
python tools/train.py configs/posec3d/slowonly_r50_sails_k400p/joint_focal.py \
    --work-dir work_dirs/posec3d/single/4class_focal_weighted \
    --validate --launcher none \
    --cfg-options "model.cls_head.loss_cls.alpha=[0.45,1.02,2.07,3.28]"

# Adjust gamma for more/less focus on hard examples
python tools/train.py configs/posec3d/slowonly_r50_sails_k400p/joint_focal.py \
    --work-dir work_dirs/posec3d/single/4class_focal_g1 \
    --validate --launcher none \
    --cfg-options model.cls_head.loss_cls.gamma=1.0
```

### When to Use Focal Loss

| Scenario | Recommendation |
|----------|----------------|
| Class imbalance | ✅ Good choice |
| Many easy examples dominate training | ✅ Excellent choice |
| Model quickly reaches 100% train acc | ✅ May help |
| Need precise control over hard examples | ✅ Tune gamma |
| Combined with class weights | ⚠️ Start with gamma=1.0 to avoid double-penalizing |

### Comparison: Focal Loss vs Class Weighting

| Method | Mechanism | Best For |
|--------|-----------|----------|
| **Class Weighting** | Up-weights minority classes | Correcting frequency imbalance |
| **Focal Loss** | Down-weights easy examples | Focusing on hard examples |
| **Both Combined** | Dual approach | Severe imbalance + hard examples |

---

## Future Work

### Not Yet Implemented

1. **Early Stopping** - Stop training when validation accuracy plateaus (patience ~5 epochs)

2. **Data Augmentation** - Skeleton-specific augmentations:
   - Random rotation (`RandomRot` exists in pyskl)
   - Random scaling
   - Gaussian noise on coordinates
   - Temporal augmentation (speed perturbation)

3. **Label Smoothing**:
   ```python
   model.cls_head.loss_cls = dict(
       type='CrossEntropyLoss',
       label_smooth_eps=0.1,
   )
   ```

4. **Mixup/CutMix** - Mix samples during training

5. **Stochastic Depth** - Randomly drop layers during training

6. **Progressive Unfreezing** - Start with frozen backbone, gradually unfreeze

---

## Key Takeaways

1. **Small dataset = severe overfitting** - 654 clips is not enough for deep models

2. **PoseC3D and STGCN++ respond differently** to regularization:
   - PoseC3D benefits from class weighting (4-class)
   - STGCN++ benefits from dropout and lower LR

3. **Frozen backbone** is the most effective anti-overfitting technique for PoseC3D

4. **5-class is harder** - More classes with fewer samples per class increases overfitting risk

5. **Monitor validation** - Best checkpoint is often NOT the final epoch

---

## Files Reference

| File | Purpose |
|------|---------|
| `tools/train_weighted.py` | Training with dynamic class weights |
| `pyskl/models/losses/cross_entropy_loss.py` | Contains FocalLoss implementation |
| `configs/posec3d/slowonly_r50_sails_k400p/joint_focal.py` | PoseC3D with Focal Loss |
| `configs/posec3d/slowonly_r50_sails_k400p/joint_regularized.py` | PoseC3D regularized config |
| `configs/stgcn++/stgcnpp_sails_ntu60p/j_regularized.py` | STGCN++ regularized config |
| `scripts/slurm/test_regularized.sh` | Test both regularized configs |
| `scripts/slurm/posec3d/*_weighted.sh` | PoseC3D weighted training scripts |
| `scripts/slurm/stgcnpp/*_weighted.sh` | STGCN++ weighted training scripts |

