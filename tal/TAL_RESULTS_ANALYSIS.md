# TAL Training Results Analysis
**Date**: January 4, 2026  
**Task**: 4-class RMM Temporal Action Localization (+ background class)  
**Folds**: 0-1 (fold 2 pending)  
**Background Subsampling**: 20% (class_prob = 0.2)

---

## 📊 Overall Performance Summary

### PoseC3D (Joint Modality)

| Metric | Fold 0 | Fold 1 | **Average** |
|--------|--------|--------|-------------|
| **Top-1 Accuracy** | 85.33% | 86.45% | **85.89%** |
| **Top-2 Accuracy** | 96.63% | 95.68% | **96.16%** |
| **Macro F1** | 40.85% | 34.39% | **37.62%** |
| **Weighted F1** | 85.32% | 85.48% | **85.40%** |
| **Cohen's Kappa** | 0.3477 | 0.3121 | **0.3299** |

### STGCN++ (All 4 Modalities)

| Modality | Fold 0 Top-1 | Fold 1 Top-1 | **Average Top-1** | Fold 0 MCA | Fold 1 MCA | **Average MCA** |
|----------|--------------|--------------|-------------------|------------|------------|-----------------|
| **Joint (j)** | 80.12% | 79.43% | **79.78%** | 35.51% | 32.22% | **33.87%** |
| **Bone (b)** | 72.72% | 76.10% | **74.41%** | 28.69% | 25.75% | **27.22%** |
| **Joint Motion (jm)** | 80.12% | 80.05% | **80.09%** | 38.78% | 37.20% | **37.99%** |
| **Bone Motion (bm)** | 77.51% | 72.36% | **74.94%** | 31.90% | 29.06% | **30.48%** |

**MCA = Mean Class Accuracy** (average accuracy across all 5 classes, including background)

---

## 🎯 Per-Class Performance (PoseC3D)

### Fold 0

| Class | F1 Score | Precision | Recall | Status |
|-------|----------|-----------|--------|--------|
| **Hands Flapping** | 45.23% | 42.06% | 48.91% | ✅ Best |
| **Jumping** | 31.33% | 31.33% | 31.33% | ⚠️ Moderate |
| **Rocking** | 0.00% | 0.00% | 0.00% | ❌ **FAILED** |
| **Spinning** | 35.82% | 28.57% | 48.00% | ⚠️ Poor |

### Fold 1

| Class | F1 Score | Precision | Recall | Status |
|-------|----------|-----------|--------|--------|
| **Hands Flapping** | 41.95% | 46.92% | 37.93% | ✅ Best |
| **Jumping** | 31.09% | 30.08% | 32.17% | ⚠️ Moderate |
| **Rocking** | 0.00% | 0.00% | 0.00% | ❌ **FAILED** |
| **Spinning** | 6.25% | 9.09% | 4.76% | ❌ **VERY POOR** |

### Average Across Folds

| Class | Avg F1 | Std Dev |
|-------|--------|---------|
| **Hands Flapping** | 43.59% | ±1.64% |
| **Jumping** | 31.21% | ±0.12% |
| **Rocking** | 0.00% | ±0.00% |
| **Spinning** | 21.04% | ±14.79% |

---

## 🔍 Key Findings

### 1. **Severe Class Imbalance Despite Subsampling**

Even with 20% background subsampling:
- **Rocking** class: **0% F1 across both folds** → Model completely failed to learn this class
- **Spinning** class: Highly unstable (35.82% → 6.25% between folds)
- **Hands Flapping**: Most learnable (43.59% avg F1)
- **Jumping**: Moderate performance (31.21% avg F1)

**Root Cause**: These minority classes are extremely rare in the training data, and even with subsampling, the effective sample count is too low.

### 2. **Top-1 Accuracy vs. Mean Class Accuracy Gap**

#### PoseC3D
- Top-1 Accuracy: **85.89%**
- Macro F1 (4 RMM classes only): **37.62%**
- **Gap**: ~48%

#### STGCN++ (Average across modalities)
- Top-1 Accuracy: **77.31%**
- Mean Class Accuracy (5 classes): **32.39%**
- **Gap**: ~45%

**Interpretation**: Both models achieve high overall accuracy by correctly predicting the dominant **background** class, but struggle significantly with minority RMM classes.

### 3. **STGCN++ Modality Comparison**

**Best Modality**: **Joint Motion (jm)**
- Top-1 Acc: 80.09%
- Mean Class Acc: 37.99% (highest!)
- **Why**: Temporal velocity of joints captures the dynamic nature of RMMs better than static positions

**Worst Modality**: **Bone (b)**
- Top-1 Acc: 74.41%
- Mean Class Acc: 27.22%
- **Why**: Bone vectors (position-invariant) lose important spatial context needed for RMM classification

**Ranking**:
1. 🥇 Joint Motion (jm): 80.09% / 37.99% MCA
2. 🥈 Joint (j): 79.78% / 33.87% MCA  
3. 🥉 Bone Motion (bm): 74.94% / 30.48% MCA
4. Bone (b): 74.41% / 27.22% MCA

**Insight**: **Motion-based features (jm, bm) generally outperform static features (j, b)** for RMM detection, which makes sense given that RMMs are defined by repetitive movements.

### 4. **PoseC3D vs. STGCN++ (Joint Modality)**

| Model | Top-1 Acc | Notes |
|-------|-----------|-------|
| PoseC3D (joint) | **85.89%** | 3D CNN on pose heatmaps |
| STGCN++ (joint) | **79.78%** | Graph convolution on skeleton |

**Difference**: +6.11% in favor of PoseC3D

**Possible Reasons**:
- PoseC3D uses **pseudo-heatmaps** (2D spatial representation) which may preserve more spatial context
- STGCN++ uses **graph structure** which is more abstract but may lose fine-grained spatial details
- PoseC3D has **more parameters** and capacity

### 5. **Cross-Fold Stability**

#### PoseC3D
- Top-1 Acc: 85.33% → 86.45% (stable, ±0.56%)
- Macro F1: 40.85% → 34.39% (variable, ±3.23%)
- **Spinning class**: 35.82% → 6.25% (**HIGHLY UNSTABLE**)

#### STGCN++ Joint Motion (best modality)
- Top-1 Acc: 80.12% → 80.05% (very stable, ±0.04%)
- Mean Class Acc: 38.78% → 37.20% (stable, ±0.79%)

**Interpretation**: STGCN++ Joint Motion shows better cross-fold stability, especially for minority classes.

---

## 🚨 Critical Issues

### Issue 1: **Rocking Class - Complete Failure**
- **F1 Score**: 0.00% (both folds)
- **Samples**: Likely < 50 windows in training set after subsampling
- **Impact**: Model never predicts this class

**Potential Solutions**:
1. **More aggressive background subsampling** (e.g., 10% instead of 20%)
2. **Focal loss** or **class-balanced loss** to focus on hard examples
3. **Synthetic data augmentation** for minority classes
4. **Two-stage approach**: First detect RMM vs. background, then classify RMM type

### Issue 2: **Spinning Class - High Variance**
- **Fold 0**: 35.82% F1
- **Fold 1**: 6.25% F1
- **Variance**: ±14.79%

**Potential Causes**:
1. **Distribution shift** between folds (different children/environments)
2. **Small sample size** leads to high sensitivity to fold composition
3. **Annotation inconsistency** for spinning vs. other movements

### Issue 3: **Background Dominance**
- **Weighted F1**: 85.40% (high because background is weighted heavily)
- **Macro F1**: 37.62% (low because minority classes fail)

**Current Strategy**: Background subsampling helps but is not sufficient alone.

---

## 📈 Recommendations

### Short-term (Can implement now):
1. **Try more aggressive background subsampling**: 10% or even 5%
2. **Use focal loss** (γ=2) to focus on hard-to-classify examples
3. **Ensemble STGCN++ modalities**: Combine jm + j predictions
4. **Analyze failure cases**: Which videos/segments does the model get wrong?

### Medium-term (Requires more work):
1. **Two-stage TAL pipeline**:
   - Stage 1: Binary RMM vs. background detection (should be easier)
   - Stage 2: 4-class RMM classification (only on detected RMM windows)
2. **Synthetic augmentation** for minority classes:
   - Temporal jittering
   - Speed perturbation
   - Mixup between same-class examples
3. **Active learning**: Manually review and re-label uncertain predictions

### Long-term (Research directions):
1. **Few-shot learning** approaches for rare classes
2. **Contrastive learning** to learn better representations
3. **Hierarchical classification**: Group similar RMMs (e.g., flapping + spinning = "upper body")
4. **Multi-modal fusion**: Combine RGB (V-JEPA) + Pose (PoseC3D/STGCN++) predictions

---

## 🎬 Next Steps

1. **Wait for V-JEPA2 to finish** (currently on Epoch 2, val_acc=87.4% after Epoch 1)
   - V-JEPA2 may perform better due to RGB features capturing more context
   - Compare RGB vs. Pose modalities

2. **Train fold 2** for PoseC3D and STGCN++ to complete 3-fold CV

3. **Run TAL evaluation pipeline** on all trained models:
   ```bash
   python actreg/tal/run_tal_eval_cv.py --model posec3d --preds-root work_dirs/posec3d/tal/...
   python actreg/tal/run_tal_eval_cv.py --model stgcn --preds-root work_dirs/stgcnpp/tal/...
   ```

4. **Analyze confusion matrices** to understand which classes are confused with each other

5. **Experiment with different postprocessing parameters** for window-to-segment conversion:
   - Smoothing window size
   - Threshold per class
   - Minimum segment duration
   - Gap merging tolerance

---

## 📝 Summary Statistics

### Dataset (Folds 0-1)
- **Total windows**: ~7,341 (avg per fold)
- **Validation windows**: ~3,670 (avg per fold)
- **Classes**: 5 (4 RMM + background)
- **Window duration**: 2 seconds
- **Background subsampling**: 20% (effective 5x reduction)

### Training Configuration
- **Epochs**: 20
- **Learning rate**: 
  - PoseC3D: 0.00125
  - STGCN++: 0.01
- **Batch size**: 16 (videos_per_gpu)
- **Loss**: CrossEntropyLoss with class weights
- **Optimizer**: SGD with momentum (0.9) and Nesterov

### Computational Cost
- **PoseC3D**: ~1.5 hours per fold (1 GPU)
- **STGCN++ (all 4 modalities)**: ~2.5 hours total (1 GPU)
- **Total GPU hours**: ~6 hours for all models (folds 0-1)

---

**Generated**: 2026-01-04  
**Author**: Automated analysis script  
**Data**: PoseC3D + STGCN++ TAL training results (folds 0-1)


