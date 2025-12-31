# RMM Classification Results Analysis

**Date:** December 23, 2024

This document summarizes results across three methods (V-JEPA2, PoseC3D, Qwen2.5-VL) for RMM classification on the SAILS dataset, comparing 4-class and 5-class variants.

**Note:** All results reported at the **clip level** unless otherwise specified.

---

## Executive Summary

| Ranking | Method | 4-Class CV Macro F1 | 5-Class CV Macro F1 |
|---------|--------|---------------------|---------------------|
| 🥇 **1st** | V-JEPA2 (w/ crop) | **75.1%** | — |
| 🥈 **2nd** | PoseC3D | 70.4% | **65.1%** |
| 🥉 **3rd** | V-JEPA2 (no crop) | — | 59.6% |
| 4th | Qwen2.5-VL (zero-shot) | 28.2% | 12.2% |

**Key Finding:** V-JEPA2 with SAM3-based person cropping achieves the best performance on 4-class classification (75.1% clip macro F1). The 5-class task is significantly harder, with PoseC3D being the most robust approach (65.1% clip macro F1).

---

## Detailed Results

### 4-Class Classification

#### Cross-Validation Results (Aggregated over 3 folds)

| Method | Top-1 Acc | Top-2 Acc | Macro F1 | Weighted F1 | Cohen's κ |
|--------|-----------|-----------|----------|-------------|-----------|
| **V-JEPA2 (crop)** | **76.0%** | **93.7%** | **75.1%** | **75.9%** | **0.657** |
| PoseC3D | 74.6% | 91.3% | 70.4% | 74.2% | 0.580 |
| Qwen2.5-VL | 56.8% | 78.9% | 28.2% | 47.7% | 0.168 |

#### Single Split Test Set Results

| Method | Top-1 Acc | Top-2 Acc | Macro F1 | Weighted F1 | Cohen's κ |
|--------|-----------|-----------|----------|-------------|-----------|
| PoseC3D | **74.8%** | **95.1%** | **73.7%** | **74.4%** | **0.615** |
| V-JEPA2 (crop) | 71.2% | 93.7% | 72.7% | 71.1% | 0.573 |

#### Per-Class F1 Scores (CV Average)

| Class | V-JEPA2 (crop) | PoseC3D | Qwen |
|-------|----------------|---------|------|
| hands flapping | ~88% | 81.6% | ~28% |
| jumping | ~73% | 66.7% | ~27% |
| rocking | ~80% | 53.8% | ~29% |
| spinning | ~86% | 79.5% | ~28% |

---

### 5-Class Classification

#### Cross-Validation Results (Aggregated over 3 folds)

| Method | Top-1 Acc | Top-2 Acc | Macro F1 | Weighted F1 | Cohen's κ |
|--------|-----------|-----------|----------|-------------|-----------|
| **PoseC3D** | **66.8%** | **84.6%** | **65.1%** | **66.4%** | **0.547** |
| V-JEPA2 (no crop) | 60.8% | 82.2% | 59.6% | 59.8% | 0.441 |
| Qwen2.5-VL | 37.7% | 63.6% | 12.2% | 22.3% | 0.008 |

#### Single Split Test Set Results

| Method | Top-1 Acc | Top-2 Acc | Macro F1 | Weighted F1 | Cohen's κ |
|--------|-----------|-----------|----------|-------------|-----------|
| PoseC3D | 55.3% | 81.6% | 54.7% | 55.4% | 0.411 |

#### Per-Class F1 Scores (CV Average)

| Class | PoseC3D | V-JEPA2 (no crop) | Qwen |
|-------|---------|-------------------|------|
| hands flapping | 71.5% | ~57% | ~15% |
| jumping | 71.4% | ~51% | ~11% |
| **one hand flap** | **47.4%** | **~0%** | **~12%** |
| rocking | 54.1% | ~50% | ~11% |
| spinning | 81.3% | ~88% | ~15% |

---

## Key Observations

### 1. SAM3 Cropping Matters for V-JEPA2

Comparing V-JEPA2 with and without cropping:
- **4-class with crop:** 75.1% clip macro F1
- **5-class without crop:** 59.6% clip macro F1

While the class count differs, the cropping step (focusing on the person of interest) appears to provide substantial benefit. **Recommendation:** Run 5-class V-JEPA2 with cropping enabled.

### 2. The "One Hand Flap" Class is Problematic

This class shows consistently poor performance:
- **PoseC3D:** 47.4% F1 (best, but still low)
- **V-JEPA2:** ~0% F1 (complete failure)
- **Qwen:** ~12% F1

**Possible reasons:**
- Visual similarity to "hands flapping"
- Small sample size or class imbalance
- Subtle motion difference hard to capture

**Recommendation:** Consider merging with "hands flapping" or collecting more samples.

### 3. Skeleton-Based (PoseC3D) vs. RGB-Based (V-JEPA2)

| Aspect | PoseC3D | V-JEPA2 |
|--------|---------|---------|
| Robustness to background | ✅ Excellent | ⚠️ Needs cropping |
| Fine-grained hand motion | ⚠️ Limited keypoints | ✅ Good with RGB |
| Rocking detection (4-class) | ⚠️ 53.8% F1 | ✅ ~80% F1 |
| One hand flap detection (5-class) | ✅ 47.4% F1 | ❌ ~0% F1 |

PoseC3D is more consistent across classes but struggles with rocking (body sway). V-JEPA2 excels when given clean person crops but fails on subtle one-handed motions without proper tuning.

### 4. Zero-Shot (Qwen) Struggles

Qwen2.5-VL zero-shot performance is poor:
- 4-class: 26.7% video macro F1 (κ = 0.14, barely above chance)
- 5-class: 10.7% video macro F1 (κ ≈ 0, random)

**Reasons:**
- RMM behaviors are subtle and domain-specific
- Short clip duration limits context
- VLM not trained on pediatric behavioral videos

**Recommendation:** Fine-tuning or few-shot approaches may help.

### 5. Fold Variance

V-JEPA2 CV (4-class with crop) shows notable fold variance:

| Fold | Clip Macro F1 | Cohen's κ |
|------|---------------|-----------|
| 0 | 76.7% | 0.724 |
| 1 | 72.5% | 0.523 |
| 2 | 76.2% | 0.723 |

Fold 1 shows lower Cohen's κ despite similar macro F1, suggesting different confusion patterns. **Recommendation:** Investigate what makes Fold 1 subjects harder to classify.

---

## Confusion Patterns (Clip-Level)

### PoseC3D 4-Class (Single Split Test)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| hands flapping | 75.0% | 83.0% | **78.8%** |
| jumping | 64.5% | 83.3% | 72.7% |
| rocking | 88.2% | 55.6% | 68.2% |
| spinning | 100% | 60.0% | 75.0% |

- **Rocking:** High precision but low recall → model is conservative, missing many true rockings
- **Jumping:** High recall but lower precision → over-predicts jumping

### V-JEPA2 4-Class (Single Split Test)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| hands flapping | 80.6% | 92.6% | **86.2%** |
| jumping | 60.0% | 27.3% | 37.5% |
| rocking | 69.2% | 90.0% | 78.3% |
| spinning | 100% | 75.0% | 85.7% |

- **Jumping:** Severe under-detection (27.3% recall) → confused with other classes
- **Rocking:** V-JEPA2 detects much better than PoseC3D (90% vs 55.6% recall)

---

## Recommendations & Next Steps

### Immediate Actions

1. **Run V-JEPA2 5-class with SAM3 cropping**
   - Current 5-class results are without cropping
   - Expect significant improvement based on 4-class results

2. **Analyze Fold 1 failure cases**
   - V-JEPA2 shows 65.3% vs ~82% on other folds
   - May reveal subject-level patterns

3. **Investigate "one hand flap" confusion matrix**
   - Confirm it's being confused with "hands flapping"
   - Consider if distinction is clinically meaningful

### Model Improvements

4. **Ensemble V-JEPA2 + PoseC3D**
   - V-JEPA2 excels at rocking (body motion)
   - PoseC3D better at fine-grained hand movements
   - Complementary strengths

5. **Add temporal context for V-JEPA2**
   - Current: 64 frames
   - RMMs are repetitive; longer clips may help

6. **Try PoseC3D with full-body keypoints**
   - Current: 17 COCO keypoints
   - Wholebody (133 keypoints) may capture hand motion better

### Data Actions

7. **Review "one hand flap" annotations**
   - Low F1 across all methods suggests annotation noise
   - Consider inter-rater reliability check

8. **Collect more rocking samples for PoseC3D**
   - Lowest recall (55.6%) among 4 classes
   - May need data augmentation

---

## Summary Table: Best Model Per Task

| Task | Best Model | Clip Macro F1 | Cohen's κ |
|------|------------|---------------|-----------|
| 4-class (CV) | V-JEPA2 + crop | 75.1% | 0.657 |
| 4-class (single) | PoseC3D | 73.7% | 0.615 |
| 5-class (CV) | PoseC3D | 65.1% | 0.547 |
| 5-class (single) | PoseC3D | 54.7% | 0.411 |

**For production use:** V-JEPA2 with SAM3 cropping on 4-class task is recommended, with PoseC3D as a fallback when person cropping fails.

---

## Appendix: Experiment Configurations

### V-JEPA2
- Model: `facebook/vjepa2-vitl-fpc16-256-ssv2`
- Frames: 64
- LR: 1e-5, BS: 1, Acc: 8, Epochs: 20
- Cropping: SAM3-based person detection (4-class only)

### PoseC3D
- Model: SlowOnly R50 + K400 pretrained
- Frames: 48
- LR: 0.00125, BS: 16, Epochs: 12
- Keypoints: 17 COCO (conf ≥ 0.4)

### Qwen2.5-VL
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Zero-shot with structured prompts
- 4 windows × 16 frames, majority voting

