# Window-Level Classification Comparison (5-Class Detection)

**Date:** January 5, 2026  
**Task:** 5-class window classification (background, hands_flapping, jumping, rocking, spinning)  
**Evaluation Metrics:** Top-1/Top-2 Accuracy, Macro F1, Per-class F1

---

## Overview

This document summarizes the **window-level classification performance** of different models for the 5-class RMM detection task. Unlike TAL (Temporal Action Localization), which evaluates segment-level localization with mAP@tIoU, this comparison focuses on **per-window classification accuracy and F1 scores**.

### Models Evaluated

| Model | Type | Input | Description |
|-------|------|-------|-------------|
| **V-JEPA** | Video Foundation Model | RGB frames | Pre-trained video encoder with linear probe |
| **PoseC3D (CE)** | 3D CNN on Pose | Pose heatmaps | SlowOnly R50 with Cross-Entropy loss |
| **PoseC3D (Focal)** | 3D CNN on Pose | Pose heatmaps | SlowOnly R50 with Focal loss (gamma=2) |
| **STGCN++ 4-Stream (CE)** | Graph Neural Network | Skeleton graphs | Joint + Bone + Motion streams, fused, CE loss |
| **STGCN++ 4-Stream (Focal)** | Graph Neural Network | Skeleton graphs | Joint + Bone + Motion streams, Focal loss (gamma=1) |
| **V-JEPA + PoseC3D (MLP)** | Late Fusion | RGB + Pose | MLP fusion on log-probabilities |
| **V-JEPA + STGCN++ (MLP)** | Late Fusion | RGB + Skeleton | MLP fusion on log-probabilities |

### Evaluation Setup

- **Cross-validation:** 2-3 folds (varies by model)
- **Window size:** 2 seconds with 1 second stride (50% overlap)
- **Classes:** 5 classes (background + 4 RMM types)
- **Class imbalance:** ~90% background windows

---

## Overall Classification Results

### Summary Table

| Model | Folds | Top-1 Acc | Top-2 Acc | Macro F1 | Cohen's κ |
|-------|-------|-----------|-----------|----------|-----------|
| **PoseC3D (CE)** | 2 | **85.89% ± 0.56%** | **96.16% ± 0.48%** | 37.62% ± 3.23% | **0.33 ± 0.02** |
| STGCN++ 4-Stream (CE) | 2 | 85.74% ± 0.14% | 95.18% ± 0.11% | 32.85% ± 1.48% | 0.30 ± 0.00 |
| PoseC3D (Focal) | 3 | 84.31% ± 1.33% | 96.07% ± 0.30% | 31.47% ± 1.80% | 0.25 ± 0.02 |
| V-JEPA | 2 | 81.75% ± 3.28% | 93.35% ± 2.03% | 36.10% ± 3.23% | N/A |
| STGCN++ 4-Stream (Focal) | 3 | 37.13% ± 3.75% | 74.95% ± 8.21% | 19.56% ± 1.18% | 0.06 ± 0.01 |
| **V-JEPA + PoseC3D (MLP)** | 2 | 48.10% ± 2.16%* | N/A | N/A | N/A |
| **V-JEPA + STGCN++ (MLP)** | 2 | 57.15% ± 8.63%* | N/A | N/A | N/A |

*Fusion OOF accuracy on validation windows only; not directly comparable to single model test accuracy.

**Key observations:**
- **PoseC3D (CE) achieves best overall accuracy** (85.89% top-1, 96.16% top-2)
- **STGCN++ CE is competitive** (85.74% top-1), nearly matching PoseC3D
- **V-JEPA shows high variance** between folds (78.5% - 85.0%)
- **Focal loss underperforms CE** for both architectures
- **STGCN++ Focal collapsed** (~37% accuracy vs ~86% for CE) — training instability

### Top-2 Accuracy Analysis

Top-2 accuracy measures whether the correct class is among the model's top 2 predictions — useful for understanding model uncertainty.

| Model | Top-1 Acc | Top-2 Acc | Gap | Interpretation |
|-------|-----------|-----------|-----|----------------|
| **PoseC3D (CE)** | 85.89% | **96.16%** | +10.3% | Strong second-choice predictions |
| PoseC3D (Focal) | 84.31% | 96.07% | +11.8% | Comparable top-2 to CE |
| STGCN++ CE | 85.74% | 95.18% | +9.4% | Good uncertainty calibration |
| V-JEPA | 81.75% | 93.35% | +11.6% | Reasonable top-2 despite lower top-1 |
| STGCN++ Focal | 37.13% | 74.95% | +37.8% | Massive gap — model is uncertain but recoverable |

**Key insights:**
- **All working models achieve >93% top-2** — the correct class is almost always in top 2
- **~10% gap between top-1 and top-2** suggests models struggle with RMM vs background decision boundary
- **STGCN++ Focal's huge gap (37.8%)** indicates the model often has the right answer as second choice, but poor confidence calibration puts background first

---

## Per-Class F1 Scores

### RMM Classes (excluding background)

| Model | Hands Flapping | Jumping | Rocking | Spinning | RMM Macro F1 |
|-------|----------------|---------|---------|----------|--------------|
| **V-JEPA** | **41.57% ± 1.3%** | **31.77% ± 4.9%** | **5.44% ± 1.3%** | 11.80% ± 6.8% | **22.65%** |
| PoseC3D (CE) | 43.59% ± 2.4% | 31.21% ± 0.1% | 0.00% ± 0.0% | **21.04% ± 14.8%** | 23.96% |
| STGCN++ CE | 39.82% ± 0.3% | 25.00% ± 0.5% | 0.00% ± 0.0% | 7.14% ± 7.1% | 17.99% |
| PoseC3D (Focal) | 35.83% ± 1.9% | 24.55% ± 3.8% | 0.00% ± 0.0% | 5.44% ± 7.7% | 16.46% |
| STGCN++ Focal | 21.11%* | 18.25%* | 0.00%* | 0.00%* | 9.84%* |

*STGCN++ Focal per-class F1 from clip-level metrics, averaged across 3 folds.

### Background Class F1

| Model | Background F1 |
|-------|---------------|
| V-JEPA | 89.90% ± 1.86% |
| PoseC3D (CE) | N/A** |
| STGCN++ CE | 92.31% ± 0.06% |
| PoseC3D (Focal) | N/A** |
| STGCN++ Focal | 47.10%* |

**PoseC3D metrics don't include explicit background F1 in available JSON files.

---

## Analysis

### Why Classification ≠ TAL Performance

| Model | Classification Rank | TAL Rank | Notes |
|-------|---------------------|----------|-------|
| PoseC3D (CE) | 1 (85.9% acc) | 3 (5.18% mAP) | High accuracy but temporally imprecise |
| STGCN++ CE | 2 (85.7% acc) | 6 (3.95% mAP) | Good classification, poor localization |
| V-JEPA | 4 (81.8% acc) | 4 (5.07% mAP) | Moderate both |
| **V-JEPA + PoseC3D** | N/A | **1 (6.36% mAP)** | Fusion excels at TAL |

**Key insight:** Window-level accuracy doesn't directly predict TAL performance because:
1. **TAL penalizes boundary errors** — Classification doesn't care about exact timing
2. **TAL requires temporal coherence** — Isolated correct predictions don't form good segments
3. **Fusion helps TAL more than classification** — Complementary errors average out in TAL

### Class-Specific Insights

| Class | Best Model | F1 Score | Challenge |
|-------|------------|----------|-----------|
| **Hands Flapping** | PoseC3D (CE) | 43.6% | Most frequent RMM, distinctive arm motion |
| **Jumping** | V-JEPA | 31.8% | Clear vertical displacement helps RGB |
| **Rocking** | V-JEPA | 5.4% | Extremely difficult — subtle, short |
| **Spinning** | PoseC3D (CE) | 21.0% | Full-body rotation captured by pose |
| **Background** | STGCN++ CE | 92.3% | Dominant class, easy to classify |

### Rocking: The Hardest Class

All models struggle with **rocking** (0-5% F1):
- **Very few training samples** relative to other RMMs
- **Subtle motion** — small amplitude, hard to distinguish from stillness
- **Short duration** — often just 1-2 windows
- **Confusion with background** — looks like sitting/standing still

---

## CE vs Focal Loss Comparison

| Architecture | CE Top-1 | Focal Top-1 | CE Top-2 | Focal Top-2 | CE Macro F1 | Focal Macro F1 |
|--------------|----------|-------------|----------|-------------|-------------|----------------|
| PoseC3D | **85.89%** | 84.31% | 96.16% | 96.07% | **37.62%** | 31.47% |
| STGCN++ 4-Stream | **85.74%** | 37.13% | **95.18%** | 74.95% | **32.85%** | 19.56% |

**Findings:**
- **CE consistently outperforms Focal** for this imbalanced dataset
- **Top-2 accuracy similar for PoseC3D** (CE vs Focal), but top-1 differs — Focal miscalibrates confidence
- **STGCN++ + Focal is catastrophic** — training collapsed (premature early stopping)
- Focal loss hypothesis (better for imbalanced classes) didn't hold here

---

## Fusion Model Analysis

### OOF Accuracy vs Single Model Accuracy

| Fusion Model | OOF Accuracy | Single Model Accuracy |
|--------------|--------------|----------------------|
| V-JEPA + PoseC3D | 48.1% | V-JEPA: 81.8%, PoseC3D: 85.9% |
| V-JEPA + STGCN++ | 57.2% | V-JEPA: 81.8%, STGCN++: 85.7% |

**Why fusion OOF accuracy is lower:**
1. **OOF training on val windows** — MLP only sees validation distribution
2. **Limited training data** — ~4000 windows per fold for inner CV
3. **Missing modality handling** — ~27% windows have no skeleton predictions

**Despite lower window accuracy, fusion achieves best TAL mAP** — suggesting the MLP learns complementary information that helps with temporal localization.

---

## Recommendations

### For Window-Level Classification
- Use **PoseC3D (CE)** for highest accuracy (85.9%)
- Use **STGCN++ CE** for comparable accuracy with lighter inference

### For Rare Class Detection
- **V-JEPA** best for rocking/spinning (non-zero F1)
- Consider **class-weighted loss** or **oversampling** for rare classes

### For Temporal Action Localization
- Use **V-JEPA + PoseC3D (MLP) fusion** — best TAL performance despite lower window accuracy
- Fusion captures complementary information between RGB and pose modalities

---

## Files and Resources

### Classification Results
- `v-jepa/runs/vjepa2_tal_cv_5class_bgsub/fold_{N}/per_class_window.csv` — V-JEPA per-class metrics
- `pyskl/work_dirs/posec3d/tal/cv_4class_5class_bgsub/fold{N}/eval_val/metrics.json` — PoseC3D CE
- `pyskl/work_dirs/posec3d/tal/cv_4class_5class_focal/cv_summary.json` — PoseC3D Focal CV summary
- `pyskl/work_dirs/stgcnpp/tal/cv_4class_5class_bgsub/4stream/cv_summary_partial.json` — STGCN++ CE
- `pyskl/work_dirs/stgcnpp/tal/cv_4class_5class_focal/4stream/cv_summary.json` — STGCN++ Focal

### Fusion Results
- `tal/eval_results/vjepa_posec3d_mlp_logp/fold{N}/fusion_stats.json` — V-JEPA + PoseC3D fusion stats
- `tal/eval_results/vjepa_stgcnpp_mlp_logp/fold{N}/fusion_stats.json` — V-JEPA + STGCN++ fusion stats

### Related Documents
- `TAL_MODEL_COMPARISON.md` — TAL (segment-level) evaluation with mAP@tIoU metrics

