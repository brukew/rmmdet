# Window-Level Classification Comparison (5-Class Detection)

**Date:** January 6, 2026  
**Task:** 5-class window classification (background, hands_flapping, jumping, rocking, spinning)  
**Evaluation Metrics:** Top-1/Top-2 Accuracy, Macro F1, Per-class F1

---

## Overview

This document summarizes the **window-level classification performance** of different models for the 5-class RMM detection task. Unlike TAL (Temporal Action Localization), which evaluates segment-level localization with mAP@tIoU, this comparison focuses on **per-window classification accuracy and F1 scores**.

### Models Evaluated

| Model | Type | Input | Description |
|-------|------|-------|-------------|
| **V-JEPA** | Video Foundation Model | RGB frames | Pre-trained video encoder with linear probe |
| **V-JEPA (Balanced)** | Video Foundation Model | RGB frames | Pre-trained encoder, class-weighted sampling† |
| **V-JEPA Binary (Balanced)** | Video Foundation Model | RGB frames | Binary (BG vs RMM), class-weighted sampling† |
| **PoseC3D (CE)** | 3D CNN on Pose | Pose heatmaps | SlowOnly R50 with Cross-Entropy loss |
| **PoseC3D (Focal)** | 3D CNN on Pose | Pose heatmaps | SlowOnly R50 with Focal loss (gamma=2) |
| **PoseC3D (CE Balanced)** | 3D CNN on Pose | Pose heatmaps | SlowOnly R50 with CE + balanced sampling |
| **STGCN++ 4-Stream (CE)** | Graph Neural Network | Skeleton graphs | Joint + Bone + Motion streams, fused, CE loss |
| **STGCN++ 4-Stream (CE Balanced)** | Graph Neural Network | Skeleton graphs | Joint + Bone + Motion streams, fused, CE + balanced sampling |
| **STGCN++ 4-Stream (Focal)** | Graph Neural Network | Skeleton graphs | Joint + Bone + Motion streams, Focal loss (gamma=1) |
| **V-JEPA + PoseC3D (MLP)** | Late Fusion | RGB + Pose | MLP fusion on log-probabilities |
| **V-JEPA + STGCN++ (MLP)** | Late Fusion | RGB + Skeleton | MLP fusion on log-probabilities |
| **V-JEPA + PoseC3D Bal (MLP)** | Late Fusion | RGB + Pose | MLP fusion with balanced PoseC3D |

### Evaluation Setup

- **Cross-validation:** 2-3 folds (varies by model)
- **Window size:** 2 seconds with 1 second stride (50% overlap)
- **Classes:** 5 classes (background + 4 RMM types)
- **Class imbalance:** ~90% background windows

### Balancing Strategies

**†V-JEPA Balanced Sampling:**
Uses class probability sampling during training (not loss weighting):

| Class | Count | Probability | Effect |
|-------|-------|-------------|--------|
| hands_flapping | ~500 | 1.0 | Keep all |
| jumping | ~237 | 1.0 | Keep all (reference) |
| rocking | ~123 | **1.93** | Upsample ~2× to match jumping |
| spinning | ~26 | **9.12** | Upsample ~9× to match jumping |
| background | ~7373 | **0.1** | Downsample to 10% |

**V-JEPA Binary Balanced:**
- RMM (all types): prob=1.0 (keep all)
- Background: prob=0.1 (downsample to 10%)

**PoseC3D/STGCN++ Balanced:**
Uses `RepeatDataset` for oversampling minority classes with balanced class weights in the sampler.

---

## Overall Classification Results

### Summary Table

| Model | Folds | Top-1 Acc | Top-2 Acc | Macro F1 | Cohen's κ |
|-------|-------|-----------|-----------|----------|-----------|
| **PoseC3D (CE)** | 2 | **85.89% ± 0.56%** | 96.16% ± 0.48% | 37.62% ± 3.23% | **0.33 ± 0.02** |
| STGCN++ 4-Stream (CE) | 2 | 85.74% ± 0.14% | 95.18% ± 0.11% | 32.85% ± 1.48% | 0.30 ± 0.00 |
| PoseC3D (Focal) | 3 | 84.31% ± 1.33% | 96.07% ± 0.30% | 31.47% ± 1.80% | 0.25 ± 0.02 |
| V-JEPA | 2 | 81.75% ± 3.28% | 93.35% ± 2.03% | 36.10% ± 3.23% | N/A |
| **PoseC3D (CE Balanced)** | 3 | 81.11% ± 1.67% | **96.34% ± 0.39%** | 35.82% ± 5.78% | 0.28 ± 0.02 |
| **V-JEPA (Balanced)** | 3 | 80.94% ± 1.73% | 95.08% ± 0.87% | **36.83% ± 3.45%** | 0.31 ± 0.02 |
| V-JEPA Binary (Balanced) | 3 | 78.12% ± 3.11% | 100%† | 62.15% ± 1.17%† | 0.27 ± 0.02 |
| STGCN++ 4-Stream (CE Balanced) | 3 | 77.48% ± 3.75% | N/A | 30.40% ± 1.35% | N/A |
| STGCN++ 4-Stream (Focal) | 3 | 37.13% ± 3.75% | 74.95% ± 8.21% | 19.56% ± 1.18% | 0.06 ± 0.01 |
| **V-JEPA + PoseC3D (MLP)** | 2 | 48.10% ± 2.16%* | N/A | N/A | N/A |
| **V-JEPA + STGCN++ (MLP)** | 2 | 57.15% ± 8.63%* | N/A | N/A | N/A |
| **V-JEPA + PoseC3D Bal (MLP)** 🏆 | 2 | 49.36% ± 1.02%* | N/A | N/A | N/A |

*Fusion OOF accuracy on validation windows only; not directly comparable to single model test accuracy.
†Binary model (BG vs RMM only) — Top-2 is trivially 100%, Macro F1 is 2-class average.
🏆 Best TAL mAP (8.40%) despite lower window accuracy.

**Key observations:**
- **PoseC3D (CE) achieves best overall accuracy** (85.89% top-1, 96.16% top-2)
- **STGCN++ CE is competitive** (85.74% top-1), nearly matching PoseC3D
- **PoseC3D Balanced trades accuracy for better rare class detection** (81.11% vs 85.89%)
- **V-JEPA Balanced is most stable** (80.94% ± 1.73%) — smallest accuracy drop (-0.8%) with balanced sampling
- **STGCN++ CE Balanced underperforms** (77.48% top-1) — balanced sampling hurt STGCN++ more than PoseC3D (-8.3% vs -4.8%)
- **V-JEPA handles balancing best** — only -0.8% accuracy drop vs -4.8% (PoseC3D) and -8.3% (STGCN++)
- **V-JEPA Binary achieves highest RMM recall** (67.9%) — good as first-stage detector
- **Focal loss underperforms CE** for both architectures
- **STGCN++ Focal collapsed** (~37% accuracy vs ~86% for CE) — training instability
- **V-JEPA + PoseC3D Balanced achieves best TAL mAP (8.40%)** despite lower OOF accuracy

### Top-2 Accuracy Analysis

Top-2 accuracy measures whether the correct class is among the model's top 2 predictions — useful for understanding model uncertainty.

| Model | Top-1 Acc | Top-2 Acc | Gap | Interpretation |
|-------|-----------|-----------|-----|----------------|
| **PoseC3D (CE)** | 85.89% | 96.16% | +10.3% | Strong second-choice predictions |
| **PoseC3D (CE Balanced)** | 81.11% | **96.34%** | +15.2% | **Best top-2!** Trades top-1 for better uncertainty |
| V-JEPA (Balanced) | 80.94% | 95.08% | +14.1% | More stable, good uncertainty |
| PoseC3D (Focal) | 84.31% | 96.07% | +11.8% | Comparable top-2 to CE |
| STGCN++ CE | 85.74% | 95.18% | +9.4% | Good uncertainty calibration |
| V-JEPA | 81.75% | 93.35% | +11.6% | Reasonable top-2 despite lower top-1 |
| STGCN++ Focal | 37.13% | 74.95% | +37.8% | Massive gap — model is uncertain but recoverable |

**Key insights:**
- **All working models achieve >93% top-2** — the correct class is almost always in top 2
- **PoseC3D Balanced achieves best top-2 (96.34%)** despite lower top-1 — valuable for fusion
- **~10-15% gap between top-1 and top-2** suggests models struggle with RMM vs background decision boundary
- **PoseC3D Balanced's larger gap (15.2%)** indicates more aggressive RMM predictions — beneficial for TAL
- **STGCN++ Focal's huge gap (37.8%)** indicates the model often has the right answer as second choice, but poor confidence calibration puts background first

---

## Per-Class F1 Scores

### RMM Classes (excluding background)

| Model | Hands Flapping | Jumping | Rocking | Spinning | RMM Macro F1 |
|-------|----------------|---------|---------|----------|--------------|
| **V-JEPA** | 41.57% ± 1.3% | 31.77% ± 4.9% | 5.44% ± 1.3% | 11.80% ± 6.8% | 22.65% |
| **V-JEPA (Balanced)** | 40.90% ± 2.9% | **33.88% ± 4.4%** | **6.67% ± 4.1%** | 13.36% ± 11.9% | **23.70%** |
| PoseC3D (CE) | **43.59% ± 2.4%** | 31.21% ± 0.1% | 0.00% ± 0.0% | **21.04% ± 14.8%** | 23.96% |
| **PoseC3D (CE Bal)** | 37.46% ± 3.0% | 27.91% ± 4.0% | 2.76% ± 2.3% | 21.48% ± 25.8% | 22.40% |
| STGCN++ CE | 39.82% ± 0.3% | 25.00% ± 0.5% | 0.00% ± 0.0% | 7.14% ± 7.1% | 17.99% |
| PoseC3D (Focal) | 35.83% ± 1.9% | 24.55% ± 3.8% | 0.00% ± 0.0% | 5.44% ± 7.7% | 16.46% |
| **STGCN++ CE Bal** | 34.32% ± 4.7% | 22.88% ± 1.5% | 4.22% ± 3.2% | 7.68% ± 7.4% | 17.28% |
| STGCN++ Focal | 21.11%* | 18.25%* | 0.00%* | 0.00%* | 9.84%* |

*STGCN++ Focal per-class F1 from clip-level metrics, averaged across 3 folds.

**Balanced Sampling Key Findings:**
- **V-JEPA Balanced: Best rocking detection** (6.67% F1) with minimal accuracy penalty (-0.8%). Also improves jumping (+2.1%) and spinning (+1.6%).
- **PoseC3D CE Balanced:** Non-zero rocking F1 (2.76%) — the only PoseC3D variant to detect rocking. High variance in spinning (0-58% across folds).
- **STGCN++ CE Balanced:** Achieves rocking F1 of 4.22% ± 3.2% — comparable to PoseC3D Balanced but with larger accuracy penalty (-8.3% vs -4.8%).

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
| PoseC3D (CE) | 1 (85.9% acc) | 4 (5.18% mAP) | High accuracy but temporally imprecise |
| STGCN++ CE | 2 (85.7% acc) | 7 (3.95% mAP) | Good classification, poor localization |
| PoseC3D (CE Balanced) | 5 (81.1% acc) | 8 (2.55% mAP) | Lower accuracy AND poor TAL |
| V-JEPA | 4 (81.8% acc) | 5 (5.07% mAP) | Moderate both |
| **V-JEPA + PoseC3D Bal** | N/A (49% OOF) | **1 (8.40% mAP)** 🏆 | **Best TAL despite lowest accuracy!** |
| V-JEPA + PoseC3D | N/A (48% OOF) | 2 (6.36% mAP) | Fusion excels at TAL |
| V-JEPA + STGCN++ | N/A (57% OOF) | 3 (5.66% mAP) | Fusion helps |

**Key insight:** Window-level accuracy doesn't directly predict TAL performance because:
1. **TAL penalizes boundary errors** — Classification doesn't care about exact timing
2. **TAL requires temporal coherence** — Isolated correct predictions don't form good segments
3. **Fusion helps TAL more than classification** — Complementary errors average out in TAL

### Class-Specific Insights

| Class | Best Model | F1 Score | Challenge |
|-------|------------|----------|-----------|
| **Hands Flapping** | PoseC3D (CE) | 43.6% | Most frequent RMM, distinctive arm motion |
| **Jumping** | V-JEPA (Balanced) | **33.9%** | Clear vertical displacement helps RGB |
| **Rocking** | V-JEPA (Balanced) | **6.7%** | Extremely difficult — subtle, short |
| **Spinning** | PoseC3D (CE Bal) | 21.5% | Full-body rotation captured by pose |
| **Background** | STGCN++ CE | 92.3% | Dominant class, easy to classify |

### Rocking: The Hardest Class

All models struggle with **rocking** (0-5% F1):
- **Very few training samples** relative to other RMMs
- **Subtle motion** — small amplitude, hard to distinguish from stillness
- **Short duration** — often just 1-2 windows
- **Confusion with background** — looks like sitting/standing still

---

## Loss Function Comparison

### CE vs Focal Loss

| Architecture | CE Top-1 | Focal Top-1 | CE Top-2 | Focal Top-2 | CE Macro F1 | Focal Macro F1 |
|--------------|----------|-------------|----------|-------------|-------------|----------------|
| PoseC3D | **85.89%** | 84.31% | 96.16% | 96.07% | **37.62%** | 31.47% |
| STGCN++ 4-Stream | **85.74%** | 37.13% | **95.18%** | 74.95% | **32.85%** | 19.56% |

**Findings:**
- **CE consistently outperforms Focal** for this imbalanced dataset
- **Top-2 accuracy similar for PoseC3D** (CE vs Focal), but top-1 differs — Focal miscalibrates confidence
- **STGCN++ + Focal is catastrophic** — training collapsed (premature early stopping)
- Focal loss hypothesis (better for imbalanced classes) didn't hold here

### CE vs CE + Balanced Sampling

| Architecture | CE Top-1 | CE Balanced Top-1 | Δ | CE Macro F1 | CE Bal Macro F1 | Rocking F1: CE | Rocking F1: CE Bal |
|--------------|----------|-------------------|---|-------------|-----------------|----------------|---------------------|
| **V-JEPA** | 81.75% | **80.94%** | **-0.8%** | 36.10% | **36.83%** | 5.44% | **6.67%** ✓ |
| PoseC3D | **85.89%** | 81.11% | -4.8% | **37.62%** | 35.82% | 0.00% | **2.76%** ✓ |
| STGCN++ 4-Stream | **85.74%** | 77.48% | -8.3% | **32.85%** | 30.40% | 0.00% | **4.22%** ✓ |

**Findings:**
- **V-JEPA handles balancing best** — only 0.8% accuracy drop while improving rare class F1
- **Balanced sampling enables rocking detection** — all architectures improve rocking F1
- **STGCN++ hurt most by balancing** — 8.3% accuracy drop vs 4.8% (PoseC3D) and 0.8% (V-JEPA)
- **PoseC3D Balanced achieves best top-2 accuracy** (96.34%) despite lower top-1
- **Trade-off:** 0.8-8% accuracy loss for rare class detection (V-JEPA has best trade-off)
- **For fusion:** V-JEPA Balanced + PoseC3D Balanced is a promising combination

### Binary (BG vs RMM) Detection

V-JEPA Binary Balanced provides a **high-recall RMM detector**:

| Metric | V-JEPA Binary (Balanced) |
|--------|--------------------------|
| **Top-1 Accuracy** | 78.12% ± 3.11% |
| **RMM Precision** | 26.14% ± 1.90% |
| **RMM Recall** | **67.89% ± 5.14%** |
| **RMM F1** | 37.67% ± 2.23% |
| **RMM vs BG AUC** | 79.72% ± 0.29% |
| **BG False Alarm Rate** | 20.89% ± 3.95% |
| **RMM Miss Rate** | 32.11% ± 5.14% |

**Use case:** First-stage detector to catch RMM events before fine-grained classification. High recall (68%) ensures most RMM events are captured, at the cost of more false positives.

---

## Fusion Model Analysis

### OOF Accuracy vs Single Model Accuracy

| Fusion Model | OOF Accuracy | Single Model Accuracy | TAL mAP |
|--------------|--------------|----------------------|---------|
| **V-JEPA + PoseC3D Bal** | 49.4% | V-JEPA: 81.8%, PoseC3D Bal: 81.1% | **8.40%** 🏆 |
| V-JEPA + PoseC3D | 48.1% | V-JEPA: 81.8%, PoseC3D: 85.9% | 6.36% |
| V-JEPA + STGCN++ | 57.2% | V-JEPA: 81.8%, STGCN++: 85.7% | 5.66% |

**Why fusion OOF accuracy is lower:**
1. **OOF training on val windows** — MLP only sees validation distribution
2. **Limited training data** — ~4000 windows per fold for inner CV
3. **Missing modality handling** — ~27% windows have no skeleton predictions

**Why V-JEPA + PoseC3D Balanced achieves best TAL:**
1. **Diverse error patterns** — Balanced PoseC3D predicts more RMMs (less BG bias)
2. **Complementary modalities** — RGB (V-JEPA) + Pose (PoseC3D) capture different cues
3. **Fusion learns to combine** — MLP exploits disagreements for better localization

**Despite lower window accuracy, fusion achieves best TAL mAP** — suggesting the MLP learns complementary information that helps with temporal localization.

---

## Recommendations

### For Window-Level Classification
- Use **PoseC3D (CE)** for highest accuracy (85.9%)
- Use **STGCN++ CE** for comparable accuracy with lighter inference

### For Rare Class Detection
- **V-JEPA Balanced is the new best** for rocking (6.67% F1) with minimal accuracy penalty (-0.8%)
- **STGCN++ CE Balanced** achieves 4.22% rocking F1 but with large accuracy penalty (-8.3%)
- **PoseC3D Balanced** achieves 2.76% rocking F1 with moderate accuracy penalty (-4.8%)
- **V-JEPA handles balanced sampling best** — other architectures should consider similar strategies

### For Two-Stage Detection
- Use **V-JEPA Binary (Balanced)** as first-stage detector (68% RMM recall)
- Follow with fine-grained classifier to reduce false positives

### For Temporal Action Localization
- Use **V-JEPA + PoseC3D Balanced (MLP) fusion** — **best TAL performance (8.40% mAP)** 🏆
- **Try V-JEPA Balanced + PoseC3D Balanced** — both models optimized for rare classes
- Fusion captures complementary information between RGB and pose modalities

---

## Files and Resources

### Classification Results
- `v-jepa/runs/vjepa2_tal_cv_5class_bgsub/fold_{N}/per_class_window.csv` — V-JEPA per-class metrics
- `v-jepa/runs/vjepa2_tal_cv_5class_balanced/cv_summary.json` — V-JEPA Balanced CV summary
- `v-jepa/runs/vjepa2_tal_cv_binary_balanced/cv_summary.json` — V-JEPA Binary Balanced CV summary
- `pyskl/work_dirs/posec3d/tal/cv_4class_5class_bgsub/fold{N}/eval_val/metrics.json` — PoseC3D CE
- `pyskl/work_dirs/posec3d/tal/cv_4class_5class_focal/cv_summary.json` — PoseC3D Focal CV summary
- `pyskl/work_dirs/posec3d/tal/cv_4class_5class_ce_balanced/cv_summary.json` — PoseC3D CE Balanced CV summary
- `pyskl/work_dirs/stgcnpp/tal/cv_4class_5class_bgsub/4stream/cv_summary_partial.json` — STGCN++ CE
- `pyskl/work_dirs/stgcnpp/tal/cv_4class_5class_focal/4stream/cv_summary.json` — STGCN++ Focal
- `pyskl/work_dirs/stgcnpp/tal/cv_4class_5class_ce_balanced/fusion_summary.json` — STGCN++ CE Balanced 4-stream fusion

### Fusion Results
- `tal/eval_results/vjepa_posec3d_mlp_logp/fold{N}/fusion_stats.json` — V-JEPA + PoseC3D fusion stats
- `tal/eval_results/vjepa_stgcnpp_mlp_logp/fold{N}/fusion_stats.json` — V-JEPA + STGCN++ fusion stats
- `tal/eval_results/vjepa_posec3d_bal_mlp_logp/fold{N}/fusion_stats.json` — V-JEPA + PoseC3D Balanced fusion stats

### Related Documents
- `TAL_MODEL_COMPARISON.md` — TAL (segment-level) evaluation with mAP@tIoU metrics

