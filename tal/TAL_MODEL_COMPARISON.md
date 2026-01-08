# Temporal Action Localization (TAL) Model Comparison

**Date:** January 6, 2026 (Updated)  
**Task:** 4-class RMM detection (hands flapping, jumping, rocking, spinning)  
**Evaluation Metric:** mAP @ tIoU thresholds {0.3, 0.5, 0.7} for window-based models; ActionFormer (OpenTAD) reports {0.3, 0.4, 0.5, 0.6, 0.7}

---

## Overview

This document summarizes the performance of different models on the Temporal Action Localization (TAL) task for detecting Restricted and Repetitive Motor Movements (RMMs) in home videos. The TAL task requires both correct classification AND accurate temporal localization of action segments.

### Models Evaluated

| Model | Type | Input | Description |
|-------|------|-------|-------------|
| **V-JEPA** | Video Foundation Model | RGB frames | Pre-trained video encoder with linear probe |
| **V-JEPA (Balanced)** | Video Foundation Model | RGB frames | Pre-trained encoder with class-weighted sampling |
| **V-JEPA Binary** | Video Foundation Model | RGB frames | Binary (RMM vs BG) with balanced sampling |
| **PoseC3D (CE)** | 3D CNN on Pose | Pose heatmaps | SlowOnly R50 with Cross-Entropy loss |
| **PoseC3D (Focal)** | 3D CNN on Pose | Pose heatmaps | SlowOnly R50 with Focal loss (gamma=2) |
| **STGCN++ 4-Stream (CE)** | Graph Neural Network | Skeleton graphs | Joint + Bone + Motion streams, fused, CE loss |
| **STGCN++ 4-Stream (Focal)** | Graph Neural Network | Skeleton graphs | Joint + Bone + Motion streams, Focal loss (gamma=1) |
| **V-JEPA + PoseC3D (MLP)** | Late Fusion | RGB + Pose | MLP on log-prob features, OOF trained |
| **V-JEPA + STGCN++ (MLP)** | Late Fusion | RGB + Skeleton | MLP on log-prob features, OOF trained |
| **V-JEPA Bal + PoseC3D (MLP)** | Late Fusion | RGB + Pose | MLP fusion with balanced V-JEPA + PoseC3D CE |
| **ActionFormer + V-JEPA (Balanced)** | End-to-end TAL | V-JEPA2 features | ActionFormer trained on V-JEPA2 features (4-class) |
| **ActionFormer + V-JEPA (Binary)** | End-to-end TAL | V-JEPA2 features | ActionFormer trained on V-JEPA2 features (RMM vs BG) |

### Evaluation Setup

- **Cross-validation:** 3 folds (some models only have 2 folds available)
- **Window size:** 2 seconds with 1 second stride (50% overlap)
- **tIoU thresholds:** 0.3 (loose), 0.5 (standard), 0.7 (strict); ActionFormer additionally reports 0.4 and 0.6
- **Classes:** 4 RMM types (background class excluded from mAP)

---

## Results with Default Postprocessing

Default parameters: `threshold=0.5, smooth_k=3, merge_gap=1.0s`

| Model | Folds | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|-------|-------|---------|---------|---------|---------|
| V-JEPA | 2 | 6.84% ± 0.82% | 4.84% ± 0.79% | 2.24% ± 0.49% | **4.64%** ± 0.70% |
| PoseC3D (CE) | 2 | 5.46% ± 0.00% | 4.55% ± 0.20% | 2.12% ± 0.46% | 4.04% ± 0.22% |
| PoseC3D (Focal) | 3 | 4.98% ± 0.76% | 3.67% ± 0.64% | 1.40% ± 0.35% | 3.35% ± 0.57% |
| STGCN++ 4-Stream (CE) | 2 | 3.53% ± 0.32% | 2.19% ± 0.29% | 0.92% ± 0.18% | 2.21% ± 0.14% |
| STGCN++ 4-Stream (Focal) | 3 | 0.00%* | 0.00%* | 0.00%* | 0.00%* |

*STGCN++ Focal scored 0% with default threshold=0.5 because fused probabilities (5 classes) never exceed ~0.3.

**Initial ranking:** V-JEPA > PoseC3D (CE) > PoseC3D (Focal) > STGCN++ CE

---

## Postprocessing Grid Search

A grid search over postprocessing parameters revealed significant room for improvement:

### Parameters Searched

| Parameter | Values | Description |
|-----------|--------|-------------|
| `threshold` | 0.3, 0.4, 0.5, 0.6 | Confidence threshold for detection |
| `smooth_k` | 1, 3, 5, 7 | Moving average window (1 = no smoothing) |
| `merge_gap_sec` | 0.5, 1.0, 1.5, 2.0 | Max gap to merge adjacent segments |

**Total combinations:** 64 per model

### Optimal Parameters Found

| Model | threshold | smooth_k | merge_gap | Notes |
|-------|-----------|----------|-----------|-------|
| PoseC3D (CE) | **0.3** | **1** | **0.5** | Lower threshold, no smoothing |
| PoseC3D (Focal) | **0.3** | **1** | **0.5** | Same as CE |
| STGCN++ 4-Stream | **0.3** | **1** | **0.5** | Same pattern |
| V-JEPA | **0.4** | **3** | **0.5** | Needs smoothing (noisier) |

### Key Findings

1. **Lower threshold (0.3) improves recall** — Default 0.5 was too conservative
2. **No smoothing (smooth_k=1) is best for pose models** — Smoothing blurs precise predictions
3. **V-JEPA benefits from smoothing** — Its predictions are noisier than pose-based models
4. **Smaller merge gap (0.5s) prevents over-merging** — Default 1.0s merged distinct events

---

## Results with Optimized Postprocessing

Parameters: `threshold=0.3, smooth_k=1, merge_gap=0.5s` (except V-JEPA: `smooth_k=3, thr=0.4`)

| Model | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP | Improvement |
|-------|---------|---------|---------|---------|-------------|
| **PoseC3D (CE)** | 7.44% | 5.83% | 2.26% | **5.18%** | +28.1% |
| V-JEPA | 7.31% | 5.41% | 2.48% | 5.07% | +9.2% |
| PoseC3D (Focal) | 7.14% | 5.24% | 1.69% | 4.69% | +40.0% |
| STGCN++ 4-Stream (CE) | 6.30% | 4.27% | 1.28% | 3.95% | +78.6% |
| STGCN++ 4-Stream (Focal) | 2.29% ± 0.99% | 1.88% ± 0.70% | 0.80% ± 0.44% | **1.65%** ± 0.70% | N/A |

**Single-model ranking:** PoseC3D (CE) > V-JEPA > PoseC3D (Focal) > STGCN++ CE > STGCN++ Focal

### Improvement Summary

| Model | Default avg_mAP | Optimized avg_mAP | Relative Gain |
|-------|-----------------|-------------------|---------------|
| PoseC3D (CE) | 4.04% | 5.18% | **+28.1%** |
| PoseC3D (Focal) | 3.35% | 4.69% | **+40.0%** |
| STGCN++ 4-Stream (CE) | 2.21% | 3.95% | **+78.6%** |
| V-JEPA | 4.64% | 5.07% | +9.2% |
| STGCN++ 4-Stream (Focal) | 0.00%* | 1.65% | N/A |

---

## Late Fusion Results (V-JEPA + Skeleton Models)

Late fusion combines V-JEPA (RGB) with skeleton-based models via a lightweight MLP trained on log-probability features using out-of-fold (OOF) cross-validation.

### Fusion Approach

- **Feature space:** Log-probabilities (Product-of-Experts style)
- **Architecture:** MLP with 16 hidden units on concatenated log-probs + missing flags
- **Training:** 5-fold grouped inner-CV on val windows (grouped by video to prevent leakage)
- **Missing modality:** Filled with uniform distribution + indicator flag

### Fusion Results (Optimized Postprocessing)

Best parameters: `threshold=0.4, smooth_k=1, merge_gap=1.0s`

| Model | Folds | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|-------|-------|---------|---------|---------|---------|
| **V-JEPA + PoseC3D (MLP)** | 2 | 9.95% ± 2.26% | 6.93% ± 2.44% | 2.22% ± 0.66% | **6.36% ± 1.79%** |
| **V-JEPA + STGCN++ (MLP)** | 2 | 8.42% ± 0.49% | 5.88% ± 0.38% | 2.67% ± 0.14% | **5.66% ± 0.34%** |

### Fusion vs Single Models

| Model | avg_mAP | vs Best Single Model |
|-------|---------|----------------------|
| **V-JEPA + PoseC3D (MLP)** | **6.36%** | **+22.8%** vs PoseC3D (5.18%) |
| V-JEPA + STGCN++ (MLP) | 5.66% | +9.3% vs PoseC3D (5.18%) |
| PoseC3D (CE) | 5.18% | baseline |
| V-JEPA | 5.07% | -2.1% |

**Key findings:**
- **Among window-based approaches, V-JEPA + PoseC3D fusion is best** (6.36% avg_mAP)
- Fusion provides **~23% relative improvement** over the best single model
- The complementary nature of RGB (appearance) and pose (motion) features benefits TAL
- V-JEPA + STGCN++ fusion underperforms V-JEPA + PoseC3D, likely due to STGCN++'s weaker skeleton predictions

### Why Fusion Helps

1. **Complementary modalities:** V-JEPA captures appearance context; PoseC3D captures precise body movements
2. **Error diversity:** Models make different mistakes, fusion averages them out
3. **Missing modality handling:** MLP learns to rely on V-JEPA when skeleton predictions are unavailable

### Fusion Limitations

1. **OOF training on val windows:** Ideally would train on train windows, but train-window preds weren't available
2. **Missing skeleton predictions:** ~27% of windows in fold 0 had no skeleton predictions
3. **Higher variance:** Fold 0 (8.15% avg_mAP) vs Fold 1 (4.58% avg_mAP) shows instability

**Window-based ranking (including fusion):**

| Rank | Model | avg_mAP |
|------|-------|---------|
| 1 | **V-JEPA + PoseC3D (MLP)** | **6.36%** |
| 2 | V-JEPA + STGCN++ (MLP) | 5.66% |
| 3 | PoseC3D (CE) | 5.18% |
| 4 | V-JEPA | 5.07% |
| 5 | PoseC3D (Focal) | 4.69% |
| 6 | STGCN++ 4-Stream (CE) | 3.95% |
| 7 | STGCN++ 4-Stream (Focal) | 1.65% |

**Note:** End-to-end ActionFormer results are substantially higher than all window-based and fusion baselines; see “ActionFormer Results (January 2026)”.

---

## Balanced Training Results (January 2026)

Training with class-balanced sampling improves performance on rare classes while maintaining overall quality.

### V-JEPA Balanced (4-class, 3-fold CV)

Trained with class probability sampling to address class imbalance:
- Background: downsampled to 10%
- Rare classes (rocking, spinning): upsampled to match jumping

| Metric | Value |
|--------|-------|
| **mAP@0.3** | 9.48% ± 1.60% |
| **mAP@0.5** | 6.37% ± 0.96% |
| **mAP@0.7** | 1.99% ± 0.80% |
| **avg_mAP** | **5.95% ± 1.10%** |
| **Recall@0.3** | 15.73% ± 1.19% |
| **Recall@0.5** | 12.45% ± 0.93% |
| **avg_Recall** | 11.30% ± 0.88% |

**Improvement over baseline V-JEPA:** +17.4% (5.95% vs 5.07%)

### V-JEPA Binary (RMM vs BG, 3-fold CV)

Simplified binary detection task (RMM vs Background):

| Metric | Value |
|--------|-------|
| **mAP@0.3** | 11.78% ± 0.63% |
| **mAP@0.5** | 6.77% ± 1.12% |
| **mAP@0.7** | 1.44% ± 0.33% |
| **avg_mAP** | **6.66% ± 0.69%** |
| **Recall@0.3** | 19.57% ± 0.12% |
| **Recall@0.5** | 14.68% ± 0.57% |
| **avg_Recall** | 13.57% ± 0.30% |
| **Precision@0.3** | 39.78% ± 4.46% |
| **Precision@0.5** | 29.98% ± 4.56% |

**Key insight:** Binary model achieves highest mAP (6.66%) and recall (19.6% @ tIoU=0.3), making it ideal for broad RMM localization where specific type classification isn't needed.

### V-JEPA Balanced + PoseC3D CE Fusion (2-fold CV)

Late fusion combining balanced V-JEPA with PoseC3D:

| Metric | Value |
|--------|-------|
| **mAP@0.3** | 9.76% ± 2.74% |
| **mAP@0.5** | 6.35% ± 1.64% |
| **mAP@0.7** | 1.85% ± 0.69% |
| **avg_mAP** | **5.99% ± 1.69%** |

**Note:** Only 2 folds available (limited by PoseC3D CE). Performance comparable to V-JEPA Balanced alone, suggesting fusion benefits diminish with balanced training.

### Balanced vs Non-Balanced Comparison

| Model | avg_mAP | avg_Recall | Notes |
|-------|---------|------------|-------|
| **V-JEPA Binary** | **6.66%** | **13.57%** | Best overall, binary task |
| V-JEPA + PoseC3D (MLP) | 6.36% | N/A | Previous best fusion |
| V-JEPA Bal + PoseC3D (MLP) | 5.99% | N/A | Balanced fusion |
| **V-JEPA Balanced** | **5.95%** | **11.30%** | Best 4-class single model |
| V-JEPA + STGCN++ (MLP) | 5.66% | N/A | STGCN++ fusion |
| PoseC3D (CE) | 5.18% | N/A | Best pose model |
| V-JEPA (baseline) | 5.07% | N/A | Original V-JEPA |

**Key Findings:**
1. **V-JEPA Binary achieves best window-based TAL mAP** (6.66%) — simpler task helps
2. **Balanced training improves V-JEPA** (+17% over baseline)
3. **Binary model has highest window-based recall** (19.6% @ tIoU=0.3) — catches more RMM events
4. **Fusion gains diminish** with balanced training (5.99% vs 6.36%)

---

## Per-Class Analysis (at tIoU=0.5)

Performance varies significantly by action class:

| Class | PoseC3D (CE) | V-JEPA | Notes |
|-------|--------------|--------|-------|
| **jumping** | 7.7% | 9.6% | Best detected (distinctive motion) |
| **hands flapping** | 7.3% | 7.8% | Second best (frequent in dataset) |
| **spinning** | 4.0% | 3.7% | Moderate (full-body rotation) |
| **rocking** | 0.0% | 1.4% | Hardest (subtle, short duration) |

### Class-Specific Observations

- **Jumping:** Easiest to detect due to clear vertical displacement
- **Hands flapping:** Well-represented in training data, distinctive arm motion
- **Spinning:** Recognizable but often confused with other movements
- **Rocking:** Very challenging — subtle, low amplitude, often short duration

---

## Model Comparison Insights

### Why PoseC3D (CE) wins with optimization

1. **Precise predictions:** Pose-based features provide sharp temporal boundaries
2. **Benefits from lower threshold:** Has high precision, can afford more detections
3. **No smoothing needed:** Predictions are already temporally coherent

### Why V-JEPA was initially best

1. **Rich visual features:** Captures appearance + motion from RGB
2. **Pre-training advantage:** Learned general video representations
3. **Robust to pose estimation errors:** Not dependent on skeleton accuracy

### Why STGCN++ underperforms

1. **Information loss:** Skeleton graphs discard spatial appearance
2. **Pose estimation errors:** Errors in HRNet propagate to classification
3. **Limited receptive field:** Graph convolutions may miss global context

### Why STGCN++ Focal performed worst (1.65% avg_mAP)

The STGCN++ Focal model suffered from **premature early stopping** during training:

| Modality | Avg Accuracy | Issue |
|----------|--------------|-------|
| **b (bone)** | ~49% | Best - trained fully |
| **jm (joint motion)** | ~42% | Good for folds 0,1; fold 2 crashed early |
| **j (joint)** | ~13% | Early stopped on 2/3 folds |
| **bm (bone motion)** | ~3% | All folds early stopped (near-random) |

The 4-stream weighted fusion (`[j=2, b=2, jm=1, bm=1]`) averaged in essentially random predictions from undertrained modalities, catastrophically harming performance.

**Root causes:**
1. **Early stopping patience=3 too aggressive** for modalities that train slowly
2. **Focal loss gamma=1.0** (reduced from 2.0) may not have been sufficient
3. **Transfer learning mismatch:** NTU60 pretrained weights may not transfer well for all modalities

### CE vs Focal Loss

| Architecture | CE Loss | Focal Loss | Winner |
|--------------|---------|------------|--------|
| PoseC3D | 5.18% | 4.69% | **CE (+10%)** |
| STGCN++ 4-Stream | 3.95% | 1.65% | **CE (+139%)** |

**Key findings:**
- **CE consistently outperforms Focal** for TAL task
- **Focal loss hypothesis didn't hold:** Despite class imbalance, CE performed better
- **STGCN++ Focal was catastrophic:** Early stopping + weak modalities + fusion = disaster
- **Possible reason:** Focal loss down-weights easy examples too aggressively, reducing learning signal for the dominant background class which is important for TAL precision

---

## Recommendations

### For Best Localization (Broad RMM Detection)
Use **ActionFormer + V-JEPA Binary** (best checkpoints; OpenTAD eval):
- **avg_mAP:** 27.93% ± 4.70%
- **mAP@0.3:** 40.83% ± 5.60%

If you need a lightweight window-based baseline, use **V-JEPA Binary** with optimized postprocessing:
```
threshold=0.6, smooth_k=1, merge_gap_sec=0.5
```

### For 4-Class Detection
Use **ActionFormer + V-JEPA Balanced** (best checkpoints; OpenTAD eval):
- **avg_mAP:** 16.70% ± 1.53%
- **mAP@0.3:** 23.97% ± 1.07%

If you need a window-based baseline, use **V-JEPA Balanced** with optimized postprocessing:
```
threshold=0.4, smooth_k=3, merge_gap_sec=0.5
```

### For Legacy Comparison
Use **V-JEPA + PoseC3D (MLP) fusion** with optimized postprocessing:
```
threshold=0.4, smooth_k=1, merge_gap_sec=1.0
```
- Best window-based fusion baseline (6.36%), but requires pose predictions

### Future Improvements
1. **ActionFormer optimization** — tune NMS / score thresholds, boundary refinement, and checkpoint selection
2. **Train fusion on train windows** — Current OOF approach on val windows is a limitation
3. **Improve skeleton coverage** — 27% missing skeleton preds hurts fusion
4. **Class-specific thresholds** — Lower threshold for rare classes (rocking, spinning)
5. **Two-stage detection** — Binary detector → 4-class classifier

---

## ActionFormer Results (January 2026)

End-to-end temporal action localization using ActionFormer with V-JEPA2 features.

### Setup

- **Model:** ActionFormer (transformer-based TAL)
- **Features:** V-JEPA2 1024-dim features extracted at 16-frame snippets
- **Training:** 50 epochs, AdamW optimizer, Cosine LR schedule
- **Evaluation:** mAP @ tIoU {0.3, 0.4, 0.5, 0.6, 0.7}
- **Reporting:** Metrics below are from OpenTAD evaluation of `best.pth` (epoch 20; EMA) on each fold's validation split

### ActionFormer + V-JEPA Balanced (4-class, 3-fold CV)

| Fold | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|---------|---------|
| 0 | 22.80% | 20.74% | 18.82% | 13.96% | 6.43% | 16.55% |
| 1 | 24.89% | 22.81% | 19.66% | 16.36% | 7.76% | 18.29% |
| 2 | 24.22% | 21.50% | 14.24% | 9.37% | 6.93% | 15.25% |
| **Mean** | **23.97% ± 1.07%** | **21.68% ± 1.05%** | **17.57% ± 2.92%** | **13.23% ± 3.55%** | **7.04% ± 0.67%** | **16.70% ± 1.53%** |

### ActionFormer + V-JEPA Binary (RMM vs BG, 3-fold CV)

| Fold | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|---------|---------|
| 0 | 40.98% | 35.21% | 30.65% | 24.92% | 12.90% | 28.93% |
| 1 | 46.35% | 41.02% | 33.56% | 25.67% | 13.59% | 32.04% |
| 2 | 35.16% | 30.81% | 21.43% | 17.24% | 9.41% | 22.81% |
| **Mean** | **40.83% ± 5.60%** | **35.68% ± 5.12%** | **28.55% ± 6.33%** | **22.61% ± 4.67%** | **11.97% ± 2.24%** | **27.93% ± 4.70%** |

### ActionFormer Top-k Recall (proposal-based, mean ± std)

These recalls are computed as top-kx recall (R@k) at each tIoU threshold during OpenTAD evaluation.

#### ActionFormer + V-JEPA Binary

| tIoU | R@1 | R@5 | R@10 | R@100 |
|------|-----|-----|------|-------|
| 0.3 | 59.48% ± 5.56% | 90.48% ± 3.06% | 94.74% ± 1.61% | 98.37% ± 1.42% |
| 0.5 | 44.99% ± 5.38% | 73.23% ± 4.24% | 81.09% ± 2.70% | 90.90% ± 4.87% |
| 0.7 | 25.65% ± 2.09% | 45.54% ± 2.16% | 53.55% ± 4.34% | 72.16% ± 3.00% |

#### ActionFormer + V-JEPA Balanced (4-class)

| tIoU | R@1 | R@5 | R@10 | R@100 |
|------|-----|-----|------|-------|
| 0.3 | 59.05% ± 6.12% | 88.88% ± 0.62% | 93.77% ± 1.89% | 97.38% ± 1.87% |
| 0.5 | 42.27% ± 0.45% | 74.96% ± 1.73% | 82.71% ± 0.52% | 90.68% ± 4.10% |
| 0.7 | 26.30% ± 1.45% | 44.67% ± 3.26% | 51.88% ± 5.35% | 66.02% ± 4.25% |

### ActionFormer vs Window-Based TAL Comparison

| Model | Type | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|-------|------|---------|---------|---------|---------|
| **ActionFormer + V-JEPA Binary** | End-to-end | 40.83% | 28.55% | 11.97% | **27.93%** |
| **ActionFormer + V-JEPA Balanced** | End-to-end | 23.97% | 17.57% | 7.04% | **16.70%** |
| V-JEPA Binary (window) | Window-based | 11.78% | 6.77% | 1.44% | 6.66% |
| V-JEPA + PoseC3D (MLP) | Fusion | 9.95% | 6.93% | 2.22% | 6.36% |
| V-JEPA Balanced (window) | Window-based | 9.48% | 6.37% | 1.99% | 5.95% |

**Key Findings:**
1. **ActionFormer is decisively better** than window-based TAL (+ postprocessing)
2. **Binary detection reaches 40.8% mAP@0.3 (best checkpoints)** — strong for broad RMM localization
3. **End-to-end TAL significantly outperforms** window-based + postprocessing
4. **V-JEPA features transfer excellently** to ActionFormer architecture

### Why ActionFormer Works Better

1. **Native temporal modeling:** Transformer attention learns temporal context
2. **Direct segment prediction:** Predicts boundaries directly, not from window classifications
3. **Multi-scale features:** FPN enables detection at multiple temporal scales
4. **End-to-end training:** Jointly optimizes localization and classification

---

## Files and Resources

### Evaluation Results
- `eval_results/grid_search_results.csv` — All parameter combinations for all models
- `eval_results/best_params_per_model.json` — Optimal settings per model
- `eval_results/fusion_summary.json` — Fusion training summary
- `eval_results/{model}/fold{N}/` — Per-fold detailed results
- `eval_results/{model}/cv_summary.json` — CV summary with mean ± std

### Fusion Results
- `eval_results/vjepa_posec3d_mlp_logp/` — V-JEPA + PoseC3D fusion
- `eval_results/vjepa_stgcnpp_mlp_logp/` — V-JEPA + STGCN++ fusion

### Scripts
- `scripts/run_tal_fusion_oof.py` — Generate fused window predictions
- `scripts/grid_search_postprocessing.py` — Parameter grid search
- `scripts/eval_best_postprocess.py` — Evaluate with best params + CV summary
- `scripts/eval_tal_from_existing_preds.sh` — Run TAL evaluation
- `convert_preds_to_tal_format.py` — Convert predictions to TAL format
- `eval_tal_from_window_preds.py` — Core TAL evaluation
- `tal_fusion.py` — Late fusion module (MLP on log-probs)

### Tests
- `test_tal_eval.py` — TAL evaluation pipeline tests
- `test_tal_fusion.py` — Fusion module tests

### Model Checkpoints
- PoseC3D (CE): `pyskl/work_dirs/posec3d/tal/cv_4class_5class_bgsub/`
- PoseC3D (Focal): `pyskl/work_dirs/posec3d/tal/cv_4class_5class_focal/`
- STGCN++ (CE): `pyskl/work_dirs/stgcnpp/tal/cv_4class_5class_bgsub/`
- STGCN++ (Focal): `pyskl/work_dirs/stgcnpp/tal/cv_4class_5class_focal/`
- V-JEPA: `v-jepa/runs/vjepa2_tal_cv_5class_bgsub/`
- V-JEPA (Balanced): `v-jepa/runs/vjepa2_tal_cv_5class_balanced/`
- V-JEPA Binary: `v-jepa/runs/vjepa2_tal_cv_binary_balanced/`
- ActionFormer (Balanced): `OpenTAD/exps/sails_rmm/actionformer_vjepa_balanced_fold{0,1,2}/`
- ActionFormer (Binary): `OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold{0,1,2}/`

### Balanced Model Results
- `eval_results/vjepa_balanced/cv_summary.json` — V-JEPA Balanced 3-fold CV
- `eval_results/vjepa_binary/cv_summary.json` — V-JEPA Binary 3-fold CV
- `eval_results/vjepa_balanced_posec3d_mlp_logp/cv_summary.json` — Balanced fusion



