# Results

## Evaluation

- **Cross-validation**: Results are reported as **mean ± std** across CV folds, using **Leave-Child-Timepoint-Out (LCTO)** grouping.
- **Window-based detection/TAL**:
    - Windows are 2s with 1s stride (50% overlap).
    - Window-level models output per-window scores which can be postprocessed into segments for TAL evaluation.
- **Segment-level TAL evaluation**:
    - Multi-class TAL reports **mAP@tIoU** at multiple thresholds (and recall metrics when available).
    - Proposal-based transformer TAL additionally reports **top-k recall (R@k)** at multiple tIoU thresholds.
- **Partial-CV note**: Some window/detection runs only have **2 folds** available; these are labeled as **partial CV**.

## Action Recognition

### Clip-level classification (RMM type)

**Goal:** Given a short clip containing an annotated RMM segment, predict the RMM type.

**Label variants:**

- **4-class:** hands flapping (incl. one-hand flap merged), jumping, rocking, spinning
- **5-class:** adds one hand flap as a distinct class

**Evaluation setup:** 3-fold cross-validation with **Leave-Child-Timepoint-Out (LCTO)** grouping to avoid leakage across subject/timepoint.

**Metrics reported:**

- Top-1 / Top-2 Accuracy
- Macro F1
- Macro Precision
- Macro Recall
- Cohen's κ (agreement beyond chance)

### 4-class results (clip-level; 3-fold CV)

**Main table (best-per-family + all fusion results)**

| Model family | Method | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Precision | Macro Recall | Cohen's κ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fusion (RGB + Pose + Skeleton) | **3-Way MLP (VJEPA-2 + PoseC3D + STGCN++)** | **84.7% ± 1.9%** | **96.2% ± 1.9%** | **82.1% ± 3.9%** | **83.2% ± 4.1%** | **83.3% ± 3.5%** | **0.759 ± 0.02** |
| Fusion (RGB + Skeleton) | MLP Fusion (VJEPA-2 + STGCN++) | 83.8% ± 2.2% | **96.5% ± 0.4%** | **82.2% ± 2.5%** | **83.8% ± 1.5%** | **83.6% ± 2.5%** | 0.745 ± 0.03 |
| Fusion (RGB + Pose) | MLP Fusion (VJEPA-2 + PoseC3D) | 80.3% ± 3.3% | 95.1% ± 2.4% | 78.9% ± 4.4% | 80.3% ± 4.1% | 80.6% ± 4.1% | 0.690 ± 0.04 |
| Skeleton | STGCN++ | 77.9% ± 2.2% | 95.1% ± 0.7% | 72.7% ± 4.5% | 77.6% ± 0.7% | 71.0% ± 6.5% | 0.635 ± 0.03 |
| RGB | VJEPA-2 | 76.0% ± 3.0% | 93.7% ± 1.3% | 75.1% ± 1.8% | 78.4% ± 0.7% | 73.0% ± 3.3% | 0.657 ± 0.09 |
| Pose | PoseC3D (non-weighted) | 74.6% ± 3.0% | 91.3% ± 0.4% | 70.4% ± 7.3% | 75.3% ± 6.2% | 68.4% ± 8.0% | 0.580 ± 0.05 |
| VLM | Qwen2.5-VL (zero-shot) | 56.8% ± 3.4% | 78.9% ± 2.2% | 28.2% ± 0.6% | 51.2% ± 10.0% | 31.2% ± 0.7% | 0.168 ± 0.02 |

**Single-model variants (loss / weighting / preprocessing)**

- **VJEPA-2 variants (4-class; clip-level)**

| Variant | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Precision | Macro Recall | Cohen's κ |
| --- | --- | --- | --- | --- | --- | --- |
| Person-centered RGB preprocessing (SAM3) | 76.0% ± 3.0% | 93.7% ± 1.3% | 75.1% ± 1.8% | 78.4% ± 0.7% | 73.0% ± 3.3% | 0.657 ± 0.09 |
- **PoseC3D variants (4-class; clip-level)**

| Variant | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Precision | Macro Recall | Cohen's κ |
| --- | --- | --- | --- | --- | --- | --- |
| Non-weighted | 74.6% ± 3.0% | 91.3% ± 0.4% | 70.4% ± 7.3% | 75.3% ± 6.2% | 68.4% ± 8.0% | 0.580 ± 0.05 |
- **STGCN++ variants (4-class; clip-level)**

| Variant | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Precision | Macro Recall | Cohen's κ |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 77.9% ± 2.2% | 95.1% ± 0.7% | 72.7% ± 4.5% | 77.6% ± 0.7% | 71.0% ± 6.5% | 0.635 ± 0.03 |

**Per-class precision/recall (4-class; clip-level; best model per family)**

| Model family | Model | hands flapping P/R | jumping P/R | rocking P/R | spinning P/R |
| --- | --- | --- | --- | --- | --- |
| Fusion (RGB + Pose + Skeleton) | 3-Way MLP | **92.3% / 87.0%** | 74.7% / **83.6%** | 76.7% / **76.7%** | 86.7% / **86.7%** |
| RGB | VJEPA-2 | 80.5% / 84.0% | 64.8% / 68.4% | **80.0%** / 61.3% | **88.3%** / 78.3% |
| Pose | PoseC3D (non-weighted) | 77.4% / **86.7%** | 69.3% / 65.2% | 64.8% / 47.1% | **89.6%** / 74.4% |
| Skeleton | STGCN++ | 81.6% / 85.7% | **74.8%** / 62.9% | 74.5% / 56.0% | 79.7% / 79.4% |
| VLM | Qwen2.5-VL (zero-shot) | 58.6% / **93.9%** | 46.1% / 27.6% | **100.0%** / 3.1% | 0.0% / 0.0% |

### 5-class results (clip-level; 3-fold CV)

**Main table (best-per-family + all fusion results)**

| Model family | Method | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Precision | Macro Recall | Cohen's κ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fusion (RGB + Pose + Skeleton) | **3-Way MLP (VJEPA-2 + PoseC3D + STGCN++)** | **69.6% ± 3.0%** | **88.5% ± 0.4%** | **69.8% ± 1.8%** | **71.7% ± 2.6%** | **71.0% ± 1.5%** | **0.594 ± 0.04** |
| Fusion (RGB + Skeleton) | MLP Fusion (VJEPA-2 + STGCN++) | 68.2% ± 1.5% | 88.1% ± 2.1% | 68.5% ± 1.9% | 70.0% ± 2.2% | 70.3% ± 1.7% | 0.578 ± 0.02 |
| Fusion (RGB + Pose) | MLP Fusion (VJEPA-2 + PoseC3D) | 67.6% ± 3.6% | 87.5% ± 1.1% | 67.8% ± 1.0% | 69.2% ± 1.8% | 69.2% ± 2.0% | 0.569 ± 0.04 |
| Pose | PoseC3D (weighted) | 67.0% ± 1.6% | 85.5% ± 1.3% | 65.9% ± 2.8% | 67.8% ± 3.1% | 65.4% ± 3.0% | 0.549 ± 0.02 |
| Skeleton | STGCN++ | 66.5% ± 1.5% | 83.3% ± 1.0% | 64.6% ± 1.4% | 67.4% ± 3.7% | 65.0% ± 3.3% | 0.541 ± 0.02 |
| RGB | VJEPA-2 | 62.5% ± 2.0% | 84.8% ± 2.9% | 63.0% ± 1.3% | 64.4% ± 2.1% | 61.5% ± 0.8% | 0.499 ± 0.05 |
| VLM | Qwen2.5-VL (zero-shot) | 37.7% ± 6.0% | 63.6% ± 2.2% | 12.2% ± 1.9% | 20.8% ± 6.3% | 20.3% ± 1.0% | 0.008 ± 0.02 |

**Single-model variants (loss / weighting / preprocessing)**

- **VJEPA-2 variants (5-class; clip-level)**

| Variant | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Precision | Macro Recall | Cohen's κ |
| --- | --- | --- | --- | --- | --- | --- |
| Person-centered RGB preprocessing (SAM3) | 62.5% ± 2.0% | 84.8% ± 2.9% | 63.0% ± 1.3% | 64.4% ± 2.1% | 61.5% ± 0.8% | 0.499 ± 0.05 |
- **PoseC3D variants (5-class; clip-level)**

| Variant | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Precision | Macro Recall | Cohen's κ |
| --- | --- | --- | --- | --- | --- | --- |
| Weighted | 67.0% ± 1.6% | 85.5% ± 1.3% | 65.9% ± 2.8% | 67.8% ± 3.1% | 65.4% ± 3.0% | 0.549 ± 0.02 |
| Weighted (sqrt) | 65.3% ± 2.2% | 83.5% ± 1.1% | 63.9% ± 3.2% | 66.1% ± 3.8% | 63.6% ± 3.6% | 0.526 ± 0.03 |
| Focal loss | 63.9% ± 0.7% | 83.0% ± 1.4% | 61.8% ± 3.5% | 65.2% ± 1.8% | 60.6% ± 4.5% | 0.502 ± 0.02 |
- **STGCN++ variants (5-class; clip-level)**

| Variant | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Precision | Macro Recall | Cohen's κ |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 66.5% ± 1.5% | 83.3% ± 1.0% | 64.6% ± 1.4% | 67.4% ± 3.7% | 65.0% ± 3.3% | 0.541 ± 0.02 |
| Weighted | 65.8% ± 3.3% | 81.6% ± 1.5% | 63.7% ± 2.2% | 65.0% ± 2.8% | 63.7% ± 3.0% | 0.534 ± 0.04 |
| Focal loss | 63.7% ± 2.8% | 83.5% ± 0.4% | 63.6% ± 2.9% | 67.6% ± 1.9% | 62.6% ± 3.1% | 0.501 ± 0.03 |

**Per-class precision/recall (5-class; clip-level; best model per family)**

| Model family | Model | hands flapping P/R | jumping P/R | one hand flap P/R | rocking P/R | spinning P/R |
| --- | --- | --- | --- | --- | --- | --- |
| Fusion (RGB + Pose + Skeleton) | 3-Way MLP | **76.2%** / 66.8% | 69.4% / **78.6%** | **50.5% / 53.6%** | 68.1% / **68.9%** | 87.0% / **88.9%** |
| Pose | PoseC3D (weighted) | 65.8% / **73.3%** | **70.2%** / 77.1% | 57.3% / 41.1% | 61.2% / 52.5% | 84.3% / 82.9% |
| Skeleton | STGCN++ | 66.5% / 71.7% | 65.9% / 74.1% | 39.2% / 37.3% | **72.8%** / 53.9% | **87.1%** / 84.4% |
| RGB | VJEPA-2 | 62.6% / 68.8% | 55.7% / 69.1% | 45.6% / 29.4% | 72.7% / 53.2% | 85.4% / 86.8% |
| VLM | Qwen2.5-VL (zero-shot) | 37.6% / **98.0%** | 66.7% / 3.5% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

### Video-level classification (when reported)

Video-level metrics aggregate clip predictions per video before computing accuracy/F1/κ.

### 4-class (video-level; 3-fold CV)

**Main table (best-per-family + all fusion results)**

| Model family | Method | Video Top-1 Acc | Video Macro F1 | Cohen's κ |
| --- | --- | --- | --- | --- |
| Fusion (RGB + Pose + Skeleton) | **3-Way MLP (VJEPA-2 + PoseC3D + STGCN++)** | **86.4% ± 2.6%** | **83.1% ± 4.9%** | **0.786 ± 0.04** |
| Fusion (RGB + Skeleton) | MLP Fusion (VJEPA-2 + STGCN++) | 84.2% ± 3.7% | 81.3% ± 4.1% | 0.752 ± 0.06 |
| Fusion (RGB + Pose) | MLP Fusion (VJEPA-2 + PoseC3D) | 82.5% ± 3.1% | 80.5% ± 3.6% | 0.728 ± 0.04 |
| Skeleton | STGCN++ | 82.4% | 75.9% | 0.635 |
| RGB | VJEPA-2 | 78.6% | 76.7% | 0.657 |
| Pose | PoseC3D (non-weighted) | 74.6% | 70.4% | 0.580 |

### 5-class (video-level; 3-fold CV)

**Main table (best-per-family + all fusion results)**

| Model family | Method | Video Top-1 Acc | Video Macro F1 | Cohen's κ |
| --- | --- | --- | --- | --- |
| Fusion (RGB + Pose + Skeleton) | **3-Way MLP (VJEPA-2 + PoseC3D + STGCN++)** | **70.7% ± 3.1%** | **70.8% ± 2.5%** | **0.611 ± 0.04** |
| Fusion (RGB + Skeleton) | MLP Fusion (VJEPA-2 + STGCN++) | 69.0% ± 2.0% | 69.6% ± 2.8% | 0.589 ± 0.03 |
| Fusion (RGB + Pose) | MLP Fusion (VJEPA-2 + PoseC3D) | 66.9% ± 4.1% | 66.9% ± 3.1% | 0.560 ± 0.05 |
| Skeleton | STGCN++ (weighted) | 70.6% ± 2.0% | 68.3% ± 1.2% | 0.534 |
| Pose | PoseC3D (weighted) | 67.0% ± 1.6% | 65.9% ± 2.8% | 0.549 |
| RGB | VJEPA-2 | 62.4% | 62.8% | 0.499 |

**Single-model variants (loss / weighting; 5-class video-level)**

- **PoseC3D variants (5-class; video-level)**

| Variant | Video Top-1 Acc | Video Macro F1 | Cohen's κ |
| --- | --- | --- | --- |
| Weighted | 67.0% ± 1.6% | 65.9% ± 2.8% | 0.549 |
| Weighted (sqrt) | 65.3% ± 2.2% | 63.9% ± 3.2% | 0.526 |
| Focal loss | 63.9% ± 0.7% | 61.8% ± 3.5% | 0.502 |
- **STGCN++ variants (5-class; video-level)**

| Variant | Video Top-1 Acc | Video Macro F1 | Cohen's κ |
| --- | --- | --- | --- |
| Default | 68.8% ± 3.0% | 66.9% ± 4.4% | 0.541 |
| Weighted | 70.6% ± 2.0% | 68.3% ± 1.2% | 0.534 |
| Focal loss | 67.7% ± 1.0% | 67.1% ± 1.5% | 0.501 |

### Window-level detection (includes background)

**Goal:** Given a short fixed-length window sampled from a full video, predict whether it contains an RMM and, if so, which RMM type.

**Labels:** 5-class window classification including background:

- background, hands flapping, jumping, rocking, spinning

**Notes:**

- The window-level detection setting is highly imbalanced (background windows dominate).
- This task produces per-window scores that can be postprocessed into segments for TAL.

**Metrics reported:**

- Top-1 / Top-2 Accuracy
- Macro F1
- Macro Precision
- Macro Recall
- Per-class F1/Precision/Recall (to quantify rare-class behavior)
- RMM Recall (all types collapsed; where available via multi-class → binary collapsing)
- Cohen's κ

### Main results (5-class window detection; best-per-family)

| Model family | Model (variant) | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Precision | Macro Recall | Cohen's κ | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pose | PoseC3D (CE; background-subsampling) | **85.89% ± 0.56%** | 96.16% ± 0.48% | 37.62% ± 3.23% | 37.17% ± 1.68% | 38.87% ± 5.08% | 0.330 ± 0.018 | Partial CV (2 folds) |
| Skeleton | STGCN++ (CE; background-subsampling) | 85.74% ± 0.14% | 95.18% ± 0.11% | 32.85% ± 1.48% | 37.98% ± 6.74% | 32.35% ± 0.82% | 0.295 ± 0.003 | Partial CV (2 folds) |
| RGB | VJEPA-2 (balanced sampling) | 80.94% ± 1.73% | 95.08% ± 0.87% | 36.83% ± 3.45% | 33.42% ± 2.43% | 47.23% ± 4.89% | 0.311 ± 0.025 | 3-fold CV |

### Model variants (sampling / loss strategies)

- **PoseC3D variants (5-class windows; mean ± std)**

| Variant | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Precision | Macro Recall | Cohen's κ | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CE (background-subsampling) | **85.89% ± 0.56%** | 96.16% ± 0.48% | 37.62% ± 3.23% | 37.17% ± 1.68% | 38.87% ± 5.08% | 0.330 ± 0.018 | Partial CV (2 folds) |
| CE (balanced sampling) | 81.11% ± 1.67% | **96.34% ± 0.39%** | 35.82% ± 5.78% | 34.90% ± 5.44% | 40.26% ± 7.48% | 0.282 ± 0.024 | 3-fold CV |
| Focal loss | 84.31% ± 1.33% | 96.07% ± 0.30% | 31.47% ± 1.80% | 31.15% ± 1.74% | 32.62% ± 2.52% | 0.248 ± 0.016 | 3-fold CV |
- **STGCN++ variants (5-class windows; mean ± std)**

| Variant | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Precision | Macro Recall | Cohen's κ | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CE (background-subsampling) | 85.74% ± 0.14% | 95.18% ± 0.11% | 32.85% ± 1.48% | 37.98% ± 6.74% | 32.35% ± 0.82% | 0.295 ± 0.003 | Partial CV (2 folds) |
| Focal loss | 37.13% ± 3.75% | 74.95% ± 8.21% | 19.56% ± 1.18% | 24.65% ± 0.97% | 39.26% ± 1.71% | 0.063 ± 0.006 | 3-fold CV |
- **VJEPA-2 variants (5-class windows; mean ± std)**

| Variant | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Precision | Macro Recall | Cohen's κ | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5-class (balanced sampling) | 80.94% ± 1.73% | 95.08% ± 0.87% | 36.83% ± 3.45% | 33.42% ± 2.43% | 47.23% ± 4.89% | 0.311 ± 0.025 | 3-fold CV |
| Binary (RMM vs BG; balanced sampling) | 78.12% ± 3.11% | 100.0% | 62.15% ± 1.17% | 61.01% ± 0.82% | 73.50% ± 0.61% | 0.275 ± 0.017 | 3-fold CV |

### Collapsed RMM vs background (derived from 5-class window predictions)

| Model | RMM Precision | RMM Recall | RMM F1 |
| --- | --- | --- | --- |
| VJEPA-2 (balanced sampling; derived RMM vs BG) | 30.01% ± 2.90% | **64.22% ± 2.39%** | 40.75% ± 2.38% |
| PoseC3D (CE balanced; derived RMM vs BG) | 31.54% ± 2.06% | 50.98% ± 5.46% | 38.88% ± 2.65% |

### Per-class precision/recall/F1 (selected models; mean ± std)

| Model | hands flapping (P/R/F1) | jumping (P/R/F1) | rocking (P/R/F1) | spinning (P/R/F1) |
| --- | --- | --- | --- | --- |
| PoseC3D (CE; bg-subsampling, partial CV) | 44.49% ± 2.43 / 43.42% ± 5.49 / 43.59% ± 1.64 | 30.70% ± 0.62 / 31.75% ± 0.42 / 31.21% ± 0.12 | 0.0 / 0.0 / 0.0 | 18.83% ± 9.74 / 26.38% ± 21.62 / 21.04% ± 14.79 |
| VJEPA-2 (balanced sampling) | 29.82% ± 3.93 / 67.26% ± 5.85 / 40.90% ± 2.90 | 24.76% ± 3.71 / 53.90% ± 5.33 / 33.88% ± 4.45 | 5.64% ± 3.65 / 8.21% ± 4.71 / 6.67% ± 4.12 | 11.35% ± 8.67 / 22.92% ± 16.15 / 13.36% ± 11.91 |

## Temporal Action Localization

### Segment-level TAL (multi-class localization)

**Goal:** Given a full video, output a set of predicted temporal segments with an associated RMM class label (hands flapping, jumping, rocking, spinning). A correct prediction requires both:

- Correct class, and
- Accurate temporal boundaries (sufficient overlap with ground truth).

**Metrics reported:**

- mAP at temporal IoU (tIoU) thresholds
- Recall@tIoU
- (For proposal-based transformer evaluation) Top-k recall (R@k) at tIoU thresholds

**Notes:**

- Background is not treated as an evaluated class in multi-class TAL mAP (mAP is over RMM classes).

### Main results (best-per-family + all fusion results)

| Model family | Model | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
| --- | --- | --- | --- | --- | --- |
| Transformer Head (RGB) | **ActionFormer on VJEPA-2 features (balanced)** | **23.97% ± 1.07%** | **17.57% ± 2.92%** | **7.04% ± 0.67%** | **16.70% ± 1.53%** |
| Window-based Fusion (RGB + Pose) | **VJEPA-2 + PoseC3D (MLP fusion; optimized postprocess)** | 9.95% ± 2.26% | 6.93% ± 2.44% | 2.22% ± 0.66% | **6.36% ± 1.79%** |
| Window-based Fusion (RGB + Pose) | VJEPA-2 (balanced) + PoseC3D (MLP fusion; 2-fold CV) | 9.76% ± 2.74% | 6.35% ± 1.64% | 1.85% ± 0.69% | 5.99% ± 1.69% |
| Window-based Fusion (RGB + Skeleton) | VJEPA-2 + STGCN++ (MLP fusion; optimized postprocess) | 8.42% ± 0.49% | 5.88% ± 0.38% | 2.67% ± 0.14% | 5.66% ± 0.34% |
| Window-based RGB | VJEPA-2 (balanced; optimized postprocess) | 9.48% ± 1.60% | 6.37% ± 0.96% | 1.99% ± 0.80% | 5.95% ± 1.10% |
| Window-based Pose | PoseC3D (CE; optimized postprocess) | 7.44% | 5.83% | 2.26% | 5.18% |
| Window-based Skeleton | STGCN++ (CE; optimized postprocess) | 6.30% | 4.27% | 1.28% | 3.95% |

### Model variants (sampling / loss; optimized postprocessing)

- **VJEPA-2 variants (window-based)**

| Variant | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
| --- | --- | --- | --- | --- |
| Baseline | 7.31% | 5.41% | 2.48% | 5.07% |
| Balanced sampling | 9.48% ± 1.60% | 6.37% ± 0.96% | 1.99% ± 0.80% | 5.95% ± 1.10% |
- **PoseC3D variants (window-based)**

| Variant | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
| --- | --- | --- | --- | --- |
| CE | 7.44% | 5.83% | 2.26% | 5.18% |
| Focal loss | 7.14% | 5.24% | 1.69% | 4.69% |
- **STGCN++ variants (window-based)**

| Variant | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
| --- | --- | --- | --- | --- |
| CE | 6.30% | 4.27% | 1.28% | 3.95% |
| Focal loss | 2.29% ± 0.99% | 1.88% ± 0.70% | 0.80% ± 0.44% | 1.65% ± 0.70% |
- **Fusion variants (window scores → segments)**

| Variant | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
| --- | --- | --- | --- | --- |
| VJEPA-2 + PoseC3D (MLP) | 9.95% ± 2.26% | 6.93% ± 2.44% | 2.22% ± 0.66% | 6.36% ± 1.79% |
| VJEPA-2 + STGCN++ (MLP) | 8.42% ± 0.49% | 5.88% ± 0.38% | 2.67% ± 0.14% | 5.66% ± 0.34% |
| VJEPA-2 (balanced) + PoseC3D (MLP) | 9.76% ± 2.74% | 6.35% ± 1.64% | 1.85% ± 0.69% | 5.99% ± 1.69% |

### Recall@tIoU (window-based; VJEPA-2 balanced)

| Metric | Value |
| --- | --- |
| Recall@0.3 | 15.73% ± 1.19% |
| Recall@0.5 | 12.45% ± 0.93% |
| avg_Recall | 11.30% ± 0.88% |

### Transformer TAL details (ActionFormer)

**End-to-end segment TAL (ActionFormer + VJEPA-2 features; 4-class)**

| Metric | Value (mean ± std) |
| --- | --- |
| mAP@0.3 | **23.97% ± 1.07%** |
| mAP@0.4 | 21.68% ± 1.05% |
| mAP@0.5 | 17.57% ± 2.92% |
| mAP@0.6 | 13.23% ± 3.55% |
| mAP@0.7 | 7.04% ± 0.67% |
| avg_mAP | **16.70% ± 1.53%** |

**Top-k recall (ActionFormer; 4-class)**

| tIoU | R@1 | R@5 | R@10 | R@100 |
| --- | --- | --- | --- | --- |
| 0.3 | 59.05% ± 6.12% | 88.88% ± 0.62% | 93.77% ± 1.89% | 97.38% ± 1.87% |
| 0.5 | 42.27% ± 0.45% | 74.96% ± 1.73% | 82.71% ± 0.52% | 90.68% ± 4.10% |
| 0.7 | 26.30% ± 1.45% | 44.67% ± 3.26% | 51.88% ± 5.35% | 66.02% ± 4.25% |

### Binary TAL (broad localization: RMM vs background)

**Goal:** Detect where any RMM occurs in a video, without requiring correct RMM subtype classification. This collapses all RMM types into a single positive class.

**Metrics reported:**

- mAP@tIoU
- Recall@tIoU
- (For proposal-based transformer evaluation) Top-k recall (R@k) at tIoU thresholds

### Main results

| Model family | Model | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP | Recall@0.3 | Recall@0.5 | avg_Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer Head (RGB) | **ActionFormer on VJEPA-2 features (binary)** | **40.83% ± 5.60%** | **28.55% ± 6.33%** | **11.97% ± 2.24%** | **27.93% ± 4.70%** | — | — | — |
| Window-based RGB | VJEPA-2 (binary; window-based) | 11.78% ± 0.63% | 6.77% ± 1.12% | 1.44% ± 0.33% | **6.66% ± 0.69%** | 19.57% ± 0.12% | 14.68% ± 0.57% | **13.57% ± 0.30%** |

### Transformer TAL details (ActionFormer; binary)

| Metric | Value (mean ± std) |
| --- | --- |
| mAP@0.3 | **40.83% ± 5.60%** |
| mAP@0.4 | 35.68% ± 5.12% |
| mAP@0.5 | 28.55% ± 6.33% |
| mAP@0.6 | 22.61% ± 4.67% |
| mAP@0.7 | 11.97% ± 2.24% |
| avg_mAP | **27.93% ± 4.70%** |

**Top-k recall (ActionFormer; binary)**

| tIoU | R@1 | R@5 | R@10 | R@100 |
| --- | --- | --- | --- | --- |
| 0.3 | 59.48% ± 5.56% | 90.48% ± 3.06% | 94.74% ± 1.61% | 98.37% ± 1.42% |
| 0.5 | 44.99% ± 5.38% | 73.23% ± 4.24% | 81.09% ± 2.70% | 90.90% ± 4.87% |
| 0.7 | 25.65% ± 2.09% | 45.54% ± 2.16% | 53.55% ± 4.34% | 72.16% ± 3.00% |