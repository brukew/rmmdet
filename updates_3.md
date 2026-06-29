## **Temporal Action Localization (TAL) — Window-Level Detection (5-class, incl. background)**

### Setup

- **Task**: 5-class window classification — `background`, `hands_flapping`, `jumping`, `rocking`, `spinning`
- **Windows**: 2s windows with 1s stride (50% overlap)
- **CV**: 2–3 folds depending on model availability
- **Metrics**: Top-1 / Top-2 Accuracy, Macro-F1 (and per-class F1)
- **Challenge**: severe class imbalance (≈90% background windows)

### Methods

- **V-JEPA2** (RGB): linear probe, with/without class-balanced sampling
- **PoseC3D** (pose heatmaps): SlowOnly R50, CE vs Focal; with/without balanced sampling
- **STGCN++ (4-stream)** (skeleton graphs): CE vs Focal; with/without balanced sampling

### Results (window-level classification)

| Model | Top-1 Acc | Top-2 Acc | Macro-F1 | Notes |
| --- | --- | --- | --- | --- |
| **PoseC3D (CE)** | **85.89% ± 0.56%** | 96.16% ± 0.48% | 37.62% ± 3.23% | Best overall Top-1 |
| STGCN++ (CE) | 85.74% ± 0.14% | 95.18% ± 0.11% | 32.85% ± 1.48% | Competitive accuracy |
| V-JEPA2 | 81.75% ± 3.28% | 93.35% ± 2.03% | 36.10% ± 3.23% | Strong macro-F1 given imbalance |
| **V-JEPA2 (Balanced)** | 80.94% ± 1.73% | 95.08% ± 0.87% | **36.83% ± 3.45%** | Best trade-off for rare classes |
| PoseC3D (CE Balanced) | 81.11% ± 1.67% | **96.34% ± 0.39%** | 35.82% ± 5.78% | Better top-2; lower top-1 |
| PoseC3D (Focal) | 84.31% ± 1.33% | 96.07% ± 0.30% | 31.47% ± 1.80% | Focal underperformed CE |
| STGCN++ (CE Balanced) | 77.48% ± 3.75% | N/A | 30.40% ± 1.35% | Balancing hurt accuracy |
| STGCN++ (Focal) | 37.13% ± 3.75% | 74.95% ± 8.21% | 19.56% ± 1.18% | Training collapsed (unstable) |

### Notes

- **Rocking remains the hardest class** in window classification. Balancing improved rocking detection, but absolute F1 is still low:
  - **V-JEPA2 (Balanced)** rocking F1: **6.67% ± 4.1%** (best among single models)
  - PoseC3D (CE) rocking F1: 0.0% → **2.76% ± 2.3%** with balanced sampling
- **CE > Focal** for both PoseC3D and STGCN++ in this setup (Focal did not help with imbalance).

---

## **TAL — Window-Level TAL (window scores → segments + mAP@tIoU)**

### Setup

- **Task**: 4-class RMM localization — `hands_flapping`, `jumping`, `rocking`, `spinning` (background excluded from mAP)
- **Goal**: convert window-level scores into temporal segments and evaluate **segment-level** localization
- **Eval**: **mAP @ tIoU {0.3, 0.5, 0.7}** (avg over these tIoUs)
- **Default postprocessing**: `threshold=0.5, smooth_k=3, merge_gap=1.0s`
- **Postprocessing grid search**:
  - `threshold`: 0.3, 0.4, 0.5, 0.6
  - `smooth_k`: 1, 3, 5, 7
  - `merge_gap_sec`: 0.5, 1.0, 1.5, 2.0

### Results (single models)

**Best single-model results after optimized postprocessing**

| Model | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP | Notes |
| --- | --- | --- | --- | --- | --- |
| **V-JEPA2 (Balanced)** | 9.48% ± 1.60% | 6.37% ± 0.96% | 1.99% ± 0.80% | **5.95% ± 1.10%** | Best 4-class window-based single model |
| PoseC3D (CE) | 7.44% | 5.83% | 2.26% | **5.18%** | Strong with low threshold + no smoothing |
| V-JEPA2 | 7.31% | 5.41% | 2.48% | 5.07% | Benefits from smoothing |
| PoseC3D (Focal) | 7.14% | 5.24% | 1.69% | 4.69% | Below CE |
| STGCN++ 4-Stream (CE) | 6.30% | 4.27% | 1.28% | 3.95% | Lower overall |

**Recall@tIoU (V-JEPA2 Balanced; 3-fold CV)**

| Metric | Value |
| --- | --- |
| Recall@0.3 | 15.73% ± 1.19% |
| Recall@0.5 | 12.45% ± 0.93% |
| **avg_Recall** | **11.30% ± 0.88%** |

### Results (late fusion)

Late fusion uses a lightweight MLP on log-probabilities (OOF trained on validation windows, grouped by video).

| Model | Folds | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **V-JEPA2 + PoseC3D (MLP)** | 2 | 9.95% ± 2.26% | 6.93% ± 2.44% | 2.22% ± 0.66% | **6.36% ± 1.79%** | Best window-based fusion |
| V-JEPA2 + STGCN++ (MLP) | 2 | 8.42% ± 0.49% | 5.88% ± 0.38% | 2.67% ± 0.14% | **5.66% ± 0.34%** | Below PoseC3D fusion |

### Notes

- **Postprocessing mattered a lot**:
  - A **lower threshold (≈0.3)** improved recall vs the default 0.5
  - Pose-based models preferred **no smoothing** (`smooth_k=1`) while V-JEPA2 benefited from **mild smoothing**
- **Primary failure mode remains “rocking”** (subtle + rare), which suppresses overall TAL mAP.

---

## **TAL — Segment-Level Localization (transformer-based, ActionFormer / OpenTAD)**

### Setup

- **Task**: 4-class RMM segment localization — `hands_flapping`, `jumping`, `rocking`, `spinning`
- **Model**: ActionFormer (transformer-based TAL)
- **Features**: V-JEPA2 1024-dim features (snippet-based)
- **Eval**: mAP @ tIoU {0.3, 0.4, 0.5, 0.6, 0.7}, **3-fold CV**

### Results (4-class ActionFormer + V-JEPA2 Balanced)

| Metric | Value (mean ± std) |
| --- | --- |
| mAP@0.3 | **23.97% ± 1.07%** |
| mAP@0.4 | 21.68% ± 1.05% |
| mAP@0.5 | 17.57% ± 2.92% |
| mAP@0.6 | 13.23% ± 3.55% |
| mAP@0.7 | 7.04% ± 0.67% |
| **avg_mAP** | **16.70% ± 1.53%** |

### Recall (ActionFormer top-k, proposal-based)

| tIoU | R@1 | R@5 | R@10 | R@100 |
| --- | --- | --- | --- | --- |
| 0.3 | 59.05% ± 6.12% | 88.88% ± 0.62% | 93.77% ± 1.89% | 97.38% ± 1.87% |
| 0.5 | 42.27% ± 0.45% | 74.96% ± 1.73% | 82.71% ± 0.52% | 90.68% ± 4.10% |
| 0.7 | 26.30% ± 1.45% | 44.67% ± 3.26% | 51.88% ± 5.35% | 66.02% ± 4.25% |

### Notes

- **End-to-end transformer TAL substantially outperformed** window-based + postprocessing baselines for 4-class localization.

---

## **Broad Localization (Binary: RMM vs Background) — added after 1/5 meeting**

### Context / Motivation

- In the **1/5 meeting**, we discussed prioritizing **broad localization** (“find RMM anywhere”) even when fine-grained type classification is uncertain.
- In response, I ran a **binary TAL formulation** (RMM vs background) as a stronger starting point for broad localization.

### Results (Window-based Binary: V-JEPA2)

| Metric | Value (mean ± std) |
| --- | --- |
| mAP@0.3 | 11.78% ± 0.63% |
| mAP@0.5 | 6.77% ± 1.12% |
| mAP@0.7 | 1.44% ± 0.33% |
| **avg_mAP** | **6.66% ± 0.69%** |
| Recall@0.3 | 19.57% ± 0.12% |
| Recall@0.5 | 14.68% ± 0.57% |
| **avg_Recall** | **13.57% ± 0.30%** |

### Results (Segment-level Binary: ActionFormer + V-JEPA2)

| Metric | Value (mean ± std) |
| --- | --- |
| mAP@0.3 | **40.83% ± 5.60%** |
| mAP@0.4 | 35.68% ± 5.12% |
| mAP@0.5 | 28.55% ± 6.33% |
| mAP@0.6 | 22.61% ± 4.67% |
| mAP@0.7 | 11.97% ± 2.24% |
| **avg_mAP** | **27.93% ± 4.70%** |

### Recall (ActionFormer top-k, proposal-based)

| tIoU | R@1 | R@5 | R@10 | R@100 |
| --- | --- | --- | --- | --- |
| 0.3 | 59.48% ± 5.56% | 90.48% ± 3.06% | 94.74% ± 1.61% | 98.37% ± 1.42% |
| 0.5 | 44.99% ± 5.38% | 73.23% ± 4.24% | 81.09% ± 2.70% | 90.90% ± 4.87% |
| 0.7 | 25.65% ± 2.09% | 45.54% ± 2.16% | 53.55% ± 4.34% | 72.16% ± 3.00% |

### Notes

- Binary localization is a strong fit for **broad RMM spotting**, and can be used as a first stage before type classification (if needed).

---

## Notes / Next Steps

- **Prediction analysis**:
  - stratify errors by **timepoint / child / environment**
  - analyze **segment length distributions** (GT vs predicted), over-merge / over-split behaviors, and boundary errors
- **Expand broad localization** to include **“other” RMMs**:
  - note: annotations for these RMMs have **just been completed**, so they can now be integrated into the broad-localization task
- **Continue SOTA transformer-based TAL**:
  - expand beyond ActionFormer (and/or tune transformer-based localization models + decoding/postprocessing)

