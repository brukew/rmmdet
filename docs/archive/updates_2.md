
## Classification

**4-Class Task**

| Method | Top-1 Acc | Top-2 Acc | Macro-F1 | Macro Prec | Macro Rec |
| --- | --- | --- | --- | --- | --- |
| **3-Way MLP Fusion** | **84.7% ± 1.9%** | 96.2% ± 1.9% | 82.1% ± 3.9% | 83.2% ± 4.1% | 83.3% ± 3.5% |
| MLP (V-JEPA2 + STGCN++) | 83.8% ± 2.2% | **96.5% ± 0.4%** | **82.2% ± 2.5%** | **83.8% ± 1.5%** | **83.6% ± 2.5%** |
| MLP (V-JEPA2 + PoseC3D) | 80.3% ± 3.3% | 95.1% ± 2.4% | 78.9% ± 4.4% | 80.3% ± 4.1% | 80.6% ± 4.1% |
| STGCN++ (4-stream) | 77.9% ± 2.2% | 95.1% ± 0.7% | 72.7% ± 4.5% | 77.6% ± 0.7% | 71.0% ± 6.5% |
| V-JEPA2 | 76.0% ± 3.3% | 93.7% ± 1.3% | 75.1% ± 1.9% | 78.4% ± 0.7% | 73.0% ± 3.3% |
| PoseC3D (non-weighted) | 74.6% ± 2.6% | 91.3% ± 0.3% | 70.4% ± 5.6% | 75.3% ± 6.2% | 68.4% ± 8.0% |
| Qwen2.5-VL (zero-shot) | 56.8% ± 3.4% | 78.9% ± 2.3% | 27.8% ± 0.7% | 51.2% ± 10.0% | 31.2% ± 0.7% |

**5-Class Task**

| Method | Top-1 Acc | Top-2 Acc | Macro F1 | Macro Prec | Macro Rec |
| --- | --- | --- | --- | --- | --- |
| **3-Way MLP** | **69.6% ± 3.0%** | **88.5% ± 0.4%** | **69.8% ± 1.8%** | **71.7% ± 2.6%** | **71.0% ± 1.5%** |
| MLP Fusion (V-JEPA2 + STGCN++) | 68.2% ± 1.5% | 88.1% ± 2.1% | 68.5% ± 1.9% | 70.0% ± 2.2% | 70.3% ± 1.7% |
| MLP Fusion (V-JEPA2 + PoseC3D) | 67.6% ± 3.6% | 87.5% ± 1.1% | 67.8% ± 1.0% | 69.2% ± 1.8% | 69.2% ± 2.0% |
| PoseC3D + Weighted | 67.0% ± 1.6% | 85.5% ± 1.3% | 65.9% ± 2.8% | 67.8% ± 3.1% | 65.4% ± 3.0% |
| STGCN++ 4-stream | 66.5% ± 1.5% | 83.3% ± 1.0% | 64.6% ± 1.4% | 67.4% ± 3.7% | 65.0% ± 3.3% |
| V-JEPA2 | 62.5% ± 2.1% | 84.8% ± 1.3% | 63.0% ± 1.1% | 64.4% ± 2.1% | 61.5% ± 0.8% |
| Qwen2.5-VL (zero-shot) | 37.7% ± 6.0% | 63.6% ± 2.2% | 12.3% ± 1.9% | 20.8% ± 6.3% | 20.3% ± 1.0% |

**Per-Class Precision/Recall — 4-Class Task**

| Model | hands_flapping | jumping | rocking | spinning |
| --- | --- | --- | --- | --- |
| 3-Way MLP | **92.3%** / 87.0% | 74.7% / **83.6%** | 76.7% / **76.7%** | 86.7% / 86.7% |
| STGCN++ (4-stream) | 81.6% / 85.7% | **74.8%** / 62.9% | 74.5% / 56.0% | 79.7% / 79.4% |
| V-JEPA2 | 80.5% / 84.0% | 64.8% / 68.4% | **80.0%** / 61.3% | **88.3%** / 78.3% |
| PoseC3D | 77.4% / **86.7%** | 69.3% / 65.2% | 64.8% / 47.1% | **89.6%** / 74.4% |
| Qwen2.5-VL | 58.6% / 93.9% | 46.1% / 27.6% | 100% / 3.1% | 0% / 0% |

**Per-Class Precision/Recall — 5-Class Task**

| Model | hands_flapping | jumping | one_hand_flap | rocking | spinning |
| --- | --- | --- | --- | --- | --- |
| 3-Way MLP | **76.2%** / 66.8% | 69.4% / **78.6%** | 50.5% / **53.6%** | 68.1% / **68.9%** | 87.0% / **88.9%** |
| STGCN++ 4-stream | 66.5% / 71.7% | 65.9% / 74.1% | 39.2% / 37.3% | **72.8%** / 53.9% | **87.1%** / 84.4% |
| PoseC3D + Weighted | 65.8% / **73.3%** | **70.2%** / 77.1% | **57.3%** / 41.1% | 61.2% / 52.5% | 84.3% / 82.9% |
| V-JEPA2 | 62.6% / 68.8% | 55.7% / 69.1% | 45.6% / 29.4% | 72.7% / 53.2% | 85.4% / 86.8% |
| Qwen2.5-VL | 37.6% / 98.0% | 66.7% / 3.5% | 0% / 0% | 0% / 0% | 0% / 0% |

**Notes**

1. **3-Way MLP Fusion achieves best performance**
    - 4-class: 84.7% Top-1 (+7% over best standalone model)
    - 5-class: 69.6% Top-1 (+2-3% improvement)
    - 412 parameters, evaluated with nested 5-fold CV
2. **Different models excel at different classes**
    - Skeleton models (STGCN++/PoseC3D) better for "jumping" (70% prec, 77% rec)
    - RGB features (V-JEPA2) better for "rocking" (73-80% precision)
    - Fusion captures complementary strengths with balanced P/R

**Next Steps:**

- Wrapping up action localization task (given full video, mark start and end times for each RMM)
    - Naive detection based localization vs Transformer model
