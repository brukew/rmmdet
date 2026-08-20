# VLM Zero-Shot RMM Type Classification

**Date**: 2026-03-16
**Pipeline**: sails-vlm framework
**Task**: 5-class RMM type classification on pre-cut canonical clips

## Setup

- **Data**: `fold_0_val.csv` — 218 samples from canonical clips
- **Classes**: hands flapping, one hand flap, rocking, spinning, jumping
- **Class distribution**: hands flapping (98), jumping (45), rocking (31), one hand flap (29), spinning (15)
- **Inference**: 4 windows × 8 frames per clip, majority vote across windows
- **Sampling**: `do_sample=True`, `temperature=0.7` — 3 stochastic runs per model to estimate variance

## Results

| Metric | Cosmos-Reason2-8B (mean ± std) | Qwen3-VL-8B (mean ± std) | Qwen3-VL-30B-A3B (mean ± std) |
|--------|-------------------------------|--------------------------|-------------------------------|
| Top-1 Accuracy | 0.497 ± 0.017 | 0.543 ± 0.011 | **0.601 ± 0.005** |
| Top-2 Accuracy | 0.676 ± 0.010 | 0.693 ± 0.007 | **0.782 ± 0.011** |
| Macro F1 | 0.278 ± 0.022 | 0.331 ± 0.011 | **0.523 ± 0.009** |
| Weighted F1 | 0.405 ± 0.017 | 0.464 ± 0.013 | — |
| Balanced Accuracy | 0.294 ± 0.016 | 0.352 ± 0.007 | — |
| Cohen's κ | 0.165 ± 0.025 | 0.271 ± 0.020 | **0.423 ± 0.006** |

> Note: Qwen3-VL-30B-A3B used 8 windows × 16 frames (vs. 4 × 8 for 8B models). Results not directly comparable due to different inference config.

### Qwen3-VL-8B: Effect of More Frames (8×4 vs 16×8)

| Config | Top-1 (mean ± std) | Top-2 (mean ± std) | Macro F1 (mean ± std) | κ (mean ± std) |
|--------|-------------------|-------------------|----------------------|----------------|
| 8f × 4w | 0.543 ± 0.011 | 0.693 ± 0.007 | 0.331 ± 0.011 | 0.271 ± 0.020 |
| 16f × 8w | 0.544 ± 0.009 | 0.713 ± 0.004 | 0.318 ± 0.012 | 0.277 ± 0.017 |

> Increasing frames/windows had minimal effect on top-1 accuracy and F1. Top-2 marginally improved (+0.02). Not worth the ~9× inference cost.

#### Per-Class Metrics — Qwen3-VL-8B 16f×8w (mean ± std)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping | 0.566 ± 0.007 | 0.918 ± 0.008 | 0.700 ± 0.007 |
| one hand flap | 0.261 ± 0.055 | 0.046 ± 0.016 | 0.078 ± 0.026 |
| rocking | 0.434 ± 0.015 | 0.602 ± 0.030 | 0.504 ± 0.021 |
| spinning | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| jumping | 0.789 ± 0.029 | 0.193 ± 0.010 | 0.309 ± 0.013 |

### Individual Runs

| Run | Model | Config | Top-1 | Top-2 | Macro F1 | κ |
|-----|-------|--------|-------|-------|----------|---|
| 20260316_2021_cnzb | Cosmos-8B | 4w×8f | 0.472 | 0.665 | 0.250 | 0.132 |
| 20260316_2021_oubm | Cosmos-8B | 4w×8f | 0.509 | 0.688 | 0.293 | 0.190 |
| 20260316_2021_oved | Cosmos-8B | 4w×8f | 0.509 | 0.674 | 0.292 | 0.173 |
| 20260316_2023_ioqi | Qwen3-8B  | 4w×8f | 0.550 | 0.702 | 0.337 | 0.286 |
| 20260316_2023_oalj | Qwen3-8B  | 4w×8f | 0.550 | 0.688 | 0.339 | 0.282 |
| 20260316_2023_pxmw | Qwen3-8B  | 4w×8f | 0.528 | 0.688 | 0.318 | 0.246 |
| 20260320_0049_ebnt | Qwen3-8B  | 8w×16f | 0.546 | 0.716 | 0.315 | 0.278 |
| 20260320_0050_etiv | Qwen3-8B  | 8w×16f | 0.555 | 0.716 | 0.335 | 0.298 |
| 20260320_0126_swld | Qwen3-8B  | 8w×16f | 0.532 | 0.706 | 0.305 | 0.257 |
| 20260320_0054_dxfq | Qwen3-30B | 8w×16f | 0.606 | 0.771 | 0.532 | 0.429 |
| 20260320_0054_wuhy | Qwen3-30B | 8w×16f | 0.596 | 0.794 | 0.513 | 0.416 |
| 20260320_0205_uxjb | Qwen3-30B | 8w×16f | 0.573 | 0.780 | 0.485 | 0.387 |

### Per-Class Metrics (mean ± std across 3 runs)

#### Cosmos-Reason2-8B

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping | 0.502 ± 0.007 | 0.925 ± 0.021 | 0.651 ± 0.008 |
| one hand flap | 0.317 ± 0.138 | 0.149 ± 0.043 | 0.202 ± 0.068 |
| rocking | 0.548 ± 0.054 | 0.323 ± 0.046 | 0.404 ± 0.042 |
| spinning | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| jumping | 0.767 ± 0.024 | 0.074 ± 0.010 | 0.135 ± 0.018 |

#### Qwen3-VL-8B

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping | 0.561 ± 0.011 | 0.929 ± 0.014 | 0.699 ± 0.012 |
| one hand flap | 0.503 ± 0.041 | 0.172 ± 0.000 | 0.257 ± 0.005 |
| rocking | 0.408 ± 0.018 | 0.516 ± 0.000 | 0.455 ± 0.011 |
| spinning | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| jumping | 1.000 ± 0.000 | 0.141 ± 0.021 | 0.246 ± 0.033 |

#### Qwen3-VL-30B-A3B (8 windows × 16 frames)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping | 0.668 ± 0.004 | 0.811 ± 0.005 | 0.733 ± 0.005 |
| one hand flap | 0.483 ± 0.017 | 0.276 ± 0.034 | 0.351 ± 0.032 |
| rocking | 0.505 ± 0.005 | 0.758 ± 0.016 | 0.606 ± 0.009 |
| spinning | 0.557 ± 0.031 | 0.667 ± 0.000 | 0.607 ± 0.018 |
| jumping | 0.556 ± 0.000 | 0.222 ± 0.000 | 0.317 ± 0.000 |

## 4-Class Results (one hand flap → hands flapping)

### Summary

| Metric | Cosmos-Reason2-8B (mean ± std) | Qwen3-VL-8B (mean ± std) |
|--------|-------------------------------|--------------------------|
| Top-1 Accuracy | 0.627 ± 0.008 | **0.654 ± 0.013** |
| Top-2 Accuracy | 0.688 ± 0.010 | **0.728 ± 0.004** |
| Macro F1 | 0.326 ± 0.016 | **0.376 ± 0.013** |
| Weighted F1 | 0.531 ± 0.012 | **0.584 ± 0.014** |
| Balanced Accuracy | 0.342 ± 0.014 | **0.401 ± 0.009** |
| Cohen's κ | 0.192 ± 0.024 | **0.320 ± 0.024** |

### Per-Class Metrics (mean ± std across 3 runs)

#### Cosmos-Reason2-8B

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping | 0.632 ± 0.007 | 0.971 ± 0.004 | 0.765 ± 0.004 |
| rocking | 0.548 ± 0.054 | 0.323 ± 0.046 | 0.404 ± 0.042 |
| spinning | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| jumping | 0.767 ± 0.024 | 0.074 ± 0.010 | 0.135 ± 0.018 |

#### Qwen3-VL-8B

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping | 0.698 ± 0.009 | 0.948 ± 0.016 | 0.804 ± 0.011 |
| rocking | 0.408 ± 0.018 | 0.516 ± 0.000 | 0.455 ± 0.011 |
| spinning | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| jumping | 1.000 ± 0.000 | 0.141 ± 0.021 | 0.246 ± 0.033 |

## Whole-Video Inference — Qwen3-VL-30B-A3B (4-class, native video input)

**Setup**: `qwen3_video` wrapper passes full video path directly to processor (`nframes=total_frames`, capped at 768). No manual frame extraction or windowed voting. 4-class task (one hand flap excluded from labels). 3 stochastic runs on H200.

> **Note**: The val set contains 27 "one hand flap" GT clips (n=181 total after some clip load failures). These clips cannot be predicted correctly in 4-class mode. Raw metrics penalize them; merged metrics treat them as "hands flapping" for fair evaluation.

### Summary

| Metric | Raw (n=181) | Merged GT (one hand flap → hands flapping) |
|--------|------------|---------------------------------------------|
| Top-1 Accuracy | 0.514 ± 0.000 | **0.613 ± 0.005** |
| Macro F1 | 0.419 ± 0.008 | **0.534 ± 0.005** |
| Cohen's κ | 0.323 ± 0.004 | **0.404 ± 0.010** |

### Per-Class Metrics (merged GT, mean ± std)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping | 0.845 ± 0.016 | 0.679 ± 0.004 | 0.753 ± 0.004 |
| rocking | 0.436 ± 0.046 | 0.467 ± 0.038 | 0.447 ± 0.010 |
| spinning | 0.334 ± 0.015 | **0.846 ± 0.000** | 0.479 ± 0.015 |
| jumping | 0.480 ± 0.014 | 0.435 ± 0.013 | 0.456 ± 0.012 |

**Notable**: spinning recall 0.85 — highest across all experiments. Whole-video input dramatically improves spinning detection (vs 0.67 in sampled 30B, 0.00 in 8B models).

### Individual Runs

| Run | n | Top-1 (raw) | Top-1 (merged) | Macro F1 (merged) | κ (merged) |
|-----|---|-------------|----------------|-------------------|------------|
| 20260322_0336_hrzh | 181 | 0.514 | 0.619 | 0.541 | 0.418 |
| 20260322_0338_etbp | 181 | 0.514 | 0.608 | 0.528 | 0.396 |
| 20260322_0338_luvs | 181 | 0.514 | 0.613 | 0.533 | 0.400 |

## Key Findings

- **GT label mismatch**: val set has 29 "one hand flap" clips; 4-class runs penalize all of them. Use merged GT metrics as canonical for 4-class comparisons.
- **Qwen3-VL-30B-A3B is the best model** — 5-class: macro F1 0.52, kappa 0.42; whole-video 4-class (merged): F1 0.53, kappa 0.40
- **Scaling helps significantly**: 8B → 30B macro F1 jumps from 0.33 → 0.52
- **Spinning**: 0 recall in 8B models; 30B (sampled) achieves 0.67 recall; whole-video 30B achieves **0.85** — major gains with more compute
- **Rocking**: 30B recall 0.76 vs 8B 0.52 — another large gain
- **Jumping**: still weak across all models (low recall ~0.07–0.22)
- **Hands flapping bias reduced in 30B**: recall drops from 0.93 → 0.81 as model distributes attention more evenly
- **Kappa 0.42 (30B, 5-class)** — moderate agreement, approaching clinical utility range (>0.4)
- **Note**: 30B used 8×16 frames vs 4×8 for 8B — not a pure model comparison

## Output Directories

- Cosmos: `/orcd/data/satra/002/projects/SAILS/pred2annot_evaluation/rmm_type/cosmos/`
- Qwen3-8B: `/orcd/data/satra/002/projects/SAILS/pred2annot_evaluation/rmm_type/qwen3/`
- Qwen3-30B (sampled): `/orcd/data/satra/002/projects/SAILS/pred2annot_evaluation/rmm_type/qwen3_30b/`
- Qwen3-30B (whole-video): `/orcd/data/satra/002/projects/SAILS/pred2annot_evaluation/rmm_type/qwen3_30b_video/`
