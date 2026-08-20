# Two-Stage TAL Results

**Date:** March 18, 2026
**Task:** 4-class RMM Temporal Action Localization (hands flapping, jumping, rocking, spinning)

## Method

**Stage 1 — Detection:** ActionFormer Binary (RMM vs Background) produces temporal proposals with detection scores. No threshold applied — all proposals passed to Stage 2.

**Stage 2 — Classification:** V-JEPA2 4-class classifier runs on each detected segment (64 frames uniformly sampled). Assigns an RMM class label and confidence.

**Score combination:** `final_score = detection_score × max(class_probabilities)`

## Models

### Stage 1: ActionFormer Binary Detections

Precomputed `result_detection.json` from binary ActionFormer trained on V-JEPA2 features.

| Fold | Detection JSON |
|------|---------------|
| 0 | `OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold0/gpu1_id99/result_detection.json` |
| 1 | `OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold1/gpu1_id99/result_detection.json` |
| 2 | `OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold2/gpu1_id99/result_detection.json` |

### Stage 2: V-JEPA2 4-Class Classifier

Fine-tuned `VJEPA2ForVideoClassification` checkpoints (64 frames, lr=1e-5, batch=1, accum=8, 20 epochs, crop augmentation).

| Fold | Checkpoint Directory |
|------|---------------------|
| 0 | `v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls/fold_0` |
| 1 | `v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls/fold_1` |
| 2 | `v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls/fold_2` |

All paths are relative to the repo root.

## Results (3-Fold Cross-Validation)

### Per-Fold mAP

| Fold | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|---------|---------|
| 0 | 30.32% | 27.56% | 23.39% | 20.90% | 14.00% | 23.23% |
| 1 | 22.87% | 20.72% | 15.87% | 13.10% | 6.28% | 15.77% |
| 2 | 22.67% | — | 13.96% | — | 6.88% | 14.50% |
| **Mean ± Std** | **25.29% ± 3.56%** | **24.14% ± 3.42%** | **17.74% ± 4.07%** | **17.00% ± 3.90%** | **9.05% ± 3.50%** | **17.83% ± 3.85%** |

### Per-Class AP (Mean ± Std across folds)

| Class | AP@0.3 | AP@0.5 | AP@0.7 |
|-------|--------|--------|--------|
| hands flapping | 39.33% ± 4.07% | 24.82% ± 3.33% | 8.45% ± 1.12% |
| jumping | 24.19% ± 12.76% | 19.89% ± 13.68% | 10.73% ± 6.12% |
| spinning | 30.47% ± 7.83% | 22.05% ± 5.56% | 15.87% ± 8.26% |
| rocking | 7.16% ± 3.29% | 4.19% ± 1.48% | 1.15% ± 0.86% |

Rocking is consistently the hardest class; spinning retains the highest AP at strict tIoU=0.7.

## Comparison with Other Models

| Model | Type | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|-------|------|---------|---------|---------|---------|
| ActionFormer Binary | End-to-end (no class labels) | 40.83% | 28.55% | 11.97% | 27.93% |
| **Two-Stage (Binary AF + V-JEPA2 4-class)** | **Detect then classify** | **25.29%** | **17.74%** | **9.05%** | **17.83%** |
| ActionFormer Balanced | End-to-end 4-class | 23.97% | 17.57% | 7.04% | 16.70% |
| V-JEPA + PoseC3D (MLP) | Window-based fusion | 9.95% | 6.93% | 2.22% | 6.36% |

- Outperforms ActionFormer Balanced by **+6.8% relative** on avg_mAP (17.83% vs 16.70%).
- Stronger at strict localization: **+28.6% relative** at mAP@0.7 (9.05% vs 7.04%).
- 2.8× better than the best window-based approach (6.36%).
- Trails binary-only ActionFormer (27.93%), which detects RMM presence without class distinction.

## Output Artifacts

Per-fold results in this directory (`two-stg/eval_results/fold{0,1,2}/`):

| File | Description |
|------|-------------|
| `predictions.csv` | All predictions (video_key, class_id, start_sec, end_sec, score) |
| `metrics.json` | mAP metrics and per-class AP breakdown |

Cross-validation summary: `cv_summary.json` (in this directory)

## Reproducing

```bash
cd <repo root>

# Single fold
sbatch --export=FOLD=0 --partition=mit_normal_gpu two-stg/run_single_fold_gpu.sh

# All folds + aggregate
bash two-stg/run_two_stage_cv.sh

# Aggregate only (if folds already done)
python two-stg/aggregate_cv_results.py
```
