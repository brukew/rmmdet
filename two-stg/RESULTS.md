4# Two-Stage TAL Results

Binary ActionFormer detects RMM segments → V-JEPA2 classifies each into 4 RMM types → scores combined as `det_score × max(class_probs)`. Evaluated on 3-fold CV with no detection threshold (all proposals passed to classifier).

## Models Used

All paths relative to `/orcd/data/satra/001/users/brukew/actreg/`.

- **Detector:** `OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold{0,1,2}/gpu1_id99/result_detection.json`
- **Classifier:** `v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls/fold_{0,1,2}` (V-JEPA2, 64 frames, 20 epochs)

## Results

| Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|
| 0 | 30.32% | 23.39% | 14.00% | 23.23% |
| 1 | 22.87% | 15.87% | 6.28% | 15.77% |
| 2 | 22.67% | 13.96% | 6.88% | 14.50% |
| **Mean ± Std** | **25.29% ± 3.56%** | **17.74% ± 4.07%** | **9.05% ± 3.50%** | **17.83% ± 3.85%** |

Per-class AP@0.3 (mean across folds): hands flapping 39.3%, spinning 30.5%, jumping 24.2%, rocking 7.2%.

## Comparison

| Model | avg_mAP | Notes |
|-------|---------|-------|
| ActionFormer Binary | 27.93% | No class labels — detects RMM vs BG only |
| **Two-Stage (this)** | **17.83%** | Best 4-class TAL model |
| ActionFormer Balanced | 16.70% | End-to-end 4-class |
| V-JEPA + PoseC3D (MLP) | 6.36% | Best window-based |

## Artifacts

- Per-fold metrics/predictions: `two-stg/eval_results/fold{0,1,2}/metrics.json`
- CV summary: `two-stg/eval_results/cv_summary.json`
- Script: `two-stg/eval_two_stage_tal.py`
