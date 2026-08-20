# TAL (Temporal Action Localization) Evaluation Pipeline

This directory contains the evaluation pipeline for TAL on SAILS RMM behaviors.

## Overview

The evaluation pipeline:
1. Takes **window-level class scores** (2s windows, 1s stride)
2. **Postprocesses** into predicted segments (threshold, smooth, merge)
3. **Evaluates** against GT segments using **mAP@tIoU {0.3, 0.5, 0.7}**

Supports: **V-JEPA**, **PoseC3D**, **ST-GCN/STGCN++**, and **late fusion** models.

## Directory Structure

```
actreg/tal/                          # This directory (evaluation)
├── README.md
├── window_to_segments.py            # Window scores → segments
├── tal_map_eval.py                  # Segment-level AP/mAP
├── eval_tal_from_window_preds.py    # End-to-end CLI
├── export_pyskl_window_preds.py     # pyskl → common format
├── run_tal_eval_cv.py               # CV fold runner
├── tal_fusion.py                    # Late fusion (MLP on log-probs)
├── test_tal_eval.py                 # Unit tests + sanity checks
├── test_tal_fusion.py               # Fusion module tests
└── scripts/
    ├── grid_search_postprocessing.py   # Postprocess param sweep
    ├── run_tal_fusion_oof.py           # Generate fused predictions
    └── eval_best_postprocess.py        # Evaluate with best params

actreg/dataprep/tal/                 # Data preparation (separate)
├── splits_cv_4class/                # TAL window CSVs
├── splits_cv_5class/
├── label_maps/
├── make_tal_window_splits.py        # Generate window splits
├── validate_tal_splits.py           # Validation
└── ...
```

## Quick Start

### 1. Run unit tests first

```bash
cd tal
python test_tal_eval.py
# or from repo root: python tal/test_tal_eval.py
```

### 2. Evaluate V-JEPA

V-JEPA saves `window_level_preds.csv` after training:

```bash
python eval_tal_from_window_preds.py \
    --window-preds /path/to/vjepa/runs/tal_cv/fold0/window_level_preds.csv \
    --out-dir eval_results/vjepa_fold0
```

### 3. Evaluate PoseC3D or ST-GCN

```bash
# from repo root (helper lives in dataprep/tal/)
python dataprep/tal/export_pyskl_window_preds.py \
    --ann-pkl pyskl/data/sails/tal/cv_4class/5class_windows_conf04/fold0.pkl \
    --scores-pkl pyskl/work_dirs/posec3d/tal/fold0/result.pkl \
    --window-csv dataprep/tal/splits_cv_4class/fold_0_val_windows.csv \
    --out-csv tal/posec3d_fold0_preds.csv

python tal/eval_tal_from_window_preds.py \
    --window-preds tal/posec3d_fold0_preds.csv \
    --out-dir tal/eval_results/posec3d_fold0
```

### 4. CV Evaluation (all folds)

```bash
python run_tal_eval_cv.py \
    --model vjepa \
    --preds-root /path/to/vjepa/runs/tal_cv \
    --out-dir eval_results/vjepa_cv
```

---

## Window Prediction CSV Format

All models must produce a CSV with these columns:

| Column | Description |
|--------|-------------|
| `window_id` | Unique window identifier |
| `video_key` | Normalized video path for matching |
| `start_sec` | Window start time |
| `end_sec` | Window end time |
| `score_class0` | Score for class 0 (hands flapping) |
| `score_class1` | Score for class 1 (jumping) |
| `score_class2` | Score for class 2 (rocking) |
| `score_class3` | Score for class 3 (spinning) |
| `score_class4` | Score for class 4 (background) — optional |

## Postprocessing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--smooth-k` | 3 | Moving average window for score smoothing |
| `--thr` | 0.5 | Global threshold for binary activation |
| `--thr-per-class` | None | Per-class thresholds, e.g., `"0:0.4,1:0.5"` |
| `--merge-gap-sec` | 1.0 | Max gap (seconds) to merge segments |
| `--min-duration-sec` | 0.0 | Minimum segment duration |
| `--score-reducer` | max | Aggregate window scores (`max` or `mean`) |

**Recommended defaults**:
- `--smooth-k 3` (mild smoothing)
- `--thr 0.5` as baseline
- `--merge-gap-sec 1.0` (matches window stride)

## Output Artifacts

Each evaluation produces:

| File | Description |
|------|-------------|
| `pred_segments.csv` | Predicted segments with scores |
| `metrics.json` | mAP and per-class AP values |
| `per_class_ap.csv` | Per-class AP table |
| `report.txt` | Human-readable summary |
| `eval_config.json` | Evaluation parameters |

CV evaluation also produces:
- `cv_summary.json` — Mean ± std across folds
- `cv_summary.txt` — Human-readable CV summary

## Metrics

- **mAP@tIoU {0.3, 0.5, 0.7}**: Segment-level mean Average Precision
- **Per-class AP**: AP for each RMM class (hands flapping, jumping, rocking, spinning)
- **avg_mAP**: Mean of mAP@0.3, mAP@0.5, mAP@0.7

### What is tIoU?

**tIoU (temporal IoU)** measures overlap between predicted and GT segments:
- `intersection / union` of time intervals
- tIoU = 0.7 means the prediction must overlap 70%+ with GT to count as correct

## Detailed Usage

### Evaluate V-JEPA

```bash
python eval_tal_from_window_preds.py \
    --window-preds /path/to/vjepa/runs/tal_cv/fold0/window_level_preds.csv \
    --splits-root ../dataprep/splits \
    --task 4class \
    --out-dir eval_results/vjepa_fold0
```

### Evaluate PoseC3D

1. **Run pyskl test**:
```bash
cd ../pyskl
python tools/test.py \
    configs/posec3d/slowonly_r50_sails_k400p/joint_tal_5class.py \
    -C work_dirs/posec3d/tal/fold0/latest.pth \
    --out work_dirs/posec3d/tal/fold0/result.pkl \
    --launcher none
```

2. **Export to common format**:
```bash
cd ../tal
python export_pyskl_window_preds.py \
    --ann-pkl ../pyskl/data/sails/tal/cv_4class/5class_windows_conf04/fold0.pkl \
    --scores-pkl ../pyskl/work_dirs/posec3d/tal/fold0/result.pkl \
    --window-csv ../dataprep/tal/splits_cv_4class/fold_0_val_windows.csv \
    --out-csv posec3d_fold0_preds.csv
```

3. **Evaluate**:
```bash
python eval_tal_from_window_preds.py \
    --window-preds posec3d_fold0_preds.csv \
    --out-dir eval_results/posec3d_fold0
```

### Evaluate ST-GCN / STGCN++

Same as PoseC3D, just use the appropriate config:

```bash
# Test
python tools/test.py \
    configs/stgcn++/stgcnpp_sails_ntu60p/j_tal_5class.py \
    -C work_dirs/stgcnpp/tal/fold0/latest.pth \
    --out work_dirs/stgcnpp/tal/fold0/result.pkl \
    --launcher none

# Export and evaluate (same as above)
```

### CV Fold Evaluation

```bash
# V-JEPA
python run_tal_eval_cv.py \
    --model vjepa \
    --preds-root /path/to/vjepa/runs/tal_cv \
    --out-dir eval_results/vjepa_cv

# PoseC3D
python run_tal_eval_cv.py \
    --model posec3d \
    --preds-root ../pyskl/work_dirs/posec3d/tal \
    --out-dir eval_results/posec3d_cv

# STGCN++
python run_tal_eval_cv.py \
    --model stgcn \
    --preds-root ../pyskl/work_dirs/stgcnpp/tal \
    --out-dir eval_results/stgcn_cv
```

---

## Late Fusion: V-JEPA + Skeleton Models

The pipeline supports **late fusion** of V-JEPA (RGB) with skeleton-based models (PoseC3D, STGCN++) via a lightweight MLP trained on log-probability features.

### How Fusion Works

1. **Feature space**: Log-probabilities from both modalities (PoE-style)
2. **Training**: Out-of-fold (OOF) inner CV on val windows, grouped by video
3. **Missing modality**: Filled with uniform distribution + missing indicator flag
4. **Output**: Fused TAL-format CSV ready for postprocessing

### Step 1: Generate Fused Predictions

```bash
cd actreg/tal

# Generate fused predictions for both pairs (V-JEPA + PoseC3D, V-JEPA + STGCN++)
python scripts/run_tal_fusion_oof.py

# Or run a specific pair
python scripts/run_tal_fusion_oof.py --pairs vjepa_posec3d

# With custom config
python scripts/run_tal_fusion_oof.py \
    --hidden-dim 16 \
    --num-epochs 100 \
    --n-inner-folds 5
```

**Output:**
- `eval_results/vjepa_posec3d_mlp_logp/fold{0,1}/tal_format_preds.csv`
- `eval_results/vjepa_stgcnpp_mlp_logp/fold{0,1}/tal_format_preds.csv`

### Step 2: Grid Search Postprocessing Parameters

```bash
# Run grid search on all models (including fusion)
python scripts/grid_search_postprocessing.py

# Or run only on fusion models
python scripts/grid_search_postprocessing.py \
    --models vjepa_posec3d_mlp_logp vjepa_stgcnpp_mlp_logp
```

**Output:**
- `eval_results/grid_search_results.csv` — All parameter combinations
- `eval_results/best_params_per_model.json` — Optimal settings per model

### Step 3: Evaluate with Best Parameters

```bash
# Evaluate all models with their best postprocessing params
python scripts/eval_best_postprocess.py

# Or evaluate specific models
python scripts/eval_best_postprocess.py \
    --models vjepa_posec3d_mlp_logp vjepa_stgcnpp_mlp_logp
```

**Output per model:**
- `{model}/fold{N}/best_eval/metrics.json`
- `{model}/fold{N}/best_eval/pred_segments.csv`
- `{model}/cv_summary.json` — Mean ± std across folds

### Full Fusion Pipeline Example

```bash
cd actreg/tal

# 1. Generate fused predictions
python scripts/run_tal_fusion_oof.py

# 2. Grid search postprocessing
python scripts/grid_search_postprocessing.py

# 3. Evaluate with best params
python scripts/eval_best_postprocess.py

# View results
cat eval_results/vjepa_posec3d_mlp_logp/cv_summary.json
cat eval_results/vjepa_stgcnpp_mlp_logp/cv_summary.json
```

---

### Run Tests

```bash
# Unit tests only
python test_tal_eval.py

# Unit tests + oracle sanity check on real data
python test_tal_eval.py --oracle --splits-root ../dataprep/splits

# Fusion module tests
python test_tal_fusion.py

# Fusion pipeline smoke test
python test_tal_eval.py --fusion
```

The oracle check creates "perfect" predictions from GT and verifies mAP ≈ 1.0.

## Class Labels (4-class)

| Index | Label |
|-------|-------|
| 0 | hands flapping |
| 1 | jumping |
| 2 | rocking |
| 3 | spinning |
| 4 | background (optional) |

## See Also

- **Data preparation**: `actreg/dataprep/tal/README.md`
- **V-JEPA training**: `actreg/v-jepa/finetune_sails_vjepa2_tal.py`
- **pyskl training**: `actreg/pyskl/tools/train_weighted.py`
