# TAL (Temporal Action Localization) Evaluation Pipeline

This directory contains the evaluation pipeline for TAL on SAILS RMM behaviors.

## Overview

The evaluation pipeline:
1. Takes **window-level class scores** (2s windows, 1s stride)
2. **Postprocesses** into predicted segments (threshold, smooth, merge)
3. **Evaluates** against GT segments using **mAP@tIoU {0.3, 0.5, 0.7}**

Supports: **V-JEPA**, **PoseC3D**, **ST-GCN/STGCN++**

## Directory Structure

```
actreg/tal/                          # This directory (evaluation)
├── README.md
├── window_to_segments.py            # Window scores → segments
├── tal_map_eval.py                  # Segment-level AP/mAP
├── eval_tal_from_window_preds.py    # End-to-end CLI
├── export_pyskl_window_preds.py     # pyskl → common format
├── run_tal_eval_cv.py               # CV fold runner
└── test_tal_eval.py                 # Unit tests + sanity checks

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
cd actreg/tal
python test_tal_eval.py
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
# 1. Export pyskl test results to common format
python export_pyskl_window_preds.py \
    --ann-pkl ../pyskl/data/sails/tal/cv_4class/5class_windows_conf04/fold0.pkl \
    --scores-pkl ../pyskl/work_dirs/posec3d/tal/fold0/result.pkl \
    --window-csv ../dataprep/tal/splits_cv_4class/fold_0_val_windows.csv \
    --out-csv posec3d_fold0_preds.csv

# 2. Evaluate
python eval_tal_from_window_preds.py \
    --window-preds posec3d_fold0_preds.csv \
    --out-dir eval_results/posec3d_fold0
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

### Run Sanity Checks

```bash
# Unit tests only
python test_tal_eval.py

# Unit tests + oracle sanity check on real data
python test_tal_eval.py --oracle --splits-root ../dataprep/splits
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
