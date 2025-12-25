# PoseC3D for SAILS RMM Dataset

This configuration finetunes PoseC3D on the SAILS Repetitive Motor Movements (RMM) dataset using Kinetics-400 pretrained weights.

## Classes

### 4-Class
| Index | Class Name |
|-------|------------|
| 0 | hands flapping |
| 1 | jumping |
| 2 | rocking |
| 3 | spinning |

### 5-Class
| Index | Class Name |
|-------|------------|
| 0 | hands flapping |
| 1 | jumping |
| 2 | one hand flap |
| 3 | rocking |
| 4 | spinning |

---

## Quick Start

```bash
cd /orcd/data/satra/001/users/brukew/actreg/pyskl

# 1. Generate annotations (all splits)
bash scripts/generate_all_pickles.sh

# 2. Submit SLURM jobs (all 4 training configurations)
bash scripts/slurm/submit_all.sh

# Or run locally for testing:
python tools/train.py configs/posec3d/slowonly_r50_sails_k400p/joint.py \
    --work-dir work_dirs/test \
    --cfg-options \
        ann_file=data/sails/cv/4class_conf04/fold0.pkl \
        data.train.dataset.ann_file=data/sails/cv/4class_conf04/fold0.pkl \
        data.val.ann_file=data/sails/cv/4class_conf04/fold0.pkl \
        data.test.ann_file=data/sails/cv/4class_conf04/fold0.pkl \
        total_epochs=2 \
        optimizer.lr=0.00125 \
    --validate \
    --launcher none
```

---

## Data Preparation

### Directory Structure

```
data/sails/
├── single/                          # Single train/val/test split
│   ├── 4class_conf04.pkl           # 4-class, confidence ≥ 0.4
│   └── 5class_conf04.pkl           # 5-class, confidence ≥ 0.4
├── cv/                              # Cross-validation folds
│   ├── 4class_conf04/
│   │   ├── fold0.pkl
│   │   ├── fold1.pkl
│   │   └── fold2.pkl
│   └── 5class_conf04/
│       ├── fold0.pkl
│       ├── fold1.pkl
│       └── fold2.pkl
```

### Generate All Pickles

```bash
# Generate all splits at once (4-class and 5-class, single and CV)
LOG_DIR="logs/pickle_generation"
mkdir -p $LOG_DIR

# 4-class single split
python tools/data/create_sails_annotations.py \
    --mode single \
    --splits-dir /orcd/data/satra/001/users/brukew/actreg/dataprep/splits/single_split_4class/ \
    --output-dir data/sails/single \
    --num-classes 4 \
    --min-keypoint-conf 0.4 2>&1 | tee $LOG_DIR/4class_single.log

# 4-class CV splits
python tools/data/create_sails_annotations.py \
    --mode cv \
    --splits-dir /orcd/data/satra/001/users/brukew/actreg/dataprep/splits/cv_splits_4class/ \
    --output-dir data/sails/cv \
    --num-classes 4 \
    --min-keypoint-conf 0.4 2>&1 | tee $LOG_DIR/4class_cv.log

# 5-class single split
python tools/data/create_sails_annotations.py \
    --mode single \
    --splits-dir /orcd/data/satra/001/users/brukew/actreg/dataprep/splits/single_split/ \
    --output-dir data/sails/single \
    --num-classes 5 \
    --min-keypoint-conf 0.4 2>&1 | tee $LOG_DIR/5class_single.log

# 5-class CV splits
python tools/data/create_sails_annotations.py \
    --mode cv \
    --splits-dir /orcd/data/satra/001/users/brukew/actreg/dataprep/splits/cv_splits/ \
    --output-dir data/sails/cv \
    --num-classes 5 \
    --min-keypoint-conf 0.4 2>&1 | tee $LOG_DIR/5class_cv.log
```

### Annotation Script Options

| Option | Description |
|--------|-------------|
| `--mode single\|cv\|manual` | Generation mode |
| `--splits-dir DIR` | Directory with CSV splits |
| `--output-dir DIR` | Output directory for pickles |
| `--num-classes 4\|5` | Number of classes (determines class mapping) |
| `--min-keypoint-conf 0.4` | Zero out keypoints with confidence < threshold |
| `--dry-run` | Check coverage without generating |

---

## Training

### SLURM Cluster Submission (Recommended)

Submit all 4 training jobs to the cluster:

```bash
cd /orcd/data/satra/001/users/brukew/actreg/pyskl
bash scripts/slurm/submit_all.sh
```

This submits:
1. **4-class single split** → `posec3d_4class_single` (12h)
2. **4-class CV (3 folds)** → `posec3d_4class_cv` (24h)
3. **5-class single split** → `posec3d_5class_single` (12h)
4. **5-class CV (3 folds)** → `posec3d_5class_cv` (24h)

**Monitor jobs:**
```bash
squeue -u $USER
```

**Logs:**
```
/orcd/data/satra/001/users/brukew/logs/posec3d_*.out
/orcd/data/satra/001/users/brukew/logs/posec3d_*.err
```

### Local Training (Single GPU)

**Important:** When using `--cfg-options` to override `ann_file`, you must override ALL nested paths:

```bash
python tools/train.py configs/posec3d/slowonly_r50_sails_k400p/joint.py \
    --work-dir work_dirs/posec3d/single/4class_conf04 \
    --cfg-options \
        ann_file=data/sails/single/4class_conf04.pkl \
        data.train.dataset.ann_file=data/sails/single/4class_conf04.pkl \
        data.val.ann_file=data/sails/single/4class_conf04.pkl \
        data.test.ann_file=data/sails/single/4class_conf04.pkl \
        total_epochs=12 \
        optimizer.lr=0.00125 \
    --validate --test-best \
    --launcher none
```

⚠️ **Why all 4 ann_file overrides?**  
The config uses a variable `ann_file` which gets resolved at parse time. Setting only `ann_file=...` does NOT update the nested `data.train.dataset.ann_file`, `data.val.ann_file`, etc. You must override each one explicitly.

### 5-Class Training

For 5 classes, also override `model.cls_head.num_classes`:

```bash
python tools/train.py configs/posec3d/slowonly_r50_sails_k400p/joint.py \
    --work-dir work_dirs/posec3d/single/5class_conf04 \
    --cfg-options \
        ann_file=data/sails/single/5class_conf04.pkl \
        data.train.dataset.ann_file=data/sails/single/5class_conf04.pkl \
        data.val.ann_file=data/sails/single/5class_conf04.pkl \
        data.test.ann_file=data/sails/single/5class_conf04.pkl \
        model.cls_head.num_classes=5 \
        total_epochs=12 \
        optimizer.lr=0.00125 \
    --validate --test-best \
    --launcher none
```

---

## Class-Weighted Training

The SAILS dataset has significant **class imbalance** (e.g., "hands flapping" dominates). We provide a weighted training script that dynamically computes class weights.

### How It Works

```python
# Inverse frequency weighting formula
weight[class_i] = total_samples / (num_classes * class_count[class_i])

# Example for 4-class SAILS:
# Class 0 (hands flapping): 264 samples → weight = 0.45
# Class 1 (jumping):        116 samples → weight = 1.02
# Class 2 (rocking):         57 samples → weight = 2.07
# Class 3 (spinning):        36 samples → weight = 3.28
```

### ⚠️ Important: 4-class vs 5-class Behavior

| Task | Weighting | Result |
|------|-----------|--------|
| **4-class** | Linear (default) | ✅ **Recommended** - 88.5% CV accuracy |
| **5-class** | Linear | ❌ **HURTS** performance - model overfits to minority classes |
| **5-class** | Sqrt scaling | ⚠️ Better than linear, but still experimental |
| **5-class** | None (unweighted) | ✅ **Recommended** - 54.7% Macro F1 vs 34% with linear weights |

### Why 5-class Weighted Fails

With only 36 spinning samples in training, the high weight (2.63×) causes the model to **memorize** minority class-specific patterns that don't generalize:

| Class | Train Samples | Linear Weight | Sqrt Weight |
|-------|---------------|---------------|-------------|
| hands_flapping | 184 | 0.51 | 0.72 |
| jumping | 116 | 0.82 | 0.90 |
| one_hand_flap | 80 | 1.18 | 1.09 |
| rocking | 57 | 1.66 | 1.29 |
| spinning | 36 | **2.63** ← Too aggressive | **1.62** |

**5-class test results:**
| Metric | Unweighted | Weighted (Linear) | Change |
|--------|------------|-------------------|--------|
| Macro F1 | **54.7%** | 34.0% | **-20.7%** 📉 |
| spinning F1 | **75.0%** | 0.0% | **-75.0%** 📉 |
| rocking F1 | **44.4%** | 9.5% | **-34.9%** 📉 |

### Weight Scaling Options

```bash
# For 4-class: use default linear weights
python tools/train_weighted.py ... --weight-scale linear  # default

# For 5-class: use sqrt scaling (milder) or skip weighting entirely
python tools/train_weighted.py ... --weight-scale sqrt    # recommended for 5-class
python tools/train_weighted.py ... --weight-scale none    # no weighting
```

| Scale | Formula | Use Case |
|-------|---------|----------|
| `linear` | `w = total / (n_classes × count)` | **4-class** (default) |
| `sqrt` | `w = sqrt(linear_weight)` | 5-class (milder) |
| `log` | `w = 1 + log(linear_weight)` | Very imbalanced data |
| `none` | `w = 1.0` | Baseline / 5-class fallback |

### Usage

```bash
# 4-class with class weighting (RECOMMENDED)
python tools/train_weighted.py configs/posec3d/slowonly_r50_sails_k400p/joint.py \
    --ann-file data/sails/single/4class_conf04.pkl \
    --work-dir work_dirs/posec3d/single/4class_conf04_weighted \
    --total-epochs 12 \
    --lr 0.00125 \
    --validate --test-best \
    --launcher none

# 5-class with sqrt weight scaling
python tools/train_weighted.py configs/posec3d/slowonly_r50_sails_k400p/joint.py \
    --ann-file data/sails/single/5class_conf04.pkl \
    --work-dir work_dirs/posec3d/single/5class_conf04_weighted \
    --num-classes 5 \
    --weight-scale sqrt \
    --total-epochs 12 \
    --lr 0.00125 \
    --validate --test-best \
    --launcher none
```

### Weighted SLURM Scripts

Submit all weighted training jobs:

```bash
cd /orcd/data/satra/001/users/brukew/actreg/pyskl/scripts/slurm/posec3d
bash submit_all_weighted.sh
```

| Script | Description |
|--------|-------------|
| `train_4class_single_weighted.sh` | 4-class single split with linear weighting |
| `train_5class_single_weighted.sh` | 5-class single split with **sqrt** weighting |
| `train_4class_cv_weighted.sh` | 4-class 3-fold CV with linear weighting |
| `train_5class_cv_weighted.sh` | 5-class 3-fold CV with **sqrt** weighting |

### 4-class Weighted Results (CV)

| Metric | Without Weighting | With Weighting (Linear) |
|--------|-------------------|-------------------------|
| Top-1 Accuracy | 74.6% | **88.5%** ✅ |
| Macro F1 | 70.4% | **88.7%** ✅ |
| Cohen's Kappa | 0.58 | **0.83** ✅ |

Class weighting significantly improves **4-class** performance by preventing the model from ignoring minority classes (rocking, spinning).

---

## Evaluation

### Comprehensive Evaluation Script

```bash
# 4-class evaluation
python tools/evaluate_sails.py \
    configs/posec3d/slowonly_r50_sails_k400p/joint.py \
    -C work_dirs/posec3d/single/4class_conf04/best_top1_acc_epoch_6.pth \
    --split test \
    --output-dir work_dirs/posec3d/single/4class_conf04/eval_test \
    --cfg-options ann_file=data/sails/single/4class_conf04.pkl
```

**For 5-class (MUST pass --class-names):**

> ⚠️ **Critical**: The evaluation script defaults to 4 classes. For 5-class, you **must** pass `--class-names` or the metrics will be wrong (one_hand_flap will be missing).

```bash
python tools/evaluate_sails.py \
    configs/posec3d/slowonly_r50_sails_k400p/joint.py \
    -C work_dirs/posec3d/single/5class_conf04/best_top1_acc_epoch_6.pth \
    --split test \
    --output-dir work_dirs/posec3d/single/5class_conf04/eval_test \
    --class-names "hands flapping" "jumping" "one hand flap" "rocking" "spinning" \
    --cfg-options \
        ann_file=data/sails/single/5class_conf04.pkl \
        model.cls_head.num_classes=5
```

### Metrics Computed

| Metric | Clip-Level | Video-Level |
|--------|------------|-------------|
| Top-1 Accuracy | ✓ | ✓ |
| Top-2 Accuracy | ✓ | ✓ |
| Macro F1 | ✓ | ✓ |
| Weighted F1 | ✓ | ✓ |
| Macro Precision | ✓ | ✓ |
| Macro Recall | ✓ | ✓ |
| Cohen's Kappa | ✓ | ✓ |
| Per-class F1/Prec/Rec | ✓ | ✓ |

### Outputs

| File | Description |
|------|-------------|
| `metrics.json` | All metrics in JSON format |
| `predictions_clip.csv` | Per-clip predictions with scores |
| `predictions_video.csv` | Per-video predictions with scores |
| `confusion_matrix_clip.png` | Clip-level confusion matrix |
| `confusion_matrix_video.png` | Video-level confusion matrix |

### Training vs Evaluation Accuracy

**Important:** You may see different accuracies between training validation and the evaluation script:

| Scenario | Pipeline | Clips/Video |
|----------|----------|-------------|
| Training validation (`--validate`) | `val_pipeline` | 1 clip |
| Evaluation (`--split val`) | `val_pipeline` | 1 clip |
| Evaluation (`--split test`) | `test_pipeline` | 10 clips (TTA) |

Both training validation and `evaluate_sails.py --split val` use the **same** 1-clip pipeline, so they should produce matching results (if using the same data).

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base LR | 0.01 | For 8 GPUs; scale proportionally |
| Batch size | 16 videos/GPU | |
| Epochs | 12 | |
| LR schedule | Step decay @ [9, 11] | |
| Pretrained | K400 PoseC3D | `k400_posec3d-041f49c6.pth` |
| Keypoints | 17 COCO | From wholebody pose (first 17) |
| Min keypoint conf | 0.4 | Below this, coordinates → 0 |
| Clip length | 48 frames | |
| Val clips | 1 | Single sample for speed |
| Test clips | 10 | Averaged for prediction (TTA) |

### GPU Scaling

Scale learning rate linearly with GPU count:

| GPUs | Learning Rate |
|------|---------------|
| 1 | 0.00125 |
| 2 | 0.0025 |
| 4 | 0.005 |
| 8 | 0.01 |

---

## Output Structure

### Single Split
```
work_dirs/posec3d/single/4class_conf04/
├── epoch_1.pth ... epoch_12.pth
├── latest.pth
├── best_top1_acc_epoch_6.pth
├── YYYYMMDD_HHMMSS.log
├── YYYYMMDD_HHMMSS.log.json
├── joint.py                     # Saved config snapshot
└── eval_test/
    ├── metrics.json
    ├── predictions_clip.csv
    ├── predictions_video.csv
    ├── confusion_matrix_clip.png
    └── confusion_matrix_video.png
```

### Cross-Validation
```
work_dirs/posec3d/cv/4class_conf04/
├── fold0/
│   ├── best_top1_acc_epoch_*.pth
│   ├── *.log
│   └── eval_val/
│       └── (metrics, predictions, confusion matrices)
├── fold1/
├── fold2/
└── cv_summary.json              # Aggregated metrics (mean ± std)
```

---

## SLURM Scripts

Located in `scripts/slurm/`:

| Script | Description |
|--------|-------------|
| `submit_all.sh` | Submit all 4 jobs |
| `train_4class_single.sh` | 4-class single split training |
| `train_4class_cv.sh` | 4-class 3-fold CV |
| `train_5class_single.sh` | 5-class single split training |
| `train_5class_cv.sh` | 5-class 3-fold CV |

**SLURM settings:**
- Partition: `mit_preemptable`
- GPUs: 1
- CPUs: 8
- Memory: 32GB
- Time: 12h (single) / 24h (CV)

---

## PyTorch 2.6+ Compatibility

The scripts have been patched to support:
- `--launcher none` for single-GPU non-distributed training/testing
- `torch.load` with `weights_only=False` for mmcv checkpoints
- Dynamic `top-k` accuracy (uses top-2 for ≤5 classes instead of top-5)

---

## Troubleshooting

### "KeyError: 'RANK'" error
Use `--launcher none` for single-GPU training.

### "AttributeError: '_use_replicated_tensor_module'" error  
PyTorch 2.9 + mmcv incompatibility. Use `--launcher none`.

### "WeightsUnpickler error: numpy.dtype" error
Scripts are patched to handle this. Ensure you're using updated `train.py`/`test.py`.

### Training validation differs from expected
Ensure you override ALL ann_file paths:
```bash
--cfg-options \
    ann_file=... \
    data.train.dataset.ann_file=... \
    data.val.ann_file=... \
    data.test.ann_file=...
```

### No pose caches found
Run the annotation script with `--dry-run` to check coverage. Ensure pose HDF5 files are at:
```
/orcd/scratch/bcs/001/sensein/sails/cache_for_tracking/pose_sam3/{video_basename}/{detector}_{thresh}_{pose}_sam3guided.h5
```

### "TypeError: FormatCode() got unexpected keyword argument 'verify'"
Downgrade yapf: `pip install yapf==0.40.1`

### 5-class evaluation shows only 4 classes
You forgot to pass `--class-names`. The evaluation script defaults to 4 classes:
```bash
# WRONG - will use 4-class names
python tools/evaluate_sails.py ... --cfg-options model.cls_head.num_classes=5

# CORRECT - must specify all 5 class names
python tools/evaluate_sails.py ... \
    --class-names "hands flapping" "jumping" "one hand flap" "rocking" "spinning" \
    --cfg-options model.cls_head.num_classes=5
```

### 5-class weighted training performs worse than unweighted
This is expected! Linear class weighting is too aggressive for 5-class with small minority samples. Options:
1. Use `--weight-scale sqrt` for milder weights
2. Use `--weight-scale none` (no weighting) - often performs better
3. Use 4-class instead (more stable)

---

## Training Analysis & Observations

### Overfitting Patterns

From training logs, we observed significant overfitting:

| Metric | Training | Validation | Gap |
|--------|----------|------------|-----|
| Top-1 Accuracy | 100% | ~76% | **24%** |
| Loss | 0.002-0.004 | N/A | N/A |

The model reaches 100% training accuracy by epoch 3-4 but validation accuracy plateaus around 75-77%.

### 4-class Results Comparison

| Setting | Fold 0 | Fold 1 | Fold 2 | Mean |
|---------|--------|--------|--------|------|
| **Unweighted Top-1** | 76.1% | 71.0% | 76.7% | 74.6% |
| **Weighted Top-1** | 92.3% | 88.5% | 84.6% | **88.5%** |
| **Unweighted Macro F1** | 74.5% | 62.5% | 74.2% | 70.4% |
| **Weighted Macro F1** | 95.0% | 89.3% | 81.7% | **88.7%** |

### 5-class Results Comparison (Single Split Test)

| Metric | Unweighted | Weighted (Linear) | Difference |
|--------|------------|-------------------|------------|
| Top-1 Accuracy | 55.3% | 52.4% | -2.9% |
| Macro F1 | **54.7%** | 34.0% | **-20.7%** 📉 |
| spinning F1 | **75.0%** | 0.0% | -75.0% 📉 |
| rocking F1 | **44.4%** | 9.5% | -34.9% 📉 |
| one_hand_flap F1 | 26.7% | 24.4% | -2.3% |

> ⚠️ **Conclusion**: For 5-class, **skip class weighting** or use `--weight-scale sqrt`.

### Recommendations for Improved Training

1. **4-class**: Use class weighting with linear scale (default) - major improvement
2. **5-class**: Use unweighted training OR sqrt weight scaling
3. **Increase dropout** from 0.5 to 0.7:
   ```bash
   --cfg-options model.cls_head.dropout=0.7
   ```
4. **Reduce epochs** to 10 (model plateaus early)
5. **Earlier LR decay** - change step milestones from [9, 11] to [5, 8]
6. **Increase weight decay** from 0.0005 to 0.001:
   ```bash
   --cfg-options optimizer.weight_decay=0.001
   ```

### How Keypoint Confidence is Handled

PoseC3D has a principled approach to keypoint confidence:

```python
# In GeneratePoseTarget (heatmap_related.py)
patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
patch = patch * max_value  # ← Confidence SCALES the Gaussian height!
```

- **High confidence (0.9)** → Tall Gaussian peak → Strong CNN activation
- **Low confidence (0.3)** → Short Gaussian peak → Weak CNN activation  
- **Zero confidence** → Keypoint skipped entirely

This means the model naturally learns to weight keypoints by their confidence, making it robust to noisy pose estimation.
