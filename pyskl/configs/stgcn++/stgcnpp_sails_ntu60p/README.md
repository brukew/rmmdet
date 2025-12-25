# STGCN++ SAILS Finetuning Pipeline

This directory contains configs for finetuning **STGCN++** on the **SAILS RMM** (Repetitive Motor Movement) dataset.

## Model Overview

**STGCN++** is a Graph Convolutional Network (GCN) for skeleton-based action recognition. Unlike PoseC3D which uses pseudo-heatmaps with 3D CNNs, STGCN++ directly operates on skeleton graphs where:
- **Nodes** = body keypoints (17 COCO keypoints)
- **Edges** = natural skeletal connections
- **Features** = (x, y, confidence) per keypoint per frame

### Key Architecture Features

| Feature | Description |
|---------|-------------|
| `gcn_adaptive='init'` | Learnable adjacency matrix (adapts graph structure) |
| `gcn_with_res=True` | Enhanced residual connections |
| `tcn_type='mstcn'` | Multi-scale temporal convolution |
| `layout='coco'` | COCO-17 keypoint graph |

### Comparison with PoseC3D

| Aspect | PoseC3D | STGCN++ |
|--------|---------|---------|
| Input | Pseudo-heatmaps (56×56 images) | Graph (17 nodes × 3 channels) |
| Backbone | 3D CNN (SlowOnly) | GCN + TCN |
| Frame sampling | 48 frames | 100 frames |
| Memory | Higher (~4-8 GB) | Lower (~1 GB) |
| Speed | Slower (~2-3 min/epoch) | **Much faster (~10-20 sec/epoch)** |
| Pretrained | Kinetics-400 (video) | NTU RGB+D (skeleton) |

### How Keypoint Confidence is Handled

The two models handle keypoint confidence **very differently**:

#### PoseC3D: Confidence = Heatmap Intensity (Built-in)

```python
# In GeneratePoseTarget (heatmap_related.py)
patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
patch = patch * max_value  # ← Confidence SCALES the Gaussian height!
```

- **0.9 confidence** → tall Gaussian peak → strong CNN activation
- **0.3 confidence** → short Gaussian peak → weak CNN activation
- **0.0 confidence** → completely skipped (`if max_value < EPS: continue`)

#### STGCN++: Confidence = Just Another Input Feature (Learned)

```python
# In FormatGCNInput (pose_related.py)
keypoint = np.concatenate((keypoint, keypoint_score[..., None]), axis=-1)
# Shape: (M, T, V, 3) where 3 = (x, y, confidence)
```

- Confidence is input channel #3 alongside x and y
- **Not** used to weight graph edges or attention
- The network must **learn** that low confidence means "ignore"

#### Practical Implications

| Aspect | PoseC3D | STGCN++ |
|--------|---------|---------|
| Low-conf keypoint effect | Weak heatmap → minimal CNN influence | Full-strength graph node |
| How model handles conf | **Built into representation** | **Must be learned** |
| Risk from noisy keypoints | Lower (weak signals) | Higher (unless learned) |

#### Keypoint Filtering Pipeline

Both models use the **same pre-filtered pickle files** (filtered at 0.4 confidence):

```
Pickle Creation (--min-keypoint-conf 0.4):
  if conf < 0.4: x=0, y=0, conf=0

STGCN++ Runtime (PreNormalize2D, threshold=0.01):
  if conf <= 0.01: x=0, y=0  (redundant since pickle already filtered)

PoseC3D Runtime (GeneratePoseTarget):
  if conf < EPS: skip heatmap generation
```

**Bottom line**: PoseC3D has a more principled approach to confidence (scaled signal strength), while STGCN++ relies on learning that `(0, 0, 0)` means "missing keypoint." This may make PoseC3D more robust to noisy pose estimation.

### Pipeline Validation (1 epoch test)

| Modality | Top-1 Accuracy | Top-2 Accuracy |
|----------|----------------|----------------|
| Joint (j) | 61.54% | 88.46% |
| Bone (b) | 48.08% | 78.85% |
| Joint Motion (jm) | 28.85% | 75.00% |
| Bone Motion (bm) | 40.38% | 86.54% |
| **4-Stream Fusion** | 50.00% | 76.92% |

> **Note**: With only 1 epoch, motion modalities are severely undertrained. With full 24-epoch training, expect all modalities to converge to ~85-90% and fusion to provide +2-5% improvement over best single stream.

---

## Class-Weighted Training

The SAILS dataset has significant **class imbalance**. We provide weighted training that dynamically computes inverse-frequency class weights per-fold.

### ⚠️ Important: STGCN++ vs PoseC3D Weighting Behavior

**STGCN++ behaves OPPOSITE to PoseC3D regarding class weighting:**

| Task | PoseC3D | STGCN++ |
|------|---------|---------|
| **4-class** | ✅ Weighting helps (+13.9% Top-1) | ❌ **Weighting hurts** (-2.6% Top-1) |
| **5-class** | ❌ Weighting hurts (-20% Macro F1) | ≈ **Neutral/slight benefit** (+2% Video Macro F1) |

### Why the Different Behavior?

| Factor | PoseC3D | STGCN++ |
|--------|---------|---------|
| **Architecture** | 3D CNN (high capacity, ~23M params) | GCN + TCN (lower capacity, ~1-2M params) |
| **Regularization** | Needs class weighting to avoid ignoring minorities | Already regularized by graph structure |
| **Training stability** | More stable | Less stable (validation swings 8-10%) |

### Recommended Settings

| Task | Recommendation |
|------|----------------|
| **4-class** | **Skip weighting** - unweighted is better |
| **5-class** | Use `--weight-scale sqrt` for slight benefit |

### 4-class CV Results Comparison

| Metric | Unweighted | Weighted | Change |
|--------|------------|----------|--------|
| Clip Top-1 | **77.9%** | 75.3% | -2.6% 📉 |
| Clip Macro F1 | 72.7% | 72.9% | +0.2% |
| Video Top-1 | **82.4%** | 79.5% | -2.9% 📉 |
| Video Macro F1 | 75.9% | 76.2% | +0.3% |
| Cohen κ | **0.670** | 0.646 | -0.024 📉 |

### 5-class CV Results Comparison

| Metric | Unweighted | Weighted | Change |
|--------|------------|----------|--------|
| Clip Top-1 | 65.9% | 65.1% | -0.8% |
| Clip Macro F1 | 63.8% | 64.2% | +0.4% |
| Video Top-1 | 68.7% | 68.9% | +0.2% |
| Video Macro F1 | 64.8% | **66.9%** | +2.1% ✅ |
| Cohen κ | 0.553 | 0.554 | +0.001 |

### Single Split Weighted Results (Test Set)

| Task | Clip Top-1 | Clip Macro F1 | Video Top-1 | Video Macro F1 |
|------|------------|---------------|-------------|----------------|
| 4-class weighted | 62.1% | 58.1% | 75.8% | 68.5% |
| 5-class weighted | 53.4% | 49.8% | 54.8% | 48.9% |

**Modality breakdown (4-class weighted single):**
| Modality | Accuracy |
|----------|----------|
| Joint (j) | 56.3% |
| Bone (b) | 56.3% |
| Joint Motion (jm) | **69.9%** ← Best single |
| Bone Motion (bm) | 60.2% |
| 4-stream fusion | 62.1% |

> ⚠️ **Problem**: Joint motion alone (69.9%) outperforms 4-stream fusion (62.1%)! Weighted training destabilizes the fusion for STGCN++.

### Weight Scaling Options

```bash
# For 4-class: skip weighting (use unweighted training)
python tools/train.py ...  # No class weighting

# For 5-class: use sqrt scaling for slight benefit
python tools/train_weighted.py ... --weight-scale sqrt
```

| Scale | Formula | Use Case |
|-------|---------|----------|
| `linear` | `w = total / (n_classes × count)` | **Not recommended for STGCN++** |
| `sqrt` | `w = sqrt(linear_weight)` | 5-class (milder) |
| `none` | `w = 1.0` | **4-class default (recommended)** |

### How It Works

```python
# Inverse frequency weighting formula
weight[class_i] = total_samples / (num_classes * class_count[class_i])

# Example for 4-class SAILS (linear):
# Class 0 (hands flapping): 264 samples → weight = 0.45
# Class 1 (jumping):        116 samples → weight = 1.02
# Class 2 (rocking):         57 samples → weight = 2.07
# Class 3 (spinning):        36 samples → weight = 3.28
```

### Weighted SLURM Scripts

> Note: For 4-class, we recommend using the **unweighted** scripts instead.

```bash
cd /orcd/data/satra/001/users/brukew/actreg/pyskl/scripts/slurm/stgcnpp
bash submit_all_weighted.sh  # Runs weighted versions
bash submit_all.sh           # Runs unweighted (recommended for 4-class)
```

| Script | Description | Recommended? |
|--------|-------------|--------------|
| `train_4class_single_4stream.sh` | 4-class unweighted | ✅ Yes |
| `train_4class_single_4stream_weighted.sh` | 4-class weighted | ❌ No |
| `train_5class_single_4stream_weighted.sh` | 5-class weighted (sqrt) | ⚠️ Experimental |
| `train_4class_cv_4stream.sh` | 4-class CV unweighted | ✅ Yes |
| `train_5class_cv_4stream_weighted.sh` | 5-class CV weighted (sqrt) | ⚠️ Experimental |

---

## Modalities

We train **4 modalities** that are fused for best performance:

### Joint (j)
Raw keypoint coordinates: `(x, y, confidence)` per keypoint per frame.
- Captures: Absolute body pose positions
- File: `j.py`

### Bone (b)  
Vectors between connected joints: `bone[child] = keypoint[child] - keypoint[parent]`
- Captures: Limb orientations (position-invariant)
- File: `b.py`

### Joint Motion (jm)
Temporal velocity of joints: `jm[t] = joint[t] - joint[t-1]`
- Captures: How fast each keypoint is moving
- File: `jm.py`

### Bone Motion (bm)
Temporal velocity of bones: `bm[t] = bone[t] - bone[t-1]`
- Captures: Rotational movements, limb dynamics
- File: `bm.py`

### Visual Summary

```
JOINT (j)              BONE (b)
┌─────────────┐        ┌─────────────┐
│ (x,y) coords│        │  vectors    │
│   ●──●──●   │        │   → → →     │
│   │        │        │   │        │
│   ●   ●    │        │   ↓   ↓    │
└─────────────┘        └─────────────┘
       │                      │
       ▼ velocity             ▼ velocity
JOINT MOTION (jm)      BONE MOTION (bm)
┌─────────────┐        ┌─────────────┐
│ Δ position  │        │  Δ vectors  │
│ (velocity)  │        │ (rot. vel.) │
└─────────────┘        └─────────────┘
```

### 4-Stream Fusion
After training all modalities, predictions are fused with weighted averaging:
```python
# Late fusion with 2:2:1:1 weighting (j:b:jm:bm)
final_score = (2*joint + 2*bone + 1*joint_motion + 1*bone_motion) / 6
```

## Pretrained Weights

We use **NTU60 HRNet** pretrained weights because:
1. **Skeleton format matches**: Both use COCO-17 2D keypoints
2. **Domain similarity**: Human body movements
3. **Proven performance**: 89-92% on NTU benchmarks

Checkpoints are downloaded to:
```
checkpoints/stgcnpp/
├── ntu60_hrnet_j.pth   # Joint modality
├── ntu60_hrnet_b.pth   # Bone modality
├── ntu60_hrnet_jm.pth  # Joint Motion modality
└── ntu60_hrnet_bm.pth  # Bone Motion modality
```

## Dataset Configurations

### 4-Class Setup
| Class | Label |
|-------|-------|
| hands flapping | 0 |
| jumping | 1 |
| rocking | 2 |
| spinning | 3 |

### 5-Class Setup
| Class | Label |
|-------|-------|
| hands flapping | 0 |
| jumping | 1 |
| one hand flap | 2 |
| rocking | 3 |
| spinning | 4 |

## Quick Start

### 1. Test Configuration

```bash
cd /orcd/data/satra/001/users/brukew/actreg/pyskl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

# Quick validation run (1 epoch)
python tools/train.py configs/stgcn++/stgcnpp_sails_ntu60p/j.py \
    --work-dir work_dirs/stgcnpp/test_run \
    --cfg-options total_epochs=1 data.train.times=1 \
    --validate --launcher none
```

### 2. Single Split Training (Interactive)

```bash
# 4-class, joint modality
python tools/train.py configs/stgcn++/stgcnpp_sails_ntu60p/j.py \
    --work-dir work_dirs/stgcnpp/single/4class_conf04_j \
    --cfg-options \
        data.train.dataset.ann_file=data/sails/single/4class_conf04.pkl \
        data.val.ann_file=data/sails/single/4class_conf04.pkl \
        data.test.ann_file=data/sails/single/4class_conf04.pkl \
    --validate --test-best \
    --launcher none

# 4-class, bone modality
python tools/train.py configs/stgcn++/stgcnpp_sails_ntu60p/b.py \
    --work-dir work_dirs/stgcnpp/single/4class_conf04_b \
    --cfg-options \
        data.train.dataset.ann_file=data/sails/single/4class_conf04.pkl \
        data.val.ann_file=data/sails/single/4class_conf04.pkl \
        data.test.ann_file=data/sails/single/4class_conf04.pkl \
    --validate --test-best \
    --launcher none
```

### 3. SLURM Batch Submission

```bash
# Submit all 4-stream jobs (4 total: each trains j, b, jm, bm + fusion)
bash scripts/slurm/stgcnpp/submit_all.sh

# Or submit individually
sbatch scripts/slurm/stgcnpp/train_4class_single_4stream.sh  # 4-class single split
sbatch scripts/slurm/stgcnpp/train_5class_single_4stream.sh  # 5-class single split  
sbatch scripts/slurm/stgcnpp/train_4class_cv_4stream.sh      # 4-class 3-fold CV
sbatch scripts/slurm/stgcnpp/train_5class_cv_4stream.sh      # 5-class 3-fold CV

# Monitor jobs
squeue -u $USER
```

Each 4-stream job:
1. Trains Joint (j) → 24 epochs + evaluation
2. Trains Bone (b) → 24 epochs + evaluation
3. Trains Joint Motion (jm) → 24 epochs + evaluation
4. Trains Bone Motion (bm) → 24 epochs + evaluation
5. Runs 4-stream weighted fusion with comprehensive metrics

### 4. Evaluation

```bash
# Evaluate a trained model
python tools/evaluate_sails.py configs/stgcn++/stgcnpp_sails_ntu60p/j.py \
    -C work_dirs/stgcnpp/single/4class_conf04_j/best_top1_acc_epoch_*.pth \
    --split test \
    --output-dir work_dirs/stgcnpp/single/4class_conf04_j/eval_test \
    --cfg-options \
        data.train.dataset.ann_file=data/sails/single/4class_conf04.pkl \
        data.val.ann_file=data/sails/single/4class_conf04.pkl \
        data.test.ann_file=data/sails/single/4class_conf04.pkl
```

## Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Pretrained | NTU60 HRNet | COCO-17 2D skeleton |
| Epochs | 24 | More than NTU default (16) for small dataset |
| Batch size | 16 | Per GPU |
| Learning rate | 0.01 | With cosine annealing |
| Clip length | 100 frames | GCN default |
| RepeatDataset | 10× | Increases iterations per epoch |

## Output Structure

```
work_dirs/stgcnpp/
├── single/
│   ├── 4class_conf04_j/       # Joint modality
│   │   ├── epoch_*.pth
│   │   ├── best_top1_acc_epoch_*.pth
│   │   └── eval_test/
│   │       ├── metrics.json
│   │       ├── predictions_clip.csv
│   │       ├── predictions_video.csv
│   │       ├── confusion_matrix_clip.png
│   │       └── confusion_matrix_video.png
│   ├── 4class_conf04_b/       # Bone modality (same structure)
│   ├── 4class_conf04_jm/      # Joint Motion modality
│   ├── 4class_conf04_bm/      # Bone Motion modality
│   ├── 4class_conf04_4stream/ # 4-stream fusion results
│   │   └── eval_test/
│   │       ├── metrics.json           # All metrics (clip + video level)
│   │       ├── predictions_clip.csv   # Per-clip predictions + scores
│   │       ├── predictions_video.csv  # Per-video predictions + scores
│   │       ├── confusion_matrix_clip.png
│   │       └── confusion_matrix_video.png
│   └── 5class_conf04_*/       # Same structure for 5-class
└── cv/
    ├── 4class_conf04_j/
    │   ├── fold0/
    │   ├── fold1/
    │   └── fold2/
    ├── 4class_conf04_b/
    ├── 4class_conf04_jm/
    ├── 4class_conf04_bm/
    ├── 4class_conf04_4stream/
    │   ├── fold0/                     # Per-fold fusion results
    │   │   ├── metrics.json
    │   │   ├── predictions_clip.csv
    │   │   ├── predictions_video.csv
    │   │   ├── confusion_matrix_clip.png
    │   │   └── confusion_matrix_video.png
    │   ├── fold1/
    │   ├── fold2/
    │   └── cv_summary.json            # Aggregated metrics (mean ± std)
    └── 5class_conf04_*/       # Same structure for 5-class
```

## Metrics Reported

The evaluation scripts (both individual modality and 4-stream fusion) report comprehensive metrics:

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

### Outputs Generated

| File | Description |
|------|-------------|
| `metrics.json` | All metrics in JSON format |
| `predictions_clip.csv` | Per-clip predictions with class scores |
| `predictions_video.csv` | Per-video predictions (majority vote) |
| `confusion_matrix_clip.png` | Clip-level confusion matrix |
| `confusion_matrix_video.png` | Video-level confusion matrix |

### CV Summary Format

For cross-validation, `cv_summary.json` contains:
- `per_fold`: Array of per-fold metrics
- `{metric}_mean`: Mean across folds
- `{metric}_std`: Standard deviation across folds

## 4-Stream Fusion (Post-Training)

The SLURM scripts automatically run 4-stream fusion after training all modalities. The fusion uses weighted averaging:

```python
import torch
import numpy as np

# Weights: j=2, b=2, jm=1, bm=1
weights = [2, 2, 1, 1]
modalities = ['j', 'b', 'jm', 'bm']

# Load scores from each modality
all_scores = []
for mod in modalities:
    # Run inference and collect softmax scores
    scores = model_inference(mod_checkpoint)  
    all_scores.append(scores)

# Weighted fusion
fused_scores = sum(w * s for w, s in zip(weights, all_scores)) / sum(weights)
fused_preds = np.argmax(fused_scores, axis=1)

# Calculate accuracy
accuracy = (fused_preds == labels).mean()
print(f"4-Stream Fused Top-1 Accuracy: {accuracy:.2%}")
```

### Why 2:2:1:1 Weights?
- **Joint & Bone** (weight=2): Primary modalities, capture pose and structure
- **Motion modalities** (weight=1): Complementary, capture dynamics
- This weighting is standard in pyskl's multi-stream benchmarks

## Logs

SLURM logs are saved to:
```
/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/
├── 4class_single_4stream_*.out   # 4-class single split (all 4 modalities)
├── 4class_single_4stream_*.err
├── 5class_single_4stream_*.out   # 5-class single split
├── 4class_cv_4stream_*.out       # 4-class 3-fold CV
├── 5class_cv_4stream_*.out       # 5-class 3-fold CV
└── ...
```

### Estimated Training Times (per job)

| Job Type | Modalities | Folds | Estimated Time |
|----------|------------|-------|----------------|
| Single split | 4 (j,b,jm,bm) | 1 | ~30-40 min |
| 3-fold CV | 4 (j,b,jm,bm) | 3 | ~2-3 hours |

## Training Analysis & Observations

### Overfitting Patterns

From training logs, we observed significant overfitting with STGCN++:

| Metric | Training | Validation | Gap |
|--------|----------|------------|-----|
| Top-1 Accuracy | 100% | ~65-75% | **25-35%** |
| Loss | 0.0004-0.002 | N/A | N/A |

The model reaches 100% training accuracy very early but validation accuracy is unstable.

### Best Results Summary

| Task | Method | Video Top-1 | Video Macro F1 | Recommendation |
|------|--------|-------------|----------------|----------------|
| 4-class CV | Unweighted | **82.4%** | **75.9%** | ✅ Use this |
| 4-class CV | Weighted | 79.5% | 76.2% | ❌ Skip |
| 5-class CV | Unweighted | 68.7% | 64.8% | ✅ Baseline |
| 5-class CV | Weighted (sqrt) | 68.9% | **66.9%** | ⚠️ Slight improvement |

### Key Issues

1. **Unstable validation** - Accuracy swings 8-10% between epochs
2. **Early peak** - Best performance often at epochs 4-8
3. **No built-in confidence handling** - Unlike PoseC3D, confidence is just an input feature
4. **Weighted training destabilizes fusion** - Joint motion alone can outperform 4-stream fusion when using weights

### Recommendations for Improved Training

1. **For 4-class: Skip class weighting** - unweighted performs better
2. **For 5-class: Use `--weight-scale sqrt`** for milder weights
3. **Add dropout** (currently 0!):
   ```bash
   --cfg-options model.cls_head.dropout=0.5
   ```
4. **Reduce epochs** from 24 to 16
5. **Lower learning rate** from 0.01 to 0.005:
   ```bash
   --cfg-options optimizer.lr=0.005
   ```
6. **Increase weight decay** from 0.0005 to 0.001:
   ```bash
   --cfg-options optimizer.weight_decay=0.001
   ```
7. **Consider early stopping** based on validation accuracy (patience ~5 epochs)

### Data Augmentation (Future Work)

STGCN++ would benefit from skeleton-specific augmentations:
- Random rotation
- Random scaling
- Random translation
- Gaussian noise on coordinates

---

## Troubleshooting

### 1. CUDA Out of Memory
Reduce batch size:
```bash
--cfg-options data.videos_per_gpu=8
```

### 2. Slow Training
Increase workers:
```bash
--cfg-options data.workers_per_gpu=4
```

### 3. PyTorch 2.6+ Checkpoint Loading
If you see `weights_only` errors, the training script includes a monkey-patch for compatibility.

### 4. Validation Accuracy Unstable
This is expected with small datasets. Recommendations:
- Use class weighting
- Add dropout (0.5)
- Consider early stopping
- Use more aggressive regularization

## Citation

```bibtex
@inproceedings{duan2022pyskl,
  title={Pyskl: Towards good practices for skeleton action recognition},
  author={Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={7351--7354},
  year={2022}
}
```


