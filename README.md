# SAILS RMM Action Recognition

Automated classification **and temporal localization** of **Repetitive Motor Movements (RMM)** in video from the SAILS dataset. The repo covers two tasks:

1. **Clip classification** — given a trimmed clip, predict the RMM class. Approaches: skeleton-based CNN (PoseC3D), skeleton-based GCN (STGCN++), video encoder finetuning (V-JEPA2), late fusion, and zero-shot VLM (Qwen2.5-VL).
2. **Temporal Action Localization (TAL)** — given an untrimmed video, find *when* RMMs occur and classify them. Approaches: window-based scoring + postprocessing, OpenTAD end-to-end detectors (ActionFormer/TriDet), and a two-stage (binary detector → classifier) pipeline.

> **New here / picking this up?** Start with [`REPRODUCE.md`](REPRODUCE.md) for the pipeline
> and expected numbers, [`docs/ENTRYPOINTS.md`](docs/ENTRYPOINTS.md) for **which script to
> run** (env, args, inputs, outputs), [`docs/INDEX.md`](docs/INDEX.md) for the full file
> catalog, and [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) for where data/checkpoints live.

---

## Task Overview

### Classification Target

Classify short video clips into one of the following RMM categories:

| Class | Description |
|-------|-------------|
| `hands flapping` | Repetitive movement of hands at the wrists (vertical or horizontal) |
| `one hand flap` | Single hand flapping motion (excluded in 4-class variant) |
| `rocking` | Front-to-back or side-to-side body movement |
| `spinning` | Turning the body in circular motion |
| `jumping` | Repetitive bouncing with knees bending, feet leaving/nearly leaving floor |

### Dataset

| Property | Value |
|----------|-------|
| Total clips | 654 |
| Total LCTO groups | 151 |
| Class variants | 4-class (excluding `one hand flap`) and 5-class |

**LCTO Groups**: Leave-Cohort-Time-Out groups represent unique subject-timepoint combinations (e.g., `A1H3H9Y3T1_36_month`). Cross-validation splits are performed at the LCTO group level to prevent data leakage between train and validation sets.

### Data Splits

| Split Type | Train | Val | Test | Description |
|------------|-------|-----|------|-------------|
| **Single** | 489 clips (105 groups) | 54 clips (18 groups) | 111 clips (28 groups) | Train/val/test split |
| **CV** (per fold) | 436 clips (~101 groups) | 218 clips (~50 groups) | — | 3-fold cross-validation |

### Class Distribution (4-class)

| Class | Train Samples | Percentage | Weight |
|-------|---------------|------------|--------|
| hands flapping | 357 | 55.9% | 0.45 |
| jumping | 157 | 24.6% | 1.02 |
| rocking | 77 | 12.1% | 2.07 |
| spinning | 49 | 7.7% | 3.28 |

> ⚠️ **Note**: Significant class imbalance with "hands flapping" dominating. Class-weighted training is recommended.

---

## Class-Weighted Training

Both skeleton-based methods (PoseC3D, STGCN++) support **inverse-frequency class weighting** to address class imbalance:

```python
# Weight formula (computed automatically per-fold for CV)
weight[class_i] = total_samples / (num_classes * class_count[class_i])
```

### Usage

```bash
cd pyskl

# PoseC3D with class weighting
python tools/train_weighted.py configs/posec3d/slowonly_r50_sails_k400p/joint.py \
    --ann-file data/sails/single/4class_conf04.pkl \
    --work-dir work_dirs/posec3d/single/4class_conf04_weighted \
    --validate --launcher none

# STGCN++ with class weighting  
python tools/train_weighted.py configs/stgcn++/stgcnpp_sails_ntu60p/j.py \
    --ann-file data/sails/single/4class_conf04.pkl \
    --work-dir work_dirs/stgcnpp/single/4class_conf04_weighted_j \
    --validate --launcher none
```

### Impact on Performance

| Metric | Without Weighting | With Weighting |
|--------|-------------------|----------------|
| Top-1 Accuracy | ~65-77% | ~75-82% |
| **Mean Class Accuracy** | ~62-73% | **~79-87%** |

Class weighting significantly improves **mean class accuracy** by preventing the model from ignoring minority classes (rocking, spinning).

---

## Methods

### 1. PoseC3D (Skeleton-Based)

**Directory:** `pyskl/`

Uses pose/skeleton data as input, processed through a 3D CNN backbone. This approach focuses on body keypoint movements rather than RGB appearance.

| Component | Details |
|-----------|---------|
| **Backbone** | SlowOnly R50 (3D CNN) |
| **Input** | 17 COCO keypoints per frame (from wholebody pose estimation) |
| **Pretrained** | Kinetics-400 PoseC3D (`k400_posec3d-041f49c6.pth`) |
| **Framework** | [pyskl](https://github.com/kennymckormick/pyskl) (fork of mmaction2) |
| **Clip length** | 48 frames |
| **Keypoint confidence** | Threshold ≥ 0.4 (below → zero coordinates) |

**Key hyperparameters:**
- Learning rate: 0.00125 (for 1 GPU; scales linearly with GPU count)
- Batch size: 16 videos/GPU
- Epochs: 12
- LR schedule: Step decay @ [9, 11]
- Test-time augmentation: 10 clips per video

**Class-Weighted Training (Recommended):**

The SAILS dataset has significant class imbalance. Use `train_weighted.py` for automatic inverse-frequency weighting:

```bash
cd pyskl
# With class weighting (recommended)
bash scripts/slurm/posec3d/submit_all_weighted.sh

# Without class weighting (baseline)
bash scripts/slurm/submit_all.sh
```

| Without Weighting | With Weighting |
|-------------------|----------------|
| Mean Class Acc: ~62-73% | Mean Class Acc: **~79-87%** |

See [`pyskl/configs/posec3d/slowonly_r50_sails_k400p/README.md`](pyskl/configs/posec3d/slowonly_r50_sails_k400p/README.md) for detailed instructions.

**Results (4-class CV):**

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Acc | Video Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-----------|----------------|-----------|
| 0 | 76.1% | 91.5% | 74.5% | 76.1% | 74.5% | 0.595 |
| 1 | 71.0% | 90.8% | 62.5% | 71.0% | 62.5% | 0.528 |
| 2 | 76.7% | 91.5% | 74.2% | 76.7% | 74.2% | 0.618 |
| **Mean** | **74.6%** | **91.3%** | **70.4%** | **74.6%** | **70.4%** | **0.580** |

**Results (5-class CV):**

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-----------|
| 0 | 67.4% | 82.7% | 63.2% | 0.556 |
| 1 | 65.4% | 84.8% | 64.4% | 0.531 |
| 2 | 67.5% | 86.3% | 67.8% | 0.554 |
| **Mean** | **66.8%** | **84.6%** | **65.1%** | **0.547** |

---

### 2. STGCN++ (Skeleton-Based GCN)

**Directory:** `pyskl/`

Graph Convolutional Network for skeleton-based action recognition. Unlike PoseC3D which uses pseudo-heatmaps with 3D CNNs, STGCN++ directly operates on skeleton graphs.

| Component | Details |
|-----------|---------|
| **Architecture** | GCN + Multi-scale TCN |
| **Input** | 17 COCO keypoints as graph nodes (x, y, confidence) |
| **Pretrained** | NTU RGB+D 60 (HRNet skeleton) |
| **Framework** | [pyskl](https://github.com/kennymckormick/pyskl) |
| **Clip length** | 100 frames |
| **Modalities** | 4-stream fusion (joint, bone, joint-motion, bone-motion) |

**Key differences from PoseC3D:**

| Aspect | PoseC3D | STGCN++ |
|--------|---------|---------|
| Input | Pseudo-heatmaps (56×56) | Graph (17 nodes × 3 channels) |
| Backbone | 3D CNN (SlowOnly) | GCN + TCN |
| Memory | Higher (~4-8 GB) | Lower (~1 GB) |
| Speed | Slower (~2-3 min/epoch) | **Much faster (~10-20 sec/epoch)** |
| Confidence handling | Built into heatmap intensity | Input feature (learned) |

**4-Stream Fusion:**
```
Final = (2×Joint + 2×Bone + 1×JointMotion + 1×BoneMotion) / 6
```

**Class-Weighted Training (Recommended):**

```bash
cd pyskl
# With class weighting (recommended)
bash scripts/slurm/stgcnpp/submit_all_weighted.sh

# Without class weighting (baseline)
bash scripts/slurm/stgcnpp/submit_all.sh
```

| Without Weighting | With Weighting |
|-------------------|----------------|
| Mean Class Acc: ~64-73% | Mean Class Acc: **~79-85%** |

See [`pyskl/configs/stgcn++/stgcnpp_sails_ntu60p/README.md`](pyskl/configs/stgcn++/stgcnpp_sails_ntu60p/README.md) for detailed instructions.

**Results (4-class CV, 4-stream fusion):**

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Acc | Video Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-----------|----------------|-----------|
| 0 | 81.0% | 96.1% | 78.7% | 80.8% | 76.3% | 0.681 |
| 1 | 76.5% | 94.5% | 67.7% | 81.1% | 71.3% | 0.607 |
| 2 | 76.2% | 94.7% | 71.8% | 85.2% | 80.2% | 0.616 |
| **Mean** | **77.9%** | **95.1%** | **72.7%** | **82.4%** | **75.9%** | **0.635** |

**Results (5-class CV, 4-stream fusion):**

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-----------|
| 0 | 65.9% | 82.0% | 62.9% | 0.525 |
| 1 | 67.7% | 85.3% | 65.3% | 0.556 |
| 2 | 64.1% | 82.5% | 63.1% | 0.515 |
| **Mean** | **65.9%** | **83.2%** | **63.8%** | **0.532** |

---

### 3. V-JEPA2 (Video Encoder Finetuning)

**Directory:** `v-jepa/`

Finetunes Meta's V-JEPA2 video encoder (ViT-L) for classification. V-JEPA is a self-supervised video model trained with joint-embedding predictive architecture.

| Component | Details |
|-----------|---------|
| **Backbone** | ViT-L (from V-JEPA2) |
| **Model** | `facebook/vjepa2-vitl-fpc16-256-ssv2` |
| **Framework** | HuggingFace Transformers |
| **Frames per clip** | 64 (best from sweep) |
| **Optional** | SAM3-based person cropping |

**Best hyperparameters (from sweep):**
- Learning rate: 1e-5
- Batch size: 1
- Gradient accumulation: 8 steps
- Epochs: 20
- Frames per clip: 64

**Run configurations:**
| Script | Description |
|--------|-------------|
| `finetune_sails_vjepa2_cv.py` | 3-fold cross-validation (5-class) |
| `finetune_sails_vjepa2_cv_crop.py` | CV with SAM3-based person cropping |
| `finetune_sails_vjepa2_single.py` | Single split (train/val/test) |

**Usage:**
```bash
cd v-jepa
sbatch slurm/run_vjepa_cv.sh              # With cropping (default)
sbatch --export=ENABLE_CROP=0 slurm/run_vjepa_cv.sh  # Without cropping
```

**Results (4-class CV with cropping):**

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Acc | Video Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-----------|----------------|-----------|
| 0 | 80.6% | 93.5% | 76.7% | 83.0% | 81.7% | 0.724 |
| 1 | 73.4% | 92.2% | 72.5% | 71.3% | 65.3% | 0.523 |
| 2 | 73.9% | 95.4% | 76.2% | 81.6% | 83.2% | 0.723 |
| **Mean** | **76.0%** | **93.7%** | **75.1%** | **78.6%** | **76.7%** | **0.657** |

**Results (5-class CV with cropping):**

| Fold | Clip Top-1 | Clip Top-2 | Clip Macro F1 | Video Acc | Video Macro F1 | Cohen's κ |
|------|------------|------------|---------------|-----------|----------------|-----------|
| 0 | 65.4% | 85.7% | 64.5% | 64.0% | 63.3% | 0.503 |
| 1 | 60.6% | 85.8% | 61.8% | 66.0% | 66.0% | 0.553 |
| 2 | 61.5% | 83.0% | 62.8% | 57.3% | 59.1% | 0.442 |
| **Mean** | **62.5%** | **84.8%** | **63.0%** | **62.4%** | **62.8%** | **0.499** |

---

### 4. Qwen2.5-VL (Zero-Shot VLM)

**Directory:** `../sailsprep/pred2annot/rmm/qwen/`

Zero-shot classification using Qwen2.5-VL-7B vision-language model. No training required—uses carefully designed prompts to classify RMM behaviors.

| Component | Details |
|-----------|---------|
| **Model** | `Qwen/Qwen2.5-VL-7B-Instruct` |
| **Approach** | Zero-shot prompting |
| **Inference** | Multi-window voting (4 windows × 16 frames each) |
| **Aggregation** | Majority voting across windows → pseudo-probability scores |

**Usage:**
```bash
cd ../sailsprep/pred2annot/rmm/qwen
sbatch scripts/run_cv_5class.sh   # 5-class CV
sbatch scripts/run_single_4class.sh  # 4-class single split
```

See [`../sailsprep/pred2annot/rmm/qwen/README.md`](../sailsprep/pred2annot/rmm/qwen/README.md) for detailed instructions.

---

### 5. Late Fusion (V-JEPA2 + PoseC3D)

**Directory:** `fusion/`

Learned late fusion combining V-JEPA2 (RGB) and PoseC3D (skeleton) predictions using a single scalar weight α.

| Component | Details |
|-----------|---------|
| **Architecture** | `z_fused = σ(α) * z_rgb + (1 - σ(α)) * z_pose` |
| **Learnable params** | 1 (scalar α, sigmoid-constrained) |
| **Encoders** | Frozen (pre-computed predictions) |
| **Loss** | CrossEntropy with inverse-frequency class weights |
| **Training** | ~100 epochs, Adam lr=0.01 |

**Motivation**: V-JEPA2 captures RGB appearance (clothing, background), while PoseC3D focuses on body movement patterns. Fusion can leverage complementary strengths.

**How it works**:
1. Load pre-computed clip-level predictions from both models
2. Train a single scalar α that weights the two modality logits
3. α is sigmoid-constrained to [0, 1] for interpretability
4. α = 0.5 means equal weighting; α > 0.5 means more V-JEPA2 weight

**Usage:**
```bash
cd <repo root>

# Run 3-fold CV fusion training
sbatch fusion/slurm/run_fusion_cv.sh

# Or run directly
python fusion/train_fusion_cv.py \
    --vjepa-root v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls \
    --posec3d-root pyskl/work_dirs/posec3d/cv/4class_conf04_weighted \
    --output-dir fusion/runs/4class_cv \
    --num-epochs 100 --lr 0.01
```

**Results (4-class CV):**

| Fold | Clip Top-1 | Clip Macro F1 | Video Top-1 | Video Macro F1 | α |
|------|------------|---------------|-------------|----------------|---|
| 0 | 85.8% | 82.7% | 85.7% | 82.6% | 0.65 |
| 1 | 73.3% | 71.1% | 72.0% | 66.3% | 0.88 |
| 2 | 80.5% | 82.6% | 84.2% | 83.8% | 0.59 |
| **Mean** | **79.8%** | **78.8%** | **80.6%** | **77.6%** | **0.71** |

**Logs**: this run predates the standardised log location; new jobs write to `slurm-logs/`.

> **Note**: α = 0.71 means V-JEPA2 contributes ~71% weight (RGB appearance dominates over skeleton motion).

---

## Temporal Action Localization (TAL)

Beyond clip classification, the repo localizes RMMs in untrimmed video. There are three TAL tracks; all are evaluated with **mAP@tIoU {0.3, 0.5, 0.7}** under 3-fold CV.

| Track | Directory | What it does |
|-------|-----------|--------------|
| **Window-based** | `tal/` | Score 2s/1s windows with a clip classifier (V-JEPA2 / PoseC3D / STGCN++ / log-prob MLP fusion), postprocess into segments, evaluate mAP. |
| **OpenTAD E2E** | `OpenTAD/` (submodule) | ActionFormer / TriDet trained end-to-end on V-JEPA2 features. Also trains **binary** detectors (single "action" class) used as Stage 1 below. |
| **Two-stage** | `two-stg/` | Binary detector (Stage 1) → V-JEPA2 / 3-way-fusion-MLP / 5-class classifier (Stage 2). |

**Headline finding:** TriDet E2E and ActionFormer E2E reach the highest avg_mAP (~19.5–19.7%), but classify near chance — their advantage is better-calibrated detection *ranking*. The two-stage pipeline classifies far better per-class (V-JEPA2 lifts per-class accuracy from ~29% to 51–84%) and closes most of the mAP gap; under class-agnostic NMS the E2E edge largely disappears. See `insights/tal/two_stage_analysis.md` and `tal/TAL_MODEL_COMPARISON.md`.

**Where to look:**
- Pipeline + commands: [`REPRODUCE.md`](REPRODUCE.md) §3–5
- Window eval: [`tal/README.md`](tal/README.md)
- Two-stage: [`two-stg/README.md`](two-stg/README.md)
- OpenTAD SAILS fork: [`opentad_sails/README.md`](opentad_sails/README.md)

---

## Data Preparation

### Directory Structure

```
dataprep/
├── splits/
│   ├── cv_splits/                    # 5-class 3-fold CV
│   │   ├── fold_0_train.csv
│   │   ├── fold_0_val.csv
│   │   ├── fold_1_train.csv
│   │   ├── fold_1_val.csv
│   │   ├── fold_2_train.csv
│   │   ├── fold_2_val.csv
│   │   └── split_metadata.json
│   ├── cv_splits_4class/             # 4-class 3-fold CV
│   │   └── ... (same structure)
│   ├── single_split/                 # 5-class train/val/test
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── split_info.json
│   └── single_split_4class/          # 4-class train/val/test
│       └── ... (same structure)
├── clip_gen/                         # Scripts to generate video clips
├── pose_gen/                         # Pose estimation pipeline
├── sam3/                             # SAM3 parsing utilities
├── rmm_sam3_parsed.csv               # SAM3 label parsing results
├── rmm_segments.csv                  # Segment definitions
└── video_meta.json                   # Video metadata
```

### Video Clips

Clips are stored in:
```
/orcd/scratch/bcs/001/sensein/sails/rmm/classification_clips/canonical_clips/
```

Each clip is named `{segment_id}.mp4` (654 total clips).

---

## Output Locations

| Method | Output Directory |
|--------|------------------|
| PoseC3D | `pyskl/work_dirs/posec3d/` |
| STGCN++ | `pyskl/work_dirs/stgcnpp/` |
| V-JEPA2 | `v-jepa/runs/` |
| Late Fusion | `fusion/runs/` |
| Qwen VLM | `insights/vlm/qwen_outputs/` (metrics + predictions, vendored) |

### Generated Files (per experiment)

| File | Description |
|------|-------------|
| `metrics.json` / `cv_summary.json` | Quantitative metrics |
| `predictions_clip.csv` | Per-clip predictions with scores |
| `predictions_video.csv` | Per-video aggregated predictions |
| `confusion_matrix_*.png` | Confusion matrix visualizations |
| `alpha.json` | (Fusion only) Learned fusion weight α |
| `loss_curve.png` | (Fusion only) Training loss over epochs |

---

## Metrics

All methods report both **clip-level** and **video-level** metrics:

| Metric | Description |
|--------|-------------|
| Top-1 Accuracy | Standard accuracy |
| Top-2 Accuracy | Correct if true label is in top-2 predictions |
| Macro F1 | Unweighted mean F1 across classes |
| Weighted F1 | Class-frequency weighted F1 |
| Macro Precision/Recall | Unweighted mean precision and recall |
| Cohen's Kappa | Agreement metric accounting for chance |

**Video-level aggregation**: Clip scores from the same video are averaged, then argmax is taken for final prediction.

---

## Method Comparison Summary

### 4-Class Task (Clip-Level Metrics)

| Rank | Method | Clip Top-1 | Clip Macro F1 | Cohen's κ |
|------|--------|------------|---------------|-----------|
| 🥇 | **Late Fusion (V-JEPA2 + PoseC3D)** | **79.8%** | **78.8%** | 0.669 |
| 🥈 | STGCN++ (4-stream) | 77.9% | 72.7% | **0.635** |
| 🥉 | V-JEPA2 + SAM3 crop | 76.0% | 75.1% | 0.657 |
| 4 | PoseC3D (non-weighted) | 74.6% | 70.4% | 0.580 |
| 5 | Qwen2.5-VL (zero-shot) | — | 28.2% | 0.168 |

> **Note**: Late Fusion achieves best clip-level accuracy and F1 by combining RGB and skeleton modalities. See [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md) for detailed results.

### 5-Class Task (Clip-Level Metrics)

| Rank | Method | Clip Top-1 | Clip Macro F1 | Cohen's κ |
|------|--------|------------|---------------|-----------|
| 🥇 | **PoseC3D** | **66.8%** | **65.1%** | **0.547** |
| 🥈 | STGCN++ (4-stream) | 65.9% | 63.8% | 0.532 |
| 🥉 | V-JEPA2 + SAM3 crop | 62.5% | 63.0% | 0.499 |
| 4 | Qwen2.5-VL (zero-shot) | — | 12.2% | 0.008 |

### Method Overview

| Method | Type | Training | Input | Speed |
|--------|------|----------|-------|-------|
| **PoseC3D** | Skeleton-CNN | Finetuning | 17 COCO keypoints → heatmaps | Fast |
| **STGCN++** | Skeleton-GCN | Finetuning | 17 COCO keypoints → graph | **Very Fast** |
| **V-JEPA2** | Video encoder | Finetuning | RGB frames (+ optional crop) | Slow |
| **Late Fusion** | Ensemble | α only (1 param) | V-JEPA2 + PoseC3D logits | **Instant** |
| **Qwen2.5-VL** | Zero-shot VLM | None | RGB frames | Very Slow |

---

## Directory Overview

```
actreg/
├── README.md                 # This file
├── REPRODUCE.md              # End-to-end pipeline (start here to run things)
├── config.yaml  paths.py     # Central path config (single source of truth)
├── envs/                     # Pinned conda environments (dataprep, pyskl, vjepa2, opentad)
├── docs/                     # INDEX.md, ARTIFACTS.md, ENTRYPOINTS.md
├── opentad_sails/            # OpenTAD fork delta (UPSTREAM_COMMIT + patch)
├── pyskl/                    # Skeleton-based classification (PoseC3D + STGCN++)
│   ├── configs/{posec3d,stgcn++}/...  # SAILS configs
│   ├── tools/                # Training, testing, evaluation scripts
│   └── work_dirs/            # Experiment outputs (git-ignored)
├── v-jepa/                   # V-JEPA2 video encoder finetuning
│   ├── finetune_sails_vjepa2_*.py     # Training scripts (incl. _tal)
│   └── runs/                 # Experiment outputs (git-ignored)
├── fusion/                   # Late fusion (V-JEPA2 + PoseC3D) + 3-way export
├── dataprep/                 # Data preparation utilities
│   ├── splits/               # Classification train/val/test split CSVs
│   ├── tal/                  # TAL window splits (2s/1s) + generators
│   ├── clip_gen/  pose_gen/  sam3/    # Clip gen, pose est., SAM3 masks
├── OpenTAD/                  # (submodule) ActionFormer/TriDet E2E + binary detectors
├── tal/                      # Window-based TAL evaluation (window→segment→mAP)
├── two-stg/                  # Two-stage TAL (binary detector → Stage-2 classifier)
├── insights/                 # Analysis notebooks, failure review, paper figs/tables
├── scripts/                  # Shared utility scripts
└── rendered_videos/          # Demo/visualization outputs (git-ignored *.mp4)
```

---

## Training Analysis & Recommendations

### Observed Issues

From training logs, we observed significant **overfitting** in both skeleton-based methods:

| Model | Training Acc | Validation Acc | Gap |
|-------|--------------|----------------|-----|
| PoseC3D | 100% | ~76% | **24%** |
| STGCN++ | 100% | ~65-75% | **25-35%** |

Both models reach 100% training accuracy by epoch 3-5 but validation accuracy plateaus much lower.

### Recommendations for Improved Training

| Recommendation | PoseC3D | STGCN++ |
|----------------|---------|---------|
| **Use class weighting** | ✓ `train_weighted.py` | ✓ `train_weighted.py` |
| **Increase dropout** | 0.5 → 0.7 | 0 → **0.5** |
| **Reduce epochs** | 12 → 10 | 24 → 16 |
| **Lower learning rate** | 0.00125 (OK) | 0.01 → 0.005 |
| **Increase weight decay** | 0.0005 → 0.001 | 0.0005 → 0.001 |

### Configuration Overrides

```bash
# PoseC3D with recommended regularization
python tools/train_weighted.py configs/posec3d/slowonly_r50_sails_k400p/joint.py \
    --ann-file data/sails/single/4class_conf04.pkl \
    --work-dir work_dirs/posec3d/single/4class_conf04_weighted \
    --total-epochs 10 \
    --lr 0.00125 \
    --validate --launcher none

# STGCN++ with recommended regularization
python tools/train_weighted.py configs/stgcn++/stgcnpp_sails_ntu60p/j.py \
    --ann-file data/sails/single/4class_conf04.pkl \
    --work-dir work_dirs/stgcnpp/single/4class_conf04_weighted_j \
    --total-epochs 16 \
    --lr 0.005 \
    --validate --launcher none
```

### How Keypoint Confidence is Handled

| Model | Confidence Handling | Robustness to Noise |
|-------|---------------------|---------------------|
| **PoseC3D** | Scales heatmap intensity (built-in) | Higher |
| **STGCN++** | Input feature channel (learned) | Lower |

PoseC3D has a more principled approach: low-confidence keypoints produce weak heatmap signals, naturally reducing their influence. STGCN++ must learn that `(0, 0, 0)` means "missing keypoint."