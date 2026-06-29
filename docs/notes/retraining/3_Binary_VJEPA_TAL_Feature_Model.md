# Deep Dive: Binary V-JEPA TAL Feature Model (Stage 1a)

## What It Is

A V-JEPA2 ViT-L model fine-tuned for binary classification (RMM vs Background) on 2-second TAL window clips. Its primary purpose is **not** to be a standalone classifier — it is used to **extract features** (1024-dim vectors) that ActionFormer consumes for temporal action localization.

---

## Training Script

**Script:** `v-jepa/finetune_sails_vjepa2_tal.py`

### Architecture
- Base model: `facebook/vjepa2-vitl-fpc16-256-ssv2` (ViT-L, 24 layers, 16 heads, hidden 1024, patch 16, tubelet 2)
- Classification head: 2 classes (RMM, Background)
- **Backbone frozen** — only the classification head is trained
- Input: 32 frames per clip (2s @ 16fps)

### Training Protocol
- Optimizer: Adam, LR = 1e-5
- Batch size: 1 (effective 4 via gradient accumulation)
- Max epochs: 20
- **Early stopping: patience=5, monitored on val_loss**
- Background downsampling: class_prob = `[1.0, 0.1]` (keep all RMM, 10% of BG)
- SAM3 child cropping enabled
- Class weights computed from balanced train distribution

### Checkpoint Saving Logic (CRITICAL)

```
if val_loss < best_val_loss:
    best_val_loss = val_loss
    model.save_pretrained(fold_out)       # <-- overwrites in place
    logger.info("New best model saved")
else:
    epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        break  # early stop, best checkpoint already saved
```

**The saved checkpoint is the best-on-outer-val-loss model.** There is no separate "final" vs "best" filename — the best overwrites in place. If early stopping triggers, the saved model is the one with lowest outer val loss.

**This is the validation bias we identified.** The checkpoint is selected by optimizing on the same outer val fold that we later evaluate on.

---

## Historical Run: Job 7707416 (Jan 5–6, 2026)

### SLURM Configuration
- **Script:** `v-jepa/slurm/tal/train_tal_cv_binary_balanced.sh`
- **Partition:** `pi_satra`
- **GPU:** 1
- **CPUs:** 8
- **Memory:** 128G
- **Time:** 48h
- **Conda env:** `vjepa2`

### Per-Fold Training History

**Fold 0:**
| Epoch | val_acc | val_loss | Action |
|-------|---------|----------|--------|
| 1 | 0.802 | 0.4754 | **New best saved** |
| 2 | 0.779 | 0.5025 | No improvement (1/5) |
| 3 | 0.755 | 0.5187 | No improvement (2/5) |
| 4 | 0.809 | **0.4394** | **New best saved** |
| 5 | 0.805 | 0.4718 | No improvement (1/5) |
| 6 | 0.821 | 0.4948 | No improvement (2/5) |
| 7 | 0.798 | 0.6474 | No improvement (3/5) |
| 8 | 0.805 | 0.7316 | No improvement (4/5) |
| 9 | 0.815 | 0.8464 | **Early stop** (5/5) |

Saved checkpoint: **epoch 4** (val_loss=0.4394)

**Fold 1:**
| Epoch | val_acc | val_loss | Action |
|-------|---------|----------|--------|
| 1 | 0.770 | 0.5014 | New best saved |
| 2 | 0.773 | 0.4845 | New best saved |
| 3 | 0.772 | **0.4741** | **New best saved** |
| 4 | 0.738 | 0.5571 | No improvement (1/5) |
| 5 | 0.739 | 0.5766 | No improvement (2/5) |
| 6 | 0.737 | 0.6482 | No improvement (3/5) |
| 7 | 0.764 | 0.6153 | No improvement (4/5) |
| 8 | 0.740 | 0.7446 | **Early stop** (5/5) |

Saved checkpoint: **epoch 3** (val_loss=0.4741)

**Fold 2:**
| Epoch | val_acc | val_loss | Action |
|-------|---------|----------|--------|
| 1 | 0.745 | 0.5696 | New best saved |
| 2 | 0.838 | **0.4094** | **New best saved** |
| 3 | 0.794 | 0.4711 | No improvement (1/5) |
| 4 | 0.726 | 0.5831 | No improvement (2/5) |
| 5 | 0.802 | 0.4535 | No improvement (3/5) |
| 6 | 0.834 | 0.4126 | No improvement (4/5) |
| 7 | 0.788 | 0.5238 | **Early stop** (5/5) |

Saved checkpoint: **epoch 2** (val_loss=0.4094)

### Summary Table

| Fold | Best epoch | Best val_loss | Early stop epoch | Total train time |
|------|-----------|---------------|-----------------|-----------------|
| 0 | 4 | 0.4394 | 9 | ~5h |
| 1 | 3 | 0.4741 | 8 | ~4.5h |
| 2 | 2 | 0.4094 | 7 | ~4h |

---

## Binary Model Window-Level Performance (Existing, on Outer Val)

| Metric | Fold 0 | Fold 1 | Fold 2 | Mean ± Std |
|--------|--------|--------|--------|------------|
| Clip Top-1 Acc | 81.5% | 74.0% | 78.8% | 78.1 ± 3.1% |
| Macro F1 | 63.8% | 61.7% | 61.0% | 62.2 ± 1.2% |
| RMM vs BG F1 | 38.4% | 40.0% | 34.7% | 37.7 ± 2.2% |
| RMM vs BG AUC | 79.4% | 79.6% | 80.1% | 79.7 ± 0.3% |
| RMM Recall | 62.6% | 74.8% | 66.3% | 67.9 ± 5.1% |
| RMM Precision | 27.7% | 27.3% | 23.5% | 26.1 ± 1.9% |

**Note:** These metrics are biased because the checkpoint was selected on the same val fold.

---

## Feature Extraction

### Script
**Path:** `v-jepa/tools/extract_vjepa_features.py`

### Process
1. Load fine-tuned V-JEPA2 checkpoint from `runs/vjepa2_tal_cv_binary_balanced/fold_{N}/`
2. For each video, extract 16-frame snippets with stride 16 (non-overlapping)
3. Pass through V-JEPA2 backbone → get `last_hidden_state`
4. Mean-pool over spatial patches → 1024-dim feature per snippet
5. Save as `{child_id}_{filename_stem}.npy` with shape `(T, 1024)`

### Output
**Location:** `/orcd/scratch/bcs/001/sensein/sails/rmm/features/vjepa2_tal_cv_binary_balanced_fold_{0,1,2}/`

| Fold | Feature files | Shape example |
|------|--------------|---------------|
| 0 | 298 | (139, 1024) |
| 1 | 298 | (T, 1024) |
| 2 | 298 | (T, 1024) |

All 298 videos have features extracted for every fold (train + val videos both).

### SLURM
- **Script:** `v-jepa/slurm/extract_features.sh`
- **Distributed wrapper:** `v-jepa/tools/extract_distributed.sh` (4 workers per fold)
- **Partition:** `mit_preemptable`
- **Time:** 4h per worker
- **Historical jobs:** 7754xxx series (Jan 6, 2026)

---

## Existing Checkpoints

**Path:** `v-jepa/runs/vjepa2_tal_cv_binary_balanced/`

```
vjepa2_tal_cv_binary_balanced/
├── cv_summary.csv
├── cv_summary.json
├── fold_0/    (1.4G model.safetensors, config.json, metrics.json, predictions_clip.csv)
├── fold_1/    (1.4G model.safetensors, ...)
└── fold_2/    (1.4G model.safetensors, ...)
```

Each fold is a HuggingFace-format directory (config.json + model.safetensors + video_preprocessor_config.json).

**Training date:** Jan 5–6, 2026 (Job 7707416)

---

## Connection to ActionFormer

These features feed into ActionFormer binary training:

```
Features: /orcd/scratch/.../features/vjepa2_tal_cv_binary_balanced_fold_{N}/
    ↓
OpenTAD config: configs/actionformer/sails_rmm_vjepa_binary_fold{N}.py
    ↓
ActionFormer binary training (num_classes=1, in_channels=1024)
    ↓
Binary detection proposals: result_detection.json
```

---

## Paper-Readiness Assessment

**NOT paper-ready.** The checkpoint is selected by best-on-outer-val-loss (early stopping on the same fold used for final evaluation). This introduces optimistic validation bias.

**To make paper-ready:** Must retrain with either:
1. **Fixed epoch policy** — train for N epochs (e.g., 3 or 4, based on observed best epochs), save final model, no val-based selection
2. **Inner dev split** — split the training data into train_inner (85%) + dev_inner (15%), early-stop on dev_inner loss, evaluate on outer val

**Recommended:** Fixed epoch = 3 (median of observed best epochs: 4, 3, 2). This avoids all val-based checkpoint selection and requires zero changes to the evaluation protocol. It just needs one retrain.
