# Two-Stage TAL Pipeline — End-to-End

## Stage 0: Data Splits
- **Input:** Annotated RMM segments, video-level fold assignments (`video_to_split_4class_cv.csv`)
- **Output:** Per-fold `fold_{N}_train_windows.csv`, `fold_{N}_val_windows.csv`, `fold{N}_anno.json` (OpenTAD format)
- **Protocol:** 3-fold CV, split at video level. Outer val is reserved for final reporting only.

## Stage 1a: Binary V-JEPA TAL feature model (per fold)
- **Input:** Fold train windows, pre-trained V-JEPA2 base model
- **Output:** Fine-tuned binary (RMM vs BG) V-JEPA2 checkpoint per fold
- **Checkpoint rule:** Fixed epoch or inner-dev early stopping. **Must not use outer val.**

## Stage 1b: Feature extraction (per fold)
- **Input:** Stage 1a checkpoint, all videos for that fold (train + val)
- **Output:** Per-video `.npy` feature files in a fold-specific directory

## Stage 1c: ActionFormer binary training (per fold)
- **Input:** Stage 1b features, `fold{N}_anno_binary.json`
- **Output:** Trained ActionFormer binary checkpoint(s)
- **Checkpoint rule:** Fixed epoch or train-loss-only selection. **Must not use outer val for selection.**

## Stage 1d: ActionFormer binary inference (per fold)
- **Input:** Stage 1c checkpoint, Stage 1b val-fold features
- **Output:** `result_detection.json` — binary RMM proposals with scores and segment boundaries

## Stage 2: V-JEPA2 4-class clip classification (per fold)
- **Input:** `result_detection.json` proposals, V-JEPA2 4-class classifier checkpoint (existing, final-epoch saved), raw video files
- **Output:** `predictions.csv` — 4-class labeled proposals with `score = det_score × max(class_probs)`

## Stage 3: Evaluation (per fold)
- **Input:** `predictions.csv`, `fold{N}_anno.json` (4-class GT)
- **Output:** `metrics.json` (OpenTAD native mAP @ tIoU {0.3, 0.4, 0.5, 0.6, 0.7}), `result_detection.json`

## Stage 4: Aggregation
- **Input:** Per-fold `metrics.json`
- **Output:** `cv_summary.json` (mean ± std across folds)

---

## What is reusable from existing work
- Stage 0: existing splits — **reusable**
- Stage 2 classifier: existing 4-class checkpoints — **reusable**
- Stage 3 + 4 tooling: existing `opentad_eval.py`, `aggregate_cv_results.py` — **reusable**

## What must be rerun
- Stage 1a: binary V-JEPA TAL feature model — **retrain under clean checkpoint rule**
- Stage 1b: feature extraction — **regenerate from clean Stage 1a**
- Stage 1c: ActionFormer binary — **retrain on clean features, clean checkpoint rule**
- Stage 1d: ActionFormer inference — **rerun on clean Stage 1c**
