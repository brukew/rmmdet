# Two-Stage TAL Pipeline

Binary ActionFormer detection followed by V-JEPA2 classification on each detected segment.

## Pipeline

1. **Stage 1:** Load precomputed ActionFormer Binary detections (`result_detection.json` per fold).
2. **Stage 2:** For each proposal, extract the video clip and run V-JEPA2 classifier; assign RMM label and combine score as `det_score × max(rmm_class_probs)`.
3. **Eval:** Save OpenTAD-style `result_detection.json` and evaluate with the native OpenTAD mAP evaluator on `fold{N}_anno.json`.

## Classifier Variants

| Variant | Classes | Checkpoint | Description |
|---------|---------|------------|-------------|
| **4-class** | hands_flapping, jumping, rocking, spinning | `vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls` | Standard RMM classifier |
| **3-way fusion** | 4 RMM | V-JEPA2 + `two-stg/fusion_checkpoints/three_way/fold_N` | Live V-JEPA2 RGB + MLP fusion. **Default:** PoseC3D/STGCN++ via `skeleton_index.csv` (nearest val clip by tIoU). **Optional:** `--posec3d-proposal-scores` + `--stgcn-proposal-scores` from `build_proposal_poses.py` + `run_proposal_skeleton_inference.sh` (true per-proposal skeleton scores). Fusion MLP is trained with **cross-fold val predictions** (decontaminated); see `RUNS.md`. |
| **5-class** | 4 RMM + background | `vjepa2_tal_cv_5class_balanced` | Includes background; scores use RMM probs only |

For 5-class, background predictions naturally downweight proposals where the classifier thinks it's not an RMM event (the max RMM probability will be lower), but proposals are **not filtered out** — this provides a soft rejection signal.

## Usage

### 4-class (standard)

**All folds (GPU):**

```bash
cd /orcd/data/satra/001/users/brukew/actreg
bash two-stg/run_two_stage_cv_gpu.sh
```

**Single fold:**

```bash
python two-stg/eval_two_stage_tal.py \
  --fold 0 \
  --detection-json OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold0/gpu1_id99/result_detection.json \
  --checkpoint-dir v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls/fold_0 \
  --output-dir two-stg/eval_results/fold0
```

### 3-way fusion (V-JEPA2 + PoseC3D + STGCN++ MLP)

Export per-fold fusion weights and skeleton index once (CPU/GPU):

```bash
cd /orcd/data/satra/001/users/brukew/actreg
conda activate vjepa2
python fusion/export_three_way_deploy_checkpoint.py --all-folds
```

This overwrites `two-stg/fusion_checkpoints/three_way/fold_{0,1,2}/` with a **decontaminated** MLP (trained on the *other* folds’ val clip predictions only) and per-fold `skeleton_index.csv`.

**Single fold on SLURM (GPU, recommended):**

```bash
sbatch --export=FOLD=0 two-stg/run_three_way_fold_gpu.sh
```

**All folds (3 separate GPU jobs):**

```bash
bash two-stg/run_three_way_cv_gpu.sh
```

Outputs: `two-stg/eval_results_3way/fold{N}/` (includes `info.json` from `write_info_json.py --stage2-backend three_way`).

**Interactive / login node (same flags as SLURM):**

```bash
python two-stg/eval_two_stage_tal.py \
  --fold 0 \
  --detection-json OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold0/gpu1_id99/result_detection.json \
  --checkpoint-dir v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls/fold_0 \
  --output-dir two-stg/eval_results_3way/fold0 \
  --classifier-backend three_way \
  --device cuda
```

You can also set `CLASSIFIER_BACKEND=three_way` with `run_single_fold_gpu.sh` (same env vars as documented in that script).

### 3-way fusion with **live** PoseC3D / STGCN++ on each proposal

1. Build pickles (CPU; uses H5 pose caches):

   ```bash
   conda activate vjepa2
   python two-stg/build_proposal_poses.py --all-folds
   ```

   Each fold also writes `two-stg/proposal_pickles/proposals_fold{N}_skips.json`: every
   skipped proposal (clip id, `video_file`, `reason`, `detail`, plus counts). Use
   `--no-skip-json` to disable.

   **Note:** If many proposals skip with “starts after end of pose cache”, the HDF5
   only covers a prefix of the video (e.g. frames `0..5399` ≈ first ~3 minutes at 30fps).
   Late ActionFormer proposals need a full-length pose cache from your pose pipeline.

2. Run skeleton models on GPU (see `two-stg/run_proposal_skeleton_inference.sh`; needs `conda activate pyskl`).

3. Evaluate with both CSVs (paths must match fold):

   ```bash
   python two-stg/eval_two_stage_tal.py \
     --fold 0 \
     --detection-json OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold0/gpu1_id99/result_detection.json \
     --checkpoint-dir v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls/fold_0 \
     --output-dir two-stg/eval_results_3way_live/fold0 \
     --classifier-backend three_way \
     --posec3d-proposal-scores two-stg/proposal_scores/posec3d/fold0/predictions_clip.csv \
     --stgcn-proposal-scores two-stg/proposal_scores/stgcnpp_fused_4stream/fold0/predictions_clip.csv \
     --device cuda
   ```

**One-shot SLURM (long job):** `sbatch --export=FOLD=0 two-stg/run_three_way_live_fold_gpu.sh`  
**All folds:** `bash two-stg/run_three_way_live_cv_gpu.sh`

### 5-class (4 RMM + background)

**All folds (GPU):**

```bash
cd /orcd/data/satra/001/users/brukew/actreg
bash two-stg/run_5class_cv_gpu.sh
```

**Single fold:**

```bash
python two-stg/eval_two_stage_tal.py \
  --fold 0 \
  --detection-json OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold0/gpu1_id99/result_detection.json \
  --checkpoint-dir v-jepa/runs/vjepa2_tal_cv_5class_balanced/fold_0 \
  --output-dir two-stg/eval_results_5class/fold0 \
  --num-classes 5
```

Results for 5-class are saved to `eval_results_5class/` and include a `bg_prob` column in `predictions.csv` for analysis.

The wrapper scripts now use two conda envs:
- `vjepa2` for proposal classification
- `opentad` for native OpenTAD metric evaluation

If classification jobs are already running or already finished, native evaluation can also be run later with:

```bash
sbatch --export=FOLDS=1,2 two-stg/run_opentad_postprocess.sh
```

## Outputs

- `two-stg/eval_results/fold{N}/predictions.csv` — all predictions (video_key, class_id, start_sec, end_sec, score).
- `two-stg/eval_results/fold{N}/result_detection.json` — OpenTAD-native prediction JSON.
- `two-stg/eval_results/fold{N}/metrics.json` — OpenTAD mAP metrics per fold.
- `two-stg/eval_results/fold{N}/metrics_opentad.json` — duplicate native OpenTAD metrics.
- `two-stg/eval_results/fold{N}/metrics_custom.json` — legacy custom-evaluator metrics for cross-checking.
- `two-stg/eval_results/fold{N}/info.json` — metadata about models, envs, timestamps, and artifacts.
- `two-stg/eval_results/cv_summary.json` — mean ± std across folds (after running the full CV or `aggregate_cv_results.py`).

## Scripts

- `eval_two_stage_tal.py` — main evaluation (load detections, run classifier, compute mAP).
- `opentad_eval.py` — convert `predictions.csv` to OpenTAD format and evaluate with OpenTAD.
- `run_opentad_postprocess.sh` — run native OpenTAD evaluation after folds finish.
- `write_info_json.py` — write per-fold `info.json` metadata files.
- `run_two_stage_cv.sh` — run all 3 folds (4-class) and aggregate.
- `run_two_stage_cv_gpu.sh` — SLURM job script for 4-class.
- `run_5class_cv_gpu.sh` — submit all 3 folds for 5-class evaluation.
- `run_5class_fold_gpu.sh` — SLURM job script for single 5-class fold.
- `run_three_way_fold_gpu.sh` — SLURM: single fold, 3-way fusion Stage-2, GPU, OpenTAD eval, `info.json`.
- `run_three_way_cv_gpu.sh` — submit folds 0–2 for 3-way pipeline.
- `build_proposal_poses.py` — ActionFormer proposals → pyskl pickle (H5 pose caches).
- `fuse_stgcn_4stream.py` — fuse four STGCN++ modality `predictions_clip.csv` files (weights 2,2,1,1).
- `run_proposal_skeleton_inference.sh` — SLURM: build pickle + PoseC3D + STGCN++ inference → proposal score CSVs.
- `run_three_way_live_fold_gpu.sh` / `run_three_way_live_cv_gpu.sh` — full 3-way eval with live proposal skeleton scores.
- `aggregate_cv_results.py` — build cv_summary.json from fold metrics.
