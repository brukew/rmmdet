# OpenTAD Integration Note

This directory now includes a native OpenTAD evaluation path for the two-stage TAL pipeline, plus a separate postprocess job that can be run after classification finishes.

## What Was Added

- `opentad_eval.py`
  - Converts `predictions.csv` into OpenTAD-style `result_detection.json`
  - Runs the native OpenTAD `mAP` evaluator against `OpenTAD/data/sails_rmm/annotations/fold{N}_anno.json`
  - Writes `metrics.json` using OpenTAD's metrics and thresholds
  - Also writes `metrics_opentad.json` as a duplicate native-metrics artifact for clarity

- `eval_two_stage_tal.py`
  - Saves `predictions.csv`
  - Saves `metrics_custom.json` from the legacy/custom evaluator
  - Supports `--skip-opentad-eval` so wrapper scripts can leave native evaluation to the `opentad` conda env
  - **5-class support (March 2026):**
    - Added `--num-classes` argument (4 or 5); auto-detects from model config if not specified
    - For 5-class (4 RMM + background): scores are computed using only RMM probabilities (indices 0-3), ignoring background (index 4)
    - This naturally downweights proposals where the classifier predicts background, without hard filtering
    - Saves `bg_prob` column in `predictions.csv` for post-hoc analysis
  - **3-way + live skeleton scores (March 2026):** optional `--posec3d-proposal-scores` and `--stgcn-proposal-scores` together with `--classifier-backend three_way`

- `run_two_stage_cv.sh`
  - Uses `vjepa2` for proposal classification
  - Then switches to `opentad` for native metric evaluation

- `run_single_fold_gpu.sh`
  - Mirrors the same two-env split for single-fold GPU jobs
  - Set `CLASSIFIER_BACKEND=three_way` for 3-way fusion Stage-2 (optional `FUSION_BUNDLE_DIR`, `OUTPUT_ROOT`)

- `run_three_way_fold_gpu.sh` **(March 2026)**
  - Single-fold SLURM job: `--classifier-backend three_way`, `--device cuda`, OpenTAD post-eval, `write_info_json.py --stage2-backend three_way`
  - `sbatch --export=FOLD=0 two-stg/run_three_way_fold_gpu.sh`

- `run_three_way_cv_gpu.sh`
  - Submits folds 0–2 as three separate GPU jobs

- `build_proposal_poses.py`, `fuse_stgcn_4stream.py`, `run_proposal_skeleton_inference.sh` **(March 2026)**
  - ActionFormer proposals → pyskl pickle from H5 pose caches; PoseC3D + STGCN++ (4-stream) inference; fused STGCN CSV for eval

- `run_three_way_live_fold_gpu.sh`, `run_three_way_live_cv_gpu.sh` **(March 2026)**
  - End-to-end 3-way eval with **live** per-proposal skeleton scores (`--posec3d-proposal-scores` / `--stgcn-proposal-scores` passed to `eval_two_stage_tal.py`)
  - Fusion MLP from `export_three_way_deploy_checkpoint.py` uses **cross-fold** training (decontaminated); see `RUNS.md`

- `write_info_json.py`
  - Optional `--posec3d-proposal-scores` / `--stgcn-proposal-scores` for `info.json` when using live proposal skeleton CSVs

- `run_5class_fold_gpu.sh` **(March 2026)**
  - Single-fold SLURM job using the 5-class V-JEPA2 classifier (`vjepa2_tal_cv_5class_balanced`)
  - Outputs to `eval_results_5class/fold{N}/`

- `run_5class_cv_gpu.sh` **(March 2026)**
  - Submits all 3 folds for 5-class evaluation

- `run_opentad_postprocess.sh`
  - Separate postprocess SLURM job
  - Intended for the case where classification jobs are already running or already finished
  - Reads finished `predictions.csv` files and writes OpenTAD-native outputs afterward
  - Can be submitted with SLURM dependencies so it starts automatically after upstream fold jobs complete

- `write_info_json.py`
  - Writes `info.json` into each per-fold results directory
  - Records detector/classifier/evaluator metadata, paths, envs, timestamps, artifact presence, and metric summaries

## Environment Split

- `vjepa2` env
  - Used for clip extraction and V-JEPA2 classification
  - Needed for `transformers`, `decord`, and model inference

- `opentad` env
  - Used for native OpenTAD metric evaluation
  - Ensures two-stage scores are directly comparable to ActionFormer/OpenTAD baselines

## Current Workflow

1. Run or finish the two-stage classifier step, producing `predictions.csv`
2. Run native OpenTAD evaluation either:
   - directly from the wrapper scripts, or
   - later via `run_opentad_postprocess.sh`
3. Write `info.json` so each fold directory records what was run and how

## Separate Postprocess Flow

If folds are already running and only the OpenTAD conversion/evaluation needs to happen later, use:

```bash
sbatch --export=FOLDS=1,2,SOURCE_JOB_IDS=<job1>,<job2> two-stg/run_opentad_postprocess.sh
```

Or with explicit dependencies:

```bash
sbatch --dependency=afterany:<job1>:<job2> \
  --export=FOLDS=1,2,SOURCE_JOB_IDS=<job1>,<job2> \
  two-stg/run_opentad_postprocess.sh
```

This job:
- waits for the upstream fold jobs if a dependency is supplied
- runs `two-stg/opentad_eval.py`
- then runs `two-stg/write_info_json.py`

## Expected Per-Fold Outputs

- `predictions.csv` — raw two-stage predictions (includes `bg_prob` column for 5-class)
- `result_detection.json` — OpenTAD-native prediction file
- `metrics.json` — OpenTAD-native evaluation metrics
- `metrics_opentad.json` — duplicate native OpenTAD metrics
- `metrics_custom.json` — legacy/custom evaluator metrics for cross-checking
- `info.json` — metadata about the models, envs, artifact timestamps, and evaluation setup

**Note:** 5-class results are saved to `eval_results_5class/` instead of `eval_results/`.

## Smoke Tests Performed

Validated fold 0 integration by running:

1. `two-stg/opentad_eval.py` on existing `two-stg/eval_results/fold0/predictions.csv`
2. The same evaluation inside the `opentad` conda environment
3. `two-stg/run_opentad_postprocess.sh` manually on fold 0
4. `two-stg/write_info_json.py` on existing fold result directories

These produced valid OpenTAD outputs and metadata, confirming the conversion path works on real classifier outputs and that the postprocess job can populate final artifacts afterward.
