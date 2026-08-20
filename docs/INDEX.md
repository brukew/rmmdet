# Documentation Index

Catalog of all documentation and key entry points in the repo. Component READMEs
live in their own directories; cross-cutting/handoff docs live under `docs/`.

## Start here (handoff / reproducibility)

- `@actreg/README.md`: Main project README (two tasks: classification + TAL; methods, commands, directory overview, results).
- `@actreg/REPRODUCE.md`: End-to-end pipeline map (setup → data prep → classifiers → window TAL → OpenTAD E2E/binary → two-stage → analysis), including expected results and checkpoint locations.
- `@actreg/docs/ENTRYPOINTS.md`: **What to run** — every launch/eval script, env, args, inputs, and output path.
- `@actreg/docs/ARTIFACTS.md`: Data/artifact retention policy and where inputs/checkpoints live.
- `@actreg/config.yaml` + `@actreg/paths.py`: Central path configuration (single source of truth; env-overridable).
- `@actreg/envs/`: Pinned conda environments (`dataprep`, `pyskl`, `vjepa2`, `opentad`).
- `@actreg/opentad_sails/`: OpenTAD fork delta — pinned upstream commit + verified-applicable SAILS patch (OpenTAD is a git submodule).

## Classification / Action Recognition (clip/video-level)

- `@actreg/MODEL_COMPARISON.md`: Central comparison doc for **classification** across methods (PoseC3D, STGCN++, V-JEPA2, Qwen) + fusion variants; includes tables, per-class P/R, and output locations.

---

## Temporal Action Localization (TAL)

### TAL results + comparisons (docs)

- `@actreg/tal/TAL_MODEL_COMPARISON.md`: Main TAL comparison (window-based postprocessing vs fusion vs ActionFormer/OpenTAD; includes balanced/binary variants and recall tables).
- `@actreg/tal/TAL_DET_MODEL_COMPARISON.md`: Window-level **5-class detection** (incl. background) comparison; balancing strategies and binary (BG vs RMM) detection metrics.
- `@actreg/insights/tal/TAL_INSIGHTS_SUMMARY.md`: Deeper behavioral analysis (top-2 gap, fusion paradox, FP characteristics) to explain observed TAL behavior.

### TAL evaluation pipeline (segment mAP@tIoU)

- `@actreg/tal/README.md`: Overview + usage for TAL evaluation pipeline (window scores → segments → mAP@tIoU).
- `@actreg/tal/eval_tal_from_window_preds.py`: End-to-end CLI to evaluate a `window_preds.csv` into segment metrics/artifacts.
- `@actreg/tal/window_to_segments.py`: Postprocessing utilities (threshold/smooth/merge) to turn window scores into predicted segments.
- `@actreg/tal/tal_map_eval.py`: Segment-level AP/mAP evaluation at tIoU thresholds.
- `@actreg/tal/run_tal_eval_cv.py`: Convenience runner to evaluate across CV folds.
- `@actreg/tal/tal_fusion.py`: Late fusion for TAL (MLP on log-prob features) used for window-based fusion baselines.
- `@actreg/tal/scripts/grid_search_postprocessing.py`: Parameter sweep for postprocessing hyperparameters.
- `@actreg/tal/scripts/run_tal_fusion_oof.py`: Generate out-of-fold fused window predictions for TAL fusion models.
- `@actreg/tal/scripts/eval_best_postprocess.py`: Evaluate models using best postprocessing params and write CV summaries.
- `@actreg/tal/scripts/eval_binary_tal.py`: Binary TAL evaluation helper (RMM vs background).

### TAL data splits / window generation

- `@actreg/dataprep/tal/README.md`: TAL window split generation + CSV schema + labeling rule (multi-label windows; background windows included).
- `@actreg/dataprep/tal/make_tal_video_assignments.py`: Build video-to-split assignments (CV + single-split).
- `@actreg/dataprep/tal/make_tal_window_splits.py`: Generate 2s/1s-stride window CSVs for TAL.
- `@actreg/dataprep/tal/validate_tal_splits.py`: Sanity checks/validation for generated TAL window splits.
- `@actreg/dataprep/tal/cut_tal_windows.py`: Optional helper to cut physical window clips to disk (debugging/inspection).
- `@actreg/dataprep/tal/create_tal_pose_pickles.sbatch`: SLURM job to generate pyskl-compatible pose pickles for TAL.
- `@actreg/dataprep/tal/create_tal_pose_pickles_4class_only.sbatch`: Variant of pose-pickle generation restricted to 4-class.
- `@actreg/dataprep/tal/splits_cv_4class/`: (Directory) CV window splits (may be populated by generation jobs).
- `@actreg/dataprep/tal/label_maps/`: (Directory) Label maps for TAL tasks.

### Two-stage TAL pipeline (binary detector → Stage-2 classifier)

- `@actreg/two-stg/README.md`: Pipeline overview + usage (4-class V-JEPA2, 3-way fusion MLP, 5-class Stage-2 variants).
- `@actreg/two-stg/eval_two_stage_tal.py`: Main driver (load Stage-1 detections, run Stage-2 classifier, OpenTAD mAP eval).
- `@actreg/two-stg/RUNS.md`: Run log / leaderboard for two-stage experiments.
- `@actreg/two-stg/ANALYSIS_NOTES.md`: Two-stage vs E2E analysis (mAP tables, balanced accuracy, NMS caveat).
- `@actreg/two-stg/analyze_classifier_effect.py`: Classifier-effect analysis (per-class accuracy, AUROC, error taxonomy).
- `@actreg/OpenTAD/`: (submodule, pinned upstream + SAILS patch) ActionFormer/TriDet E2E and binary detectors; see `@actreg/opentad_sails/README.md`.

---

## Models / Training code

### V-JEPA2 (RGB)

- `@actreg/v-jepa/finetune_sails_vjepa2_cv.py`: V-JEPA2 CV finetuning script for classification.
- `@actreg/v-jepa/finetune_sails_vjepa2_cv_crop.py`: V-JEPA2 CV finetuning with SAM3 cropping.
- `@actreg/v-jepa/finetune_sails_vjepa2_single.py`: V-JEPA2 single split finetuning script.
- `@actreg/v-jepa/finetune_sails_vjepa2_tal.py`: V-JEPA2 training/inference code for TAL (window-based / feature extraction for TAL).
- `@actreg/v-jepa/tools/extract_vjepa_features.py`: Feature extraction utility for V-JEPA2 (used for TAL / downstream models).
- `@actreg/v-jepa/tools/summarize_vjepa_cv_results.py`: Summarize CV results for V-JEPA runs.
- `@actreg/v-jepa/slurm/`: SLURM entrypoints for running V-JEPA experiments (incl. TAL balanced/binary training scripts).

### PySKL (PoseC3D / STGCN++)

- `@actreg/pyskl/`: Forked training/evaluation stack for skeleton-based models (configs, tools, and checkpoints).
- `@actreg/pyskl/tools/train_weighted.py`: Training entrypoint with class-weighting support (used heavily due to imbalance).
- `@actreg/pyskl/tools/test.py`: Standard pyskl test/eval entrypoint.
- `@actreg/pyskl/tools/compute_class_weights.py`: Compute inverse-frequency class weights for splits.
- `@actreg/pyskl/tools/evaluate_sails.py`: Project-specific evaluation utilities for SAILS splits.

### OpenTAD / transformer TAL

- `@actreg/OpenTAD/`: Transformer-based TAL codebase used for ActionFormer experiments (end-to-end segment localization).

---

## Fusion (classification)

- `@actreg/fusion/train_fusion_cv.py`: Train/evaluate fusion models over CV (scalar/per-class/MLP/3-way variants depending on args/scripts).
- `@actreg/fusion/slurm/`: SLURM scripts for running fusion CV jobs (2-way and 3-way; 4-class and 5-class).

---

## Data preparation (classification + pose + SAM3)

- `@actreg/dataprep/clip_gen/create_clip_segments.py`: Generate canonical classification clips/segments.
- `@actreg/dataprep/pose_gen/batch_sam_pose.py`: Batch pipeline for generating SAM3 masks + pose caches.
- `@actreg/dataprep/pose_gen/visualize_pose_from_cache.py`: Visualize cached pose outputs for QA/debugging.
- `@actreg/dataprep/sam3/parse_sam3_labels.py`: Parse SAM3 label exports into usable CSVs/structures.
- `@actreg/dataprep/sam3/sam3_crops_utils.py`: Utilities for crops/masks used in RGB pipelines (e.g., V-JEPA cropping).
- `@actreg/dataprep/data_review/DATA_FLOW.md`: Data flow notes for the preprocessing pipeline.

---

## Insights / diagnostics

- `@actreg/insights/failures/common_failures_4class.ipynb`: Notebook for qualitative failure analysis on 4-class classification.
- `@actreg/insights/failures/review_flags.md`: Notes/flags from reviewing failures and edge cases.
- `@actreg/insights/tal/tal_model_analysis.ipynb`: Notebook backing TAL behavior analysis (top-2 gap / fusion behavior).

---

## Utility scripts / notebooks

- `@actreg/scripts/extract_precision_recall.py`: Helper script for extracting precision/recall-style summaries from predictions/metrics.
- `@actreg/scripts/validate_clips.py`: Validate that clips are readable/consistent (I/O QA).
- `@actreg/scripts/render_rmm_numbered.py`: Render/visualize clips with numbering/labels for review.
- `@actreg/scripts/render_tracked_videos.py`: Render tracked videos (debugging tracking/COI extraction).

---

## Outputs (generated artifacts)

- `@actreg/v-jepa/runs/`: Saved V-JEPA models + confusion matrices for classification/TAL runs.
- `@actreg/pyskl/work_dirs/`: Saved checkpoints and eval outputs for PoseC3D/STGCN++ experiments (classification + TAL).
- `@actreg/fusion/runs/`: Saved fusion run artifacts (confusion matrices, loss curves, fold outputs).
- `@actreg/tal/eval_results/`: TAL evaluation outputs (e.g., CV summaries, per-fold reports; may be populated by eval scripts).

