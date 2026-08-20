# Entry points: what to run, from where, with what

Every command below is submitted **from the repo root** (the directory that contains
`paths.py`). `sbatch` log lines are relative (`slurm-logs/...`); if you submit from
`OpenTAD/` or `pyskl/` the logs land in the wrong place.

```bash
git clone <rmmdet> && cd rmmdet
git submodule update --init OpenTAD
cd OpenTAD && git apply --whitespace=nowarn ../opentad_sails/sails_changes.patch && cd ..
# TriDet only: build align1d on a GPU node (opentad_sails/README.md)
rsync -a "$(python paths.py --get checkpoints_root)"/ .
conda activate <env>          # scripts assume conda lives at ~/miniconda3
```

| Env | `conda activate` | Used by |
| :--- | :--- | :--- |
| `vjepa2` | V-JEPA2, fusion, two-stage | most GPU eval |
| `opentad` | ActionFormer / TriDet | OpenTAD train/test |
| `pyskl` | PoseC3D / STGCN++ | skeleton train/test |
| `dataprep` | splits / pose pickles | CPU data prep |

Reported headline numbers are already in git (`insights/tables/`, `tal/eval_results/*/cv_summary.json`,
`two-stg/eval_results/cv_summary.json`). Re-running GPU jobs **checks the pipeline**; it is
not required to read the paper tables.

---

## A. Reproduce eval (no retraining)

### OpenTAD detectors — env `opentad`, 1 GPU, ~1 min/fold

Needs: overlay `best.pth`, annotation JSONs (in the SAILS patch + overlay), and
**align1d** for TriDet. Convert `--task` is `binary`\|`4class`; SLURM `TASK` is
`binary`\|`balanced` (see `OpenTAD/tools/prepare_data/sails_rmm/README.md`).

| Script | Args | Writes |
| :--- | :--- | :--- |
| `OpenTAD/slurm/test_actionformer_cv.sh` | `<binary\|balanced> <fold>` | `OpenTAD/exps/sails_rmm/actionformer_vjepa_<task>_fold<N>/gpu1_id99/result_detection.json` (**overwrites**) |
| `OpenTAD/slurm/test_tridet_cv.sh` | same | `.../tridet_vjepa_<task>_fold<N>/gpu1_id99/result_detection.json` |
| `OpenTAD/slurm/test_e2e_nomc_cv.sh` | `<actionformer\|tridet> <balanced\|binary> <fold>` | re-NMS into `gpu1_id98/` |

```bash
sbatch OpenTAD/slurm/test_actionformer_cv.sh binary 0
sbatch OpenTAD/slurm/test_tridet_cv.sh balanced 0
```

Two-stage Stage-1 reads **ActionFormer binary** `gpu1_id99`. Do not clobber that file
unless you intend to refresh Stage-1.

Optional train (not needed for reported numbers):
`OpenTAD/slurm/train_actionformer_cv.sh`, `train_tridet_cv.sh` — same `<task> <fold>`.

### Two-stage TAL — env `vjepa2`, 1 GPU, **128 GB RAM**, ~3 h/fold

Needs: ActionFormer binary `result_detection.json` (id99), V-JEPA2 `fold_N/model.safetensors`,
lab clips via `config.yaml`. Peak RSS ~42 GB; 32 GB OOMs.

| Script | How | Output |
| :--- | :--- | :--- |
| `two-stg/run_single_fold_gpu.sh` | `sbatch --export=FOLD=0 two-stg/run_single_fold_gpu.sh` | `two-stg/eval_results/fold0/` |
| `two-stg/run_two_stage_cv_gpu.sh` | `sbatch two-stg/run_two_stage_cv_gpu.sh` | all 3 folds, 4-class V-JEPA2 Stage-2 |
| `two-stg/run_three_way_fold_gpu.sh` | `sbatch --export=FOLD=0 …` | `two-stg/eval_results_3way/` — first `python fusion/export_three_way_deploy_checkpoint.py --all-folds` |
| `two-stg/run_three_way_cv_gpu.sh` | `sbatch` or `bash` (submits 3 jobs) | 3-way MLP Stage-2 |
| `two-stg/run_5class_fold_gpu.sh` / `run_5class_cv_gpu.sh` | same pattern | 5-class Stage-2 |
| `two-stg/run_tridet_single_fold_gpu.sh` | Stage-1 = TriDet binary instead of AF | `two-stg/eval_results_tridet/` |
| `two-stg/run_tridet_three_way_live_fold_gpu.sh` | TriDet + live 3-way | |
| `two-stg/run_three_way_live_fold_gpu.sh` | per-proposal skeleton scores | needs `run_proposal_skeleton_inference.sh` first |
| `two-stg/run_opentad_postprocess.sh` | CPU OpenTAD mAP on an existing `result_detection.json` | |
| `two-stg/run_two_stage_cv.sh` | CPU/login-node wrapper (no GPU sbatch) | |

Detail: `two-stg/README.md`. Committed fold metrics: `two-stg/eval_results/cv_summary.json`.

### Window-based TAL — env `vjepa2` or `opentad`, **CPU**

Needs: existing window prediction CSVs (in `v-jepa/runs/vjepa2_tal_cv_*` after overlay, or
pyskl TAL `result.pkl` + overlay TAL pickles). Committed summaries already live in
`tal/eval_results/*/cv_summary.json`.

| Script | How | Notes |
| :--- | :--- | :--- |
| `tal/test_tal_eval.py` | `python tal/test_tal_eval.py` | oracle sanity, mAP ≈ 1.0 |
| `tal/scripts/eval_tal_from_existing_preds.sh` | `sbatch tal/scripts/eval_tal_from_existing_preds.sh` | CPU; re-derives segments from CSVs |
| `tal/scripts/eval_vjepa_binary_cv.sh` / `eval_vjepa_balanced_cv.sh` | `sbatch` | window→segment for V-JEPA TAL runs |
| `tal/scripts/eval_best_postprocess.py` | `python tal/scripts/eval_best_postprocess.py` | uses `tal/eval_results/best_params_per_model.json` |
| `tal/scripts/eval_binary_tal.py` | `--predictions <csv>` | binary RMM-vs-BG |
| `tal/run_tal_eval_cv.py` | `--model vjepa --preds-root … --out-dir …` | CV aggregator |
| `tal/scripts/grid_search_postprocessing.py` | CPU sweep | overwrites best-params JSON |
| `tal/scripts/run_tal_fusion_oof.py` | CPU | writes fused window preds |

`tal/README.md` still shows `export_pyskl_window_preds.py` as `python export_…` from `tal/`;
from repo root use `python dataprep/tal/export_pyskl_window_preds.py` (the helper lives in
`dataprep/tal/`). V-JEPA `window_level_preds.csv` lacks `start_sec`/`end_sec` — join from
`dataprep/tal/splits_cv_*/fold_N_val_windows.csv` or use committed `tal_format_preds.csv`.

### Clip classification — env `vjepa2` / `pyskl`

| Script | Env | How | Needs |
| :--- | :--- | :--- | :--- |
| `v-jepa/slurm/rerun_vjepa_eval_4class.sh` | `vjepa2` | `sbatch v-jepa/slurm/rerun_vjepa_eval_4class.sh` | overlay `model.safetensors`, lab `classification_clips` + SAM3 cache; writes `clip_level_preds.csv` for fusion |
| `v-jepa/slurm/rerun_vjepa_eval_5class.sh` | `vjepa2` | same | 5-class |
| `pyskl/tools/test.py` | `pyskl` | `cd pyskl && python tools/test.py <work_dir>/joint.py -C <work_dir>/best_*.pth --launcher none --cfg-options data.test.split=val --out <out>.pkl --eval top_k_accuracy mean_class_accuracy` | pose pickle + `best_*.pth`. Use `--launcher none` (single-GPU); `dist_test.sh` is blocked by the DDP breakage in gotcha 5. Each `work_dir` holds the exact dumped config that produced its checkpoint — use that rather than the template in `configs/`. |
| `pyskl/scripts/slurm/test_regularized.sh` | `pyskl` | `sbatch` | same env/split issues |

Classification pickles: git `pyskl/data/sails/{cv,single,*.pkl}` (~15 MB). TAL window
pickles (~149 MB): overlay only (`pyskl/data/sails/tal/`).

### Late fusion — env `vjepa2`, 1 GPU, fast (fits α / MLP on frozen logits)

Needs prediction CSVs next to the checkpoints (now in the overlay: PoseC3D
`work_dirs/**/predictions_clip.csv`, V-JEPA `clip_level_preds.csv`). Fusion MLP weights
are already in `fusion/runs/` after overlay.

| Script | What it fuses |
| :--- | :--- |
| `fusion/slurm/run_fusion_cv.sh` | V-JEPA2 + PoseC3D, scalar α, 4-class |
| `fusion/slurm/run_fusion_mlp_cv.sh` | same, MLP |
| `fusion/slurm/run_fusion_perclass_cv.sh` | per-class α |
| `fusion/slurm/run_fusion_3way_cv.sh` | V-JEPA2 + PoseC3D + STGCN++ |
| `fusion/slurm/run_fusion_stgcn_cv.sh` / `run_fusion_stgcn_mlp_cv.sh` | V-JEPA2 + STGCN++ |
| `fusion/slurm/run_fusion_5class_*.sh` | 5-class variants |

```bash
sbatch fusion/slurm/run_fusion_cv.sh
```

---

## B. Retrain (optional; not needed for tables)

All of these write git-ignored `runs/` / `work_dirs/` / `exps/`. Submit from repo root.

| Task | Script | Env | Notes |
| :--- | :--- | :--- | :--- |
| V-JEPA2 4-class CV | `v-jepa/slurm/run_vjepa_cv.sh` | `vjepa2` | crop on by default; `ENABLE_CROP=0` to disable. 256 GB, 48 h |
| V-JEPA2 single split | `v-jepa/slurm/run_vjepa_single.sh` | `vjepa2` | |
| V-JEPA2 TAL windows | `v-jepa/slurm/tal/train_tal_cv_binary_balanced.sh`, `train_tal_cv_5class_balanced.sh` | `vjepa2` | |
| V-JEPA2 TAL 5-class + bg-subsample | `sbatch v-jepa/finetune_tal_cv.sbatch` | `vjepa2` | reads clips from `$TAL_CLIPS_ROOT` |
| V-JEPA2 feature extract | `v-jepa/slurm/extract_features.sh` | `vjepa2` | OpenTAD input `.npy` |
| PoseC3D weighted (reported) | `bash pyskl/scripts/slurm/posec3d/submit_all_weighted.sh` | `pyskl` | submits 4 jobs |
| STGCN++ weighted | `bash pyskl/scripts/slurm/stgcnpp/submit_all_weighted.sh` | `pyskl` | |
| PoseC3D/STGCN TAL | `pyskl/scripts/slurm/{posec3d,stgcnpp}/train_tal_*.sh` | `pyskl` | |
| OpenTAD train | `OpenTAD/slurm/train_{actionformer,tridet}_cv.sh` | `opentad` | |

Unweighted / focal / 4-stream variants live next to those files (`train_4class_cv.sh`,
`train_*_focal.sh`, …). Every `*.sh` / `*.sbatch` in the repo passes `bash -n`.

---

## C. Data prep (optional; caches already exist)

| Script | Env | What |
| :--- | :--- | :--- |
| `dataprep/tal/make_tal_video_assignments.py` | `dataprep` | per-fold video assignment |
| `dataprep/tal/make_tal_window_splits.py` | `dataprep` | 2 s / 1 s window CSVs |
| `dataprep/tal/validate_tal_splits.py` | `dataprep` | split QA |
| `dataprep/tal/cut_tal_windows.sbatch` | `dataprep` | cuts window clips to `$TAL_CLIPS_ROOT` |
| `dataprep/tal/create_tal_pose_pickles.sbatch` | `dataprep` | TAL window pickles for pyskl (all 4 configs) |
| `dataprep/tal/create_tal_pose_pickles_4class_only.sbatch` | `dataprep` | same, CV 4-class only |
| `pyskl/tools/data/create_sails_annotations.py` | `pyskl` | classification pose pickles |
| `dataprep/clip_gen/create_clip_segments.py` | `dataprep` | clip cutting |
| `dataprep/pose_gen/batch_sam_pose.py` | **mmpose/mmdet, not `dataprep`** | pose H5 caches (already in lab space) |

---

## Gotchas (clone check)

1. **Conda path.** Wrappers `source ~/miniconda3/etc/profile.d/conda.sh`. If conda lives
   elsewhere, export `REPO_ROOT` and edit that line, or `conda activate` before a
   non-sbatch run.
2. **Path defaults are clone-relative.** Every launcher (`*.sh`, `*.sbatch`) resolves
   `REPO_ROOT` by walking up to the directory holding `paths.py`, and every Python
   argparse default now derives from that same root, so no-arg invocations work from
   any clone. Submit from the repo root so `#SBATCH --output=slurm-logs/...` resolves.
3. **Fusion CSV names.** V-JEPA classification writes `clip_level_preds.csv`; PoseC3D
   writes `eval_val/predictions_clip.csv`. `fusion/train_fusion_cv.py` looks for both.
4. **`predictions_clip.csv` is gitignored** (`runs/`, plus the filename). Overlay now
   includes those CSVs so fusion can run without re-eval.
5. **pyskl: use `--launcher none`, not `dist_test.sh`.** DDP is broken on the pinned
   `envs/pyskl.yml` (Torch 2.9 against mmcv-full 1.7, which raises
   `AttributeError: '_use_replicated_tensor_module'`). `tools/test.py --launcher none`
   runs single-GPU via `MMDataParallel` and avoids that path entirely. CV fold pickles
   have only `train`/`val`, so add `--cfg-options data.test.split=val`.
6. **Qwen VLM baseline.** Metrics/predictions are vendored at
   `insights/vlm/qwen_outputs/` (~1.3 MB), so `scripts/extract_precision_recall.py`
   runs from a clone with no overlay. The source videos/model are not included.
7. **`tal/` vs `dataprep/tal/` overlap.** Both hold `run_tal_eval_cv.py`,
   `tal_map_eval.py`, and `test_tal_eval.py`, and the copies have **drifted**. The
   reported TAL numbers come from the `tal/` copies; treat `dataprep/tal/` as the
   split-generation side and edit the two deliberately.
8. **`tal/analyze_tal_results.py` needs `--stgcn-log`.** It scrapes per-class metrics
   from an ST-GCN++ training `.err` log, which is a SLURM artifact not shipped here;
   pass a log from your own run.

Pipeline map and expected numbers: [`REPRODUCE.md`](../REPRODUCE.md). File catalog:
[`INDEX.md`](INDEX.md). Data/checkpoint locations: [`ARTIFACTS.md`](ARTIFACTS.md).
