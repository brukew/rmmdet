# Reproducing the actreg pipeline

End-to-end map of the project, in dependency order. Each stage links to the component
README that has the detailed commands. Data/artifact locations are documented in
[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md).

> **Filesystem locations** are resolved from a single source of truth,
> [`config.yaml`](config.yaml) via [`paths.py`](paths.py). Shared SAILS data now lives in
> the lab's durable project space under `/orcd/data/satra/002/projects/SAILS/` (migrated
> off `/orcd/scratch`, which has been purged). Run `python paths.py` to print every
> resolved path, or `python paths.py --export` to emit `export KEY=VALUE` lines for shell
> scripts.

> Four conda envs are used: **`pyskl`** (PoseC3D/STGCN++), **`vjepa2`** (V-JEPA2 +
> two-stage), **`opentad`** (OpenTAD TAL eval), and **`dataprep`** (window cutting +
> pyskl-format TAL pose-pickle creation, i.e. `dataprep/tal/*.sbatch`). All four are
> pinned in [`envs/`](envs/) and each includes PyYAML (required by `paths.py`).
>
> **Not pinned:** the actual *pose generation* step (`dataprep/pose_gen/batch_sam_pose.py`,
> HRNet + SAM3 mask inference) needs a separate `mmpose`/`mmdet` environment. This step is
> **optional for reproducing results** — the SAM3 pose H5 caches it produces already exist
> under `cache_for_tracking/pose_sam3/` and are consumed directly downstream.

---

## Data availability & verification

All inputs required to reproduce the reported results are present under the lab project
space `/orcd/data/satra/002/projects/SAILS/`. Every path in `config.yaml` resolves on disk
except two directories, both of which are **non-blocking**:

| Absent path | Why it does not block reproduction |
| :--- | :--- |
| `classification_clips_cropped` | Output-only crop target of `dataprep/clip_gen/create_clip_segments.py`; nothing reads it. V-JEPA crops *live* from `classification_clips` + the `cache_for_tracking` SAM3 mask cache (`run_vjepa_*.sh --enable-crop`). |
| `rmm/videos` | Only the fallback `--videos-root` default in `v-jepa/tools/extract_vjepa_features.py`; videos are resolved from each window-split CSV's absolute `video_path` under `dataset_root/Phase_III_Videos/` (present), and features are already pre-extracted. |

Trained weights and materialized metrics are on disk: V-JEPA (`.safetensors`),
PoseC3D / STGCN++ / OpenTAD (`.pth`), OpenTAD `result_detection.json`, fusion runs, and
per-fold `metrics.json` / `cv_summary.json`. **Stages 3–6 can be re-derived without
retraining.**

**Smoke-tested (fold 0, Aug 2026).** CPU stages pass: TAL split validation, TAL oracle
(`test_tal_eval.py`, mAP≈1.0), classifier-effect analysis, and `eval_best_postprocess`.
GPU inference via SLURM against the migrated lab paths: OpenTAD TriDet test
(→ `result_detection.json`, avg-mAP 31.7%) and PoseC3D eval (Top-1 78.3%) both pass; the
two-stage eval loads the binary detection JSON + V-JEPA2 classifier from the lab paths and
runs end-to-end.

---

## 0. Setup

```bash
git clone <this-repo> actreg && cd actreg

# OpenTAD is a submodule pinned to upstream 1aa8ca4; the SAILS-RMM changes are a patch.
git submodule update --init OpenTAD
cd OpenTAD && git apply --whitespace=nowarn ../opentad_sails/sails_changes.patch && cd ..
# (then rebuild OpenTAD's align1d CUDA op -- see opentad_sails/README.md)

# Environments
conda env create -f envs/pyskl.yml
conda env create -f envs/vjepa2.yml
conda env create -f envs/opentad.yml
conda env create -f envs/dataprep.yml   # pose gen (mmpose/mmdet) + TAL pose pickles
```

See [`opentad_sails/README.md`](opentad_sails/README.md) for the OpenTAD fork details.

---

## 1. Data preparation

| Step | Command / location | README |
| :--- | :--- | :--- |
| Clip generation (654 segment clips) | `dataprep/clip_gen/` | — |
| Pose (HRNet) + SAM3 masks | `dataprep/pose_gen/`, `dataprep/sam3/` | — |
| Classification splits (4/5-class, single + CV, LCTO-safe) | `dataprep/splits/` | `README.md` §Data Preparation |
| TAL window splits (2s/1s windows) | `python dataprep/tal/make_tal_window_splits.py --task 4class --mode cv` | [`dataprep/tal/README.md`](dataprep/tal/README.md) |

Clips and features live under the lab project space
`/orcd/data/satra/002/projects/SAILS/` (see `config.yaml` and `docs/ARTIFACTS.md`).

---

## 2. Clip-classification models

| Model | Entry point | README |
| :--- | :--- | :--- |
| PoseC3D | `pyskl/` → `bash scripts/slurm/posec3d/submit_all_weighted.sh` | [`README.md`](README.md) §1, `pyskl/configs/posec3d/slowonly_r50_sails_k400p/README.md` |
| STGCN++ (4-stream) | `pyskl/` → `bash scripts/slurm/stgcnpp/submit_all_weighted.sh` | [`README.md`](README.md) §2 |
| V-JEPA2 (ViT-L finetune) | `v-jepa/` → `sbatch slurm/run_vjepa_cv.sh` | [`README.md`](README.md) §3 |
| Late fusion (V-JEPA2 + PoseC3D, scalar α) | `sbatch fusion/slurm/run_fusion_cv.sh` | [`README.md`](README.md) §5 |

---

## 3. TAL — window-based evaluation (`tal/`)

Window class scores → postprocess into segments → mAP@tIoU{0.3,0.5,0.7}, for V-JEPA2,
PoseC3D, STGCN++, and a log-prob MLP late fusion.

```bash
cd tal
python test_tal_eval.py                       # sanity (oracle mAP ~ 1.0)
python run_tal_eval_cv.py --model vjepa  --preds-root <vjepa_runs> --out-dir eval_results/vjepa_cv
python scripts/run_tal_fusion_oof.py          # fused preds
python scripts/grid_search_postprocessing.py  # tune postproc
python scripts/eval_best_postprocess.py       # final per-model mAP
```

> **Reproducibility note (final numbers are materialized).** The reported TAL mAP for every
> model variant is already tracked under `tal/eval_results/<variant>/cv_summary.json`, and the
> per-fold `tal_format_preds.csv` are committed. The final tables can therefore be re-derived
> *without re-running GPU jobs* via `scripts/grid_search_postprocessing.py` →
> `scripts/eval_best_postprocess.py` on the committed predictions.
>
> The from-scratch `run_tal_eval_cv.py --model vjepa` path is currently **stale for the V-JEPA
> branch**: V-JEPA `window_level_preds.csv` files do not carry per-window `start_sec`/`end_sec`,
> which must be joined from `dataprep/tal/splits_cv_*/fold_{N}_val_windows.csv` before
> `window_to_segments`. Until that join is restored, regenerate V-JEPA segment predictions from
> the committed `tal_format_preds.csv` rather than from raw window scores. The `posec3d`/`stgcn`
> branches (which export timing from the pyskl annotation pickle) are unaffected.
>
> Note also that `tal/` and `dataprep/tal/` currently hold parallel copies of the eval helpers
> (`run_tal_eval_cv.py`, `window_to_segments.py`, `tal_map_eval.py`, …); `tal_map_eval.py`
> differs between them. Treat `tal/` as the eval entry point (its `run_tal_eval_cv.py` loads the
> shared `export_pyskl_window_preds` helper from `dataprep/tal/` by explicit path).

Full detail: [`tal/README.md`](tal/README.md).

---

## 4. TAL — OpenTAD E2E + binary detectors (`OpenTAD/`)

ActionFormer / TriDet trained on V-JEPA2 features, both **balanced** (E2E multiclass) and
**binary** (single "action" class, used as Stage-1 of the two-stage pipeline).

```bash
cd OpenTAD
# splits -> OpenTAD annotation JSON (per fold + task; --fold and --task are REQUIRED)
for fold in 0 1 2; do
  python tools/prepare_data/sails_rmm/convert_cv_splits_to_opentad_json.py --fold $fold --task binary
  python tools/prepare_data/sails_rmm/convert_cv_splits_to_opentad_json.py --fold $fold --task 4class
done
# train (args: <task> <fold>, task in {balanced,binary})
for fold in 0 1 2; do sbatch slurm/train_actionformer_cv.sh balanced $fold; done
for fold in 0 1 2; do sbatch slurm/train_tridet_cv.sh       binary   $fold; done
# test -> result_detection.json (best.pth per fold; args: <task> <fold>)
for fold in 0 1 2; do sbatch slurm/test_tridet_cv.sh binary $fold; done
```

> Feature `data_path`s in `configs/_base_/datasets/sails_rmm/features_vjepa_*.py` resolve to
> the lab project space (`/orcd/data/satra/002/projects/SAILS/rmm/features/...`).

> The model picks `best.pth` by **minimum validation loss**, which for the multiclass E2E
> runs is *not* the peak-mAP epoch — peak mAP comes a few epochs later. Reported E2E
> numbers use peak-epoch mAP from the training log; binary detectors' `best.pth` already
> coincides with their peak. (Binary detector `best.pth` is what feeds Stage-1.)

`result_detection.json` outputs are git-ignored; copy paths feed Stage 5.

---

## 5. Two-stage TAL (`two-stg/`)

Binary detector (Stage 1) + V-JEPA2 / 3-way-fusion-MLP / 5-class classifier (Stage 2),
evaluated with native OpenTAD mAP.

```bash
cd /orcd/data/satra/001/users/brukew/actreg
bash two-stg/run_two_stage_cv_gpu.sh          # 4-class V-JEPA2 Stage-2
bash two-stg/run_three_way_cv_gpu.sh          # 3-way fusion MLP Stage-2
bash two-stg/run_5class_cv_gpu.sh             # 5-class (+ background) Stage-2
```

Full detail: [`two-stg/README.md`](two-stg/README.md).

---

## 6. Analysis & paper artifacts

- Classifier-effect / two-stage-vs-E2E analysis: `two-stg/analyze_classifier_effect.py`,
  `two-stg/ANALYSIS_NOTES.md`, `two-stg/RUNS.md`.
- Failure analysis, figures, tables: `insights/`.
- Result tables already tracked as `metrics*.json` / `cv_summary.json`; reported mAP can
  be re-derived from these without re-running GPU jobs.

> **Paper tables are the source of truth.** The committed
> `insights/tables/table_5_*.json` were finalized from richer per-model metrics than
> `insights/paper_plots/generate_all_table_jsons.py` currently reads. Re-running that
> generator today produces a *subset* — it nulls out `cohens_kappa`/`macro_f1`/`top2_acc`,
> drops the `per_class_ap` breakdown, and drops some std values. Treat the committed
> `insights/tables/*.json` as authoritative and do **not** overwrite them with the current
> generator output.

---

## Known gaps for the next maintainer

- `OpenTAD/tools/prepare_data/sails_rmm/README.md` is empty — the convert step is captured
  in the patch and in the script's own `--help`; the required `--fold`/`--task` args are
  documented in §4 above.
- The paper-table generator and the from-scratch V-JEPA `run_tal_eval_cv.py` path are both
  stale (see the notes in §3 and §6); final numbers are reproduced from the committed
  `cv_summary.json` / `tal_format_preds.csv` / `insights/tables/*.json` instead.
- Filesystem paths are now centralized in [`config.yaml`](config.yaml) + [`paths.py`](paths.py)
  (the earlier hardcoded-`/orcd/scratch` paths have been migrated to the lab project space).
