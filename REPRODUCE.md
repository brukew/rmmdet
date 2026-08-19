# Reproducing the actreg pipeline

End-to-end map of the project, in dependency order. Each stage links to the component
README that has the detailed commands. Data/artifact locations are documented in
[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md); a catalog of every doc/entry point is in
[`docs/INDEX.md`](docs/INDEX.md).

> **Filesystem locations** are resolved from a single source of truth,
> [`config.yaml`](config.yaml) via [`paths.py`](paths.py). Shared SAILS data lives in the
> lab's durable project space under `/orcd/data/satra/002/projects/SAILS/` (migrated off
> `/orcd/scratch`, which has been purged). Run `python paths.py` to print every resolved
> path, or `python paths.py --export` to emit `export KEY=VALUE` lines for shell scripts.

> **Do the reported numbers require retraining?** No. Trained weights and materialized
> metrics are committed/on-disk (see [Expected results](#expected-results) and
> [Where the checkpoints live](#where-the-checkpoints-live)), so **stages 3–6 can be
> re-derived without any GPU training**. Retraining paths are documented for completeness.

---

## Conda environments

Four envs are pinned in [`envs/`](envs/); each includes PyYAML (required by `paths.py`):

| Env | Used for | Stages |
| :--- | :--- | :--- |
| `dataprep` | window cutting + pyskl-format TAL pose-pickle creation | 1 |
| `pyskl` | PoseC3D / STGCN++ train + eval | 2 |
| `vjepa2` | V-JEPA2 finetune/extract, fusion, two-stage eval | 2, 3, 5, 6 |
| `opentad` | OpenTAD ActionFormer/TriDet train + eval | 4 |

The window-based TAL eval (`tal/`, stage 3) only needs numpy/pandas/scikit-learn and runs
in `vjepa2` or `opentad`.

> **Not pinned (optional):** the actual *pose generation* step
> (`dataprep/pose_gen/batch_sam_pose.py`, HRNet + SAM3 mask inference) needs a separate
> `mmpose`/`mmdet` environment. It is **not required to reproduce results** — the SAM3 pose
> H5 caches it produces already exist under `cache_for_tracking/pose_sam3/` and are consumed
> directly downstream.

---

## Expected results

Confirm a successful reproduction against these committed headline numbers (3-fold CV).
The authoritative tables are [`insights/tables/*.json`](insights/tables/) (see also the
[caveat](#known-gaps-for-the-next-maintainer) on the table generator); per-fold metrics are
under each component's `eval_results/`.

**Clip classification — CV macro-F1** (`table_5_1`/`table_5_2`):

| Task | Best fusion (3-way MLP) | Best single model |
| :--- | :--- | :--- |
| 4-class | **82.1%** | V-JEPA2+SAM3-crop 75.1%, STGCN++ 72.7%, PoseC3D 70.4% |
| 5-class | **69.8%** | PoseC3D+weighted 65.9%, V-JEPA2+crop 63.0% |

**Temporal action localization — segment avg-mAP@tIoU{0.3,0.5,0.7}** (`table_5_7`/`5_9`,
`two-stg/RESULTS.md`):

| Model | avg-mAP | Notes |
| :--- | :--- | :--- |
| ActionFormer **binary** | 27.93% | RMM-vs-background detection only (Stage-1) |
| **Two-stage** (AF-binary → V-JEPA2) | **17.83%** | best 4-class TAL model |
| ActionFormer balanced (E2E 4-class) | 16.70% | peak-epoch mAP (see §4) |
| V-JEPA2 + PoseC3D MLP (window-based) | 6.36% | best window→segment fusion |

---

## 0. Setup

```bash
git clone <this-repo> actreg && cd actreg

# OpenTAD is a submodule pinned to upstream 1aa8ca4; the SAILS-RMM changes are a patch.
git submodule update --init OpenTAD
cd OpenTAD && git apply --whitespace=nowarn ../opentad_sails/sails_changes.patch && cd ..
```

> **Most common setup failure:** OpenTAD's `align1d` CUDA op must be rebuilt for the local
> toolchain before any TAL train/eval. Follow the build step in
> [`opentad_sails/README.md`](opentad_sails/README.md).

```bash
# Environments (see the table above)
conda env create -f envs/dataprep.yml
conda env create -f envs/pyskl.yml
conda env create -f envs/vjepa2.yml
conda env create -f envs/opentad.yml
```

---

## 1. Data preparation  ·  env: `dataprep`

| Step | Command / location | README |
| :--- | :--- | :--- |
| Clip generation (654 segment clips) | `dataprep/clip_gen/` | — |
| Pose (HRNet) + SAM3 masks *(optional; caches exist)* | `dataprep/pose_gen/`, `dataprep/sam3/` | — |
| Classification splits (4/5-class, single + CV, LCTO-safe) | `dataprep/splits/` | `README.md` §Data Preparation |
| TAL window splits (2s/1s windows) | `python dataprep/tal/make_tal_window_splits.py --task 4class --mode cv` | [`dataprep/tal/README.md`](dataprep/tal/README.md) |

Clips and features live under the lab project space
`/orcd/data/satra/002/projects/SAILS/` (see `config.yaml` and `docs/ARTIFACTS.md`).

---

## 2. Clip-classification models  ·  envs: `pyskl`, `vjepa2`

| Model | Entry point | README |
| :--- | :--- | :--- |
| PoseC3D | `bash pyskl/scripts/slurm/posec3d/submit_all_weighted.sh` | [`README.md`](README.md) §1, `pyskl/configs/posec3d/slowonly_r50_sails_k400p/README.md` |
| STGCN++ (4-stream) | `bash pyskl/scripts/slurm/stgcnpp/submit_all_weighted.sh` | [`README.md`](README.md) §2 |
| V-JEPA2 (ViT-L finetune) | `sbatch v-jepa/slurm/run_vjepa_cv.sh` | [`README.md`](README.md) §3 |
| Late fusion (V-JEPA2 + PoseC3D, scalar α) | `sbatch fusion/slurm/run_fusion_cv.sh` | [`README.md`](README.md) §5 |

---

## 3. TAL — window-based evaluation (`tal/`)  ·  env: `vjepa2`/`opentad`

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

The final tables are re-derivable from the committed per-fold `tal_format_preds.csv` via
`grid_search_postprocessing.py` → `eval_best_postprocess.py`, without re-running GPU jobs.
See the [known gaps](#known-gaps-for-the-next-maintainer) for two stale from-scratch paths.
Full detail: [`tal/README.md`](tal/README.md).

---

## 4. TAL — OpenTAD E2E + binary detectors (`OpenTAD/`)  ·  env: `opentad`

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

Feature `data_path`s in `configs/_base_/datasets/sails_rmm/features_vjepa_*.py` resolve to
the lab project space (`/orcd/data/satra/002/projects/SAILS/rmm/features/...`).

> The model picks `best.pth` by **minimum validation loss**, which for the multiclass E2E
> runs is *not* the peak-mAP epoch — peak mAP comes a few epochs later. Reported E2E numbers
> use peak-epoch mAP from the training log; binary detectors' `best.pth` already coincides
> with their peak. (Binary detector `best.pth` is what feeds Stage-1.)

`result_detection.json` outputs are git-ignored; their paths feed Stage 5.

---

## 5. Two-stage TAL (`two-stg/`)  ·  env: `vjepa2`  ·  128 GB RAM / GPU

Binary detector (Stage 1) + V-JEPA2 / 3-way-fusion-MLP / 5-class classifier (Stage 2),
evaluated with native OpenTAD mAP.

```bash
cd /orcd/data/satra/001/users/brukew/actreg
sbatch two-stg/run_two_stage_cv_gpu.sh        # 4-class V-JEPA2 Stage-2
sbatch two-stg/run_three_way_cv_gpu.sh        # 3-way fusion MLP Stage-2
sbatch two-stg/run_5class_cv_gpu.sh           # 5-class (+ background) Stage-2
```

> **Resource note:** fold-0 (8154 proposals) peaks at ~42 GB RSS and runs ~3 h on one GPU;
> the fold wrappers request 128 GB (a 32 GB request OOMs). Verified fold-0 reproduces
> `average_mAP = 0.2323` exactly on the migrated lab paths.

Full detail: [`two-stg/README.md`](two-stg/README.md).

---

## 6. Analysis & paper artifacts

- Classifier-effect / two-stage-vs-E2E analysis: `two-stg/analyze_classifier_effect.py`,
  `two-stg/ANALYSIS_NOTES.md`, `two-stg/RUNS.md`.
- Failure analysis, figures, tables: `insights/`.
- Headline numbers are re-derivable from committed `metrics*.json` / `cv_summary.json` /
  `insights/tables/*.json` without re-running GPU jobs (see [Expected results](#expected-results)).

---

## Where the checkpoints live

For the no-retrain path (run only the eval/test steps of stages 3–6):

| Artifact | Location |
| :--- | :--- |
| V-JEPA2 classifiers (`.safetensors`) | `v-jepa/runs/vjepa2_rmm_cv/.../fold_{0,1,2}` |
| OpenTAD detectors (`best.pth` + `result_detection.json`) | `OpenTAD/exps/sails_rmm/{actionformer,tridet}_vjepa_{binary,balanced}_fold{0,1,2}/gpu1_id*/` |
| PoseC3D / STGCN++ (`.pth`) | `pyskl/work_dirs/` |
| Fusion runs | `fusion/runs/` |
| Materialized metrics/tables | `tal/eval_results/*/cv_summary.json`, `two-stg/eval_results/cv_summary.json`, `insights/tables/*.json` |

`docs/ARTIFACTS.md` has the full retention policy and the data-preservation risk list.

---

## Data availability & verification

All inputs required to reproduce the reported results are present under
`/orcd/data/satra/002/projects/SAILS/`. Every path in `config.yaml` resolves on disk except
two directories, both **non-blocking**:

| Absent path | Why it does not block reproduction |
| :--- | :--- |
| `classification_clips_cropped` | Output-only crop target of `dataprep/clip_gen/create_clip_segments.py`; nothing reads it. V-JEPA crops *live* from `classification_clips` + the `cache_for_tracking` SAM3 mask cache (`run_vjepa_*.sh --enable-crop`). |
| `rmm/videos` | Only the fallback `--videos-root` default in `v-jepa/tools/extract_vjepa_features.py`; videos resolve from each window-split CSV's absolute `video_path` under `dataset_root/Phase_III_Videos/` (present), and features are already pre-extracted. |

**Smoke-tested (fold 0, Aug 2026).** CPU stages pass: TAL split validation, TAL oracle
(`test_tal_eval.py`, mAP≈1.0), classifier-effect analysis, `eval_best_postprocess`,
OpenTAD split→JSON convert (idempotent), and fusion train/export. GPU inference via SLURM
against the migrated lab paths all pass end-to-end:

- OpenTAD TriDet test → `result_detection.json` (fold-0 avg-mAP 31.7%).
- PoseC3D eval (fold-0 Top-1 78.3%, κ 0.62).
- V-JEPA2 **crop** eval (fold-0 Top-1 0.820, κ 0.734; `crop_box` populated → live SAM3
  crop path verified).
- Two-stage eval (fold-0 `average_mAP` 0.2323 — reproduces the committed result exactly).

---

## Known gaps for the next maintainer

- **Paper tables are the source of truth.** The committed `insights/tables/table_5_*.json`
  were finalized from richer per-model metrics than
  `insights/paper_plots/generate_all_table_jsons.py` currently reads. Re-running that
  generator today produces a *subset* (nulls `cohens_kappa`/`macro_f1`/`top2_acc`, drops
  `per_class_ap` and some stds). Treat the committed tables as authoritative and do **not**
  overwrite them with the current generator output.
- **From-scratch V-JEPA window TAL path is stale.** V-JEPA `window_level_preds.csv` files do
  not carry per-window `start_sec`/`end_sec` (must be joined from
  `dataprep/tal/splits_cv_*/fold_{N}_val_windows.csv` before `window_to_segments`).
  Regenerate V-JEPA segment predictions from the committed `tal_format_preds.csv` instead.
  The `posec3d`/`stgcn` branches (which export timing from the pyskl annotation pickle) are
  unaffected.
- **`tal/` vs `dataprep/tal/` duplication.** Both hold parallel copies of the eval helpers
  (`run_tal_eval_cv.py`, `window_to_segments.py`, `tal_map_eval.py`); `tal_map_eval.py`
  differs between them. Treat `tal/` as the eval entry point (its `run_tal_eval_cv.py` loads
  the shared `export_pyskl_window_preds` helper from `dataprep/tal/` by explicit path).
- `OpenTAD/tools/prepare_data/sails_rmm/README.md` is empty — the convert step's required
  `--fold`/`--task` args are documented in §4 and in the script's `--help`.
- Filesystem paths are centralized in [`config.yaml`](config.yaml) + [`paths.py`](paths.py)
  (the earlier hardcoded-`/orcd/scratch` paths were migrated to the lab project space).
