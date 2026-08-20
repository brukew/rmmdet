# Reproducing the actreg pipeline

End-to-end map of the project, in dependency order. Each stage links to the component
README that has the detailed commands. Data/artifact locations are documented in
[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md); a catalog of every launch/eval script (env, args,
inputs, outputs) is [`docs/ENTRYPOINTS.md`](docs/ENTRYPOINTS.md); a catalog of every
doc/file is [`docs/INDEX.md`](docs/INDEX.md).

> **Filesystem locations** are resolved from a single source of truth,
> [`config.yaml`](config.yaml) via [`paths.py`](paths.py). Shared SAILS data lives in the
> lab's durable project space under `/orcd/data/satra/002/projects/SAILS/` (mirrored off the
> shared, impermanent `/orcd/scratch`, which may be cleaned at any time). Run `python
> paths.py` to print every resolved path, or `python paths.py --export` to emit
> `export KEY=VALUE` lines for shell scripts.

> **Running the launch/eval scripts (location-independent).** Every `*.sh` under this repo
> resolves its own repo root at runtime (the directory containing `paths.py`) and references
> data/checkpoints through it, so a fresh clone of any name works out of the box. **Submit
> SLURM jobs from the repo root** (e.g. `sbatch two-stg/run_single_fold_gpu.sh`,
> `sbatch OpenTAD/slurm/test_tridet_cv.sh binary 0`); job logs are written to the tracked
> `slurm-logs/` at the repo root. Under `sbatch`, root resolution falls back to
> `$SLURM_SUBMIT_DIR`; you can also force it with `REPO_ROOT=/path/to/clone sbatch …`.
> Trained weights are git-ignored — overlay the shared checkpoint copy first (see
> [Where the checkpoints live](#where-the-checkpoints-live)).

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
The authoritative tables are [`insights/tables/*.json`](insights/tables/). Do not
overwrite those JSON files with `insights/paper_plots/generate_all_table_jsons.py` — that
generator currently emits a thinner subset. Per-fold metrics live under each component's
`eval_results/`.

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

> **Most common setup failures (clone check, Aug 2026):**
> 1. OpenTAD's `align1d` CUDA op is not built — TriDet cannot import. Build it on a GPU
>    node after applying the patch (`opentad_sails/README.md`). ActionFormer does not need it.
> 2. Checkpoints are git-ignored — overlay `checkpoints_root` before any test script
>    (weights, OpenTAD annotation JSONs, and pyskl pickles all live there; classification
>    pickles and the JSONs are also in git / the SAILS patch after `git apply`).
> 3. Submit `sbatch` from the **repo root**, not from `OpenTAD/`, so logs land in
>    `slurm-logs/`.

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
Full detail: [`tal/README.md`](tal/README.md) and [`docs/ENTRYPOINTS.md`](docs/ENTRYPOINTS.md).

---

## 4. TAL — OpenTAD E2E + binary detectors (`OpenTAD/`)  ·  env: `opentad`

ActionFormer / TriDet trained on V-JEPA2 features, both **balanced** (E2E 4-class) and
**binary** (single "action" class, used as Stage-1 of the two-stage pipeline).

### Eval from a fresh clone (order matters)

These four steps are easy to skip or mix up; each failed during the Aug 2026 clone check:

1. **Submodule + patch + `align1d` build** (setup §0). TriDet imports the CUDA ROI-align
   op; ActionFormer is anchor-free and does not need it. Build on a GPU node
   (`opentad_sails/README.md`). Without the `.so`, TriDet dies at import.
2. **Overlay checkpoints** (`rsync` in [Where the checkpoints live](#where-the-checkpoints-live)).
   `best.pth` is git-ignored; test scripts exit if it is missing. The overlay also carries
   annotation JSONs and pyskl pickles.
3. **Annotation JSONs** ship with `git apply opentad_sails/sails_changes.patch` (and with
   the overlay). Re-run convert only if you change splits:

   ```bash
   cd OpenTAD && conda activate opentad
   for fold in 0 1 2; do
     python tools/prepare_data/sails_rmm/convert_cv_splits_to_opentad_json.py --fold $fold --task binary
     python tools/prepare_data/sails_rmm/convert_cv_splits_to_opentad_json.py --fold $fold --task 4class
   done
   ```

   **Name mismatch:** convert `--task` is `binary` \| `4class`. SLURM / config `TASK` is
   `binary` \| `balanced`. There is no `--task balanced`. `--task 4class` writes
   `fold{N}_anno.json`, which the `*balanced*` configs read. Full table:
   [`OpenTAD/tools/prepare_data/sails_rmm/README.md`](OpenTAD/tools/prepare_data/sails_rmm/README.md).
4. **Submit tests from the repo root** (so `#SBATCH --output=slurm-logs/...` lands in the
   tracked `slurm-logs/` next to `paths.py`, not inside `OpenTAD/`):

   ```bash
   # from repo root; args: <task> <fold>, task in {binary,balanced}
   sbatch OpenTAD/slurm/test_actionformer_cv.sh binary 0
   sbatch OpenTAD/slurm/test_actionformer_cv.sh balanced 0
   sbatch OpenTAD/slurm/test_tridet_cv.sh       binary 0
   sbatch OpenTAD/slurm/test_tridet_cv.sh       balanced 0
   ```

Each test writes `OpenTAD/exps/sails_rmm/<model>_vjepa_<task>_fold<N>/gpu1_id99/result_detection.json`
and **overwrites** any existing id99 JSON. Two-stage Stage-1 reads the **ActionFormer
binary** id99 file; re-running that test is what refreshes Stage-1 proposals.

Fold-0 clone check (overlay + convert + `tools/test.py`, Aug 2026): ActionFormer binary
avg-mAP 28.92%, ActionFormer balanced 16.55%, TriDet binary 31.72%, TriDet balanced 18.47%.
(Reported paper numbers are 3-fold CV / peak-epoch; see below.)

### Train (optional; not needed for the reported numbers)

```bash
cd OpenTAD
for fold in 0 1 2; do sbatch slurm/train_actionformer_cv.sh balanced $fold; done
for fold in 0 1 2; do sbatch slurm/train_tridet_cv.sh       binary   $fold; done
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
cd <this-repo>
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

The trained checkpoints are **git-ignored** — a `git clone` does **not** include them. A
~24 GB lean copy of the reported-result finals lives in the durable lab project space at
`checkpoints_root`, using the same repo-relative layout. **For the no-retrain path, overlay
it into the working tree first:**

```bash
rsync -a "$(python paths.py --get checkpoints_root)"/ .   # -> /orcd/data/satra/002/projects/SAILS/checkpoints
```

This populates the git-ignored output roots so every eval/test step below finds its weights
at the path it already expects:

| Artifact | Location (after overlay) |
| :--- | :--- |
| V-JEPA2 classifiers (`model.safetensors`) | `v-jepa/runs/vjepa2_rmm_cv/...`, `v-jepa/runs/vjepa2_tal_cv_*/fold_{0,1,2}` |
| OpenTAD detectors (`best.pth` + `result_detection.json`) | `OpenTAD/exps/sails_rmm/{actionformer,tridet}_vjepa_{binary,balanced}_fold{0,1,2}/gpu1_id*/` |
| PoseC3D / STGCN++ (`best_*.pth`) | `pyskl/work_dirs/{posec3d,stgcnpp}/...` |
| Fusion MLPs | `fusion/runs/` |
| Materialized metrics/tables (already in git) | `tal/eval_results/*/cv_summary.json`, `two-stg/eval_results/cv_summary.json`, `insights/tables/*.json` |

`docs/ARTIFACTS.md` has the full inventory (what's copied vs. regenerable) and the git-ignore
rules.

---

## Data availability & verification

All inputs required to reproduce the reported results are present under
`/orcd/data/satra/002/projects/SAILS/`. Every path in `config.yaml` resolves on disk except
two directories, both **non-blocking**:

| Absent path | Why it does not block reproduction |
| :--- | :--- |
| `classification_clips_cropped` | Output-only crop target of `dataprep/clip_gen/create_clip_segments.py`; nothing reads it. V-JEPA crops *live* from `classification_clips` + the `cache_for_tracking` SAM3 mask cache (`run_vjepa_*.sh --enable-crop`). |
| `rmm/videos` | Only the fallback `--videos-root` default in `v-jepa/tools/extract_vjepa_features.py`; videos resolve from each window-split CSV's absolute `video_path` under `dataset_root/Phase_III_Videos/` (present), and features are already pre-extracted. |

**Smoke-tested from a fresh clone of `clean-repo` (fold 0, Aug 2026).** Clone → submodule
+ patch → checkpoint overlay → convert annotations → SLURM eval. Path-resolution audit of
every reported eval input passed. GPU inference:

| Job | Loaded ckpt + data | Inference | Wrote |
| :--- | :--- | :--- | :--- |
| OpenTAD TriDet binary | ✓ | avg-mAP 31.72% | `.../gpu1_id99/result_detection.json` |
| OpenTAD TriDet balanced | ✓ | avg-mAP 18.47% | `.../gpu1_id99/result_detection.json` |
| OpenTAD ActionFormer binary | ✓ | avg-mAP 28.92% | `.../gpu1_id97/` (id99 left for two-stage) |
| OpenTAD ActionFormer balanced | ✓ | avg-mAP 16.55% | `.../gpu1_id97/result_detection.json` |
| Two-stage (AF-binary → V-JEPA2) | ✓ | in-progress / load+infer verified | clone-relative |

CPU stages (TAL split validation, TAL oracle `test_tal_eval.py` mAP≈1.0, convert) pass.

**Not a clean clone pass:** PoseC3D/STGCN++ `pyskl/tools/test.py` (pinned `envs/pyskl.yml`
is Torch 2.9 + mmcv-full 1.7, which breaks DDP). See [`docs/ENTRYPOINTS.md`](docs/ENTRYPOINTS.md).
Classification headline numbers were produced before that env drift.
