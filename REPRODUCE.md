# Reproducing the actreg pipeline

End-to-end map of the project, in dependency order. Each stage links to the component
README that has the detailed commands. Data/artifact locations are documented in
[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md); a catalog of every doc/entry point is in
[`docs/INDEX.md`](docs/INDEX.md).

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
See the [known gaps](#known-gaps-for-the-next-maintainer) for two stale from-scratch paths.
Full detail: [`tal/README.md`](tal/README.md).

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

**Not a clean clone pass:** PoseC3D/STGCN++ `pyskl/tools/test.py` — see known gaps
(env torch/mmcv DDP mismatch; CV pickles have `train`/`val` only while configs set
`split='test'`; `--cfg-options` is not accepted by this `test.py`). Classification
numbers in the tables were produced before that env drift. Overlay includes
`pyskl/work_dirs/**/best_*.pth` but **not** `pyskl/data/sails/**/*.pkl` (generate with
`pyskl/tools/data/create_sails_annotations.py`, see
`pyskl/configs/posec3d/slowonly_r50_sails_k400p/README.md`).

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
- `OpenTAD/tools/prepare_data/sails_rmm/README.md` documents convert `--fold`/`--task`
  (`binary` \| `4class`) and the `4class` ↔ SLURM `balanced` name mismatch.
- Filesystem paths are centralized in [`config.yaml`](config.yaml) + [`paths.py`](paths.py)
  (the earlier hardcoded-`/orcd/scratch` paths were migrated to the lab project space).
- **pyskl eval env has drifted to an incompatible PyTorch.** `envs/pyskl.yml` pins
  `torch==2.9.0` with `mmcv-full==1.7.0`; the mmcv-1.7 `MMDistributedDataParallel` wrapper
  reads a Torch-1.x internal (`_use_replicated_tensor_module`) that Torch 2.x removed, so
  `pyskl/tools/test.py` (which always runs through `init_dist`) aborts *after* the model +
  pose pickle load and the forward pass starts:
  `AttributeError: 'MMDistributedDataParallel' object has no attribute '_use_replicated_tensor_module'`.
  `--launcher none` is also rejected (`Invalid launcher type: none`). Path resolution,
  checkpoint loading, and data loading are verified from a fresh clone; only the DDP eval
  wrapper is affected. **Fix for the next maintainer:** recreate the `pyskl` env with a
  Torch/mmcv pair that mmcv-full 1.7.0 supports (Torch ≤ ~1.13), or upgrade mmcv, before
  re-running PoseC3D/STGCN++ eval. The reported classification numbers were generated
  before this Torch upgrade.
- **`pyskl/tools/test.py` indentation fix.** Two `dist.barrier()` calls under `if distributed:`
  were unindented (committed syntax error); fixed on `clean-repo`. This `test.py` also does
  **not** accept `--cfg-options`.
- **PoseC3D CV configs test a split the pickle does not have.** Fold pickles
  (`pyskl/data/sails/cv/4class_conf04/fold0.pkl`) expose `train` and `val` only (val = the
  held-out fold). The dumped `work_dirs/.../joint.py` sets `data.test.split = 'test'`.
  Point test at `val` (or add a `test` key to the pickle) before eval. Classification
  pickles are tracked in git (~15 MB each) and in `checkpoints_root`. TAL window pickles
  (`pyskl/data/sails/tal/`, ~149 MB each) cannot go on GitHub (100 MB file limit) — get
  them from the overlay only.
- There was no `OpenTAD/slurm/test_actionformer_cv.sh` until `clean-repo`; only
  `test_tridet_cv.sh` existed, which is why ActionFormer eval is easy to miss. Use the
  ActionFormer wrapper (same args as TriDet: `<task> <fold>`).
