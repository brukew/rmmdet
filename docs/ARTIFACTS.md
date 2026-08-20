# Artifacts: where the data and checkpoints live

Reference for everything that is **not** in git: the input data and the trained
checkpoints. The repo working tree is ~168 GB; the **tracked source** in git (code,
configs, splits, annotations, metrics JSON, docs, figures) is only **~57 MB**. Everything
else is described here. Sizes verified Aug 2026.

A fresh reproduction needs three things:

1. **The repo** (GitHub) — code, splits, metrics, tables, and the OpenTAD patch.
2. **The input data** — in the durable lab project space (below).
3. **The trained checkpoints** — git-ignored; a durable copy lives in the lab project space
   (below). Only needed to reproduce results *without retraining*.

---

## 1. Input data — durable lab project space

All inputs resolve under `/orcd/data/satra/002/projects/SAILS/`, addressed via
[`config.yaml`](../config.yaml)/[`paths.py`](../paths.py). This tree is **self-contained**
(no symlinks into scratch — verified; see §4).

| Path (`config.yaml` key) | Contents | Size |
| :--- | :--- | ---: |
| `rmm/features` (`rmm_features`) | V-JEPA2 TAL features (OpenTAD input) | 933 MB |
| `rmm/classification_clips` (`classification_clips`) | source classification clips (+ per-fold split symlinks into local `canonical_clips/`) | 4.0 GB |
| `rmm/tal_windows_4class` (`tal_clips_root`) | TAL window clips + splits | 37 GB |
| `rmm/vjepa2_finetune_clips` (`vjepa2_finetune_clips`) | V-JEPA finetune clips | — |
| `cache_for_tracking` (`cache_for_tracking`) | SAM3 pose H5 caches (live-crop + pose pickles) | — |
| `feature_processing/pipeline_outputs` (`pipeline_outputs`) | intermediate pipeline outputs | — |
| `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos` (`dataset_root`) | raw source videos (window CSVs' `video_path`) | — |

Two `config.yaml` paths are **absent everywhere and non-blocking**:
`classification_clips_cropped` (output-only crop target; nothing reads it) and `rmm/videos`
(a dead `--videos-root` fallback). See the data-availability table in `REPRODUCE.md`.

The originals still exist on **shared scratch** (`/orcd/scratch/bcs/001/sensein/sails`,
~2.0 TB; `/orcd/scratch/bcs/001/brukew/sails`, 36 GB) but scratch is **scheduled for
cleanup** — the lab copy above is the durable source of truth.

---

## 2. Trained checkpoints — git-ignored, with a durable shared copy

**None of the checkpoints are in git or on GitHub** (see §3). In this working tree they sit,
git-ignored, under the repo output roots:

```
OpenTAD/exps/sails_rmm/.../gpu1_id0/checkpoint/best.pth     # + gpu1_id99/result_detection.json
v-jepa/runs/vjepa2_rmm_cv/... , v-jepa/runs/vjepa2_tal_cv_*/...
pyskl/work_dirs/{posec3d,stgcnpp}/...
fusion/runs/...
```

A **~24 GB lean copy of just the reported-result finals** is mirrored to the durable lab
project space at `checkpoints_root` — using the **same repo-relative layout** inside:

```
/orcd/data/satra/002/projects/SAILS/checkpoints/
  OpenTAD/exps/sails_rmm/.../best.pth        (12) + result_detection.json (16)
  v-jepa/runs/...                            (14 model.safetensors across 5 reported runs)
  pyskl/work_dirs/...                        (157 best_*.pth)
  fusion/runs/...                            (fusion MLPs)
```

**To reproduce from a fresh clone**, overlay the copy into the working tree (drops each file
at the exact path the scripts expect):

```bash
rsync -a "$(python paths.py --get checkpoints_root)"/ .
```

This is weights, OpenTAD annotation JSONs, pyskl pose pickles, and
`result_detection.json`. Overlay into the working tree:

```bash
rsync -a "$(python paths.py --get checkpoints_root)"/ .
```

| Extra eval input (not a trained weight) | In git? | In overlay? |
| :--- | :--- | :--- |
| `OpenTAD/data/sails_rmm/annotations/*.json` | yes (SAILS patch) | yes |
| `pyskl/data/sails/{cv,single,*.pkl}` classification pickles | yes (~15 MB each) | yes |
| `pyskl/data/sails/tal/**/*.pkl` TAL window pickles | **no** (~149 MB each; GitHub file limit is 100 MB) | yes |


### What the shared copy contains (the reported-result finals)

| Reported model(s) | Path prefix | Files | Size |
| :--- | :--- | ---: | ---: |
| OpenTAD ActionFormer + TriDet (binary+balanced × 3 folds) | `OpenTAD/exps/sails_rmm/` | 12 `best.pth` + 16 `result_detection.json` | 1.9 GB |
| V-JEPA2 (cls 4/5-cls; TAL 5cls-balanced, 5cls-bgsub, binary) | `v-jepa/runs/` | 14 `model.safetensors` + clip-level CSVs | 19.6 GB |
| PoseC3D + STGCN++ 4-stream (cls + TAL) | `pyskl/work_dirs/` | 157 `best_*.pth` + `predictions_clip.csv` | 2.2 GB |
| Late-fusion MLPs (2-way / 3-way) | `fusion/runs/` | — | 0.01 GB |

Prediction CSVs (`predictions_clip.csv`, `clip_level_preds.csv`) are gitignored but **are**
in the overlay so fusion can run without re-eval.

> `vjepa2_tal_cv_5class_bgsub` has 2 folds (fold_2 was not saved) — reflected in the tables.
> The full training dumps (all epochs / experimental runs) are **not** copied; see §5.

---

## 3. What is git-ignored (and therefore not on GitHub)

| Path | Ignored by |
| :--- | :--- |
| `v-jepa/runs/` | `.gitignore` (`runs/`) |
| `pyskl/work_dirs/` | `pyskl/.gitignore` (`work_dirs/`) |
| `OpenTAD/exps/` | `OpenTAD/.gitignore` (`/exps/`, inside the submodule) |

`OpenTAD` is a **submodule** pinned to upstream `sming256/OpenTAD@1aa8ca4`; the SAILS delta
is `opentad_sails/sails_changes.patch`, not a pushed fork. So on GitHub the `OpenTAD` folder
links to upstream and contains **no** `exps/`. The small tracked weights that *are* in git
(pretrained inits) are the NTU60 STGCN++ files under `pyskl/checkpoints/` (allow-listed in
`.gitignore`).

---

## 4. Regenerable bulk and disposable

Everything below is derivable from code + the §1 inputs + the §2 finals, and is **not**
copied to the shared space:

| What | Where | Size |
| :--- | :--- | ---: |
| Experimental / per-epoch checkpoints, optimizer states, logs | `v-jepa/runs/`, `pyskl/work_dirs/`, `OpenTAD/exps/` | 83 / 43 / 40 GB |
| Rendered demo videos | `rendered_videos/*.mp4` | 890 MB |
| Caches, smoke dirs | `__pycache__/`, ad-hoc smoke output | small |

Of the 40 GB `OpenTAD/exps/`, only the 1.9 GB of `best.pth` + detection JSON is a final; of
the 83 GB `v-jepa/runs/`, only the 19.6 GB reported subset is a final.

**Verification (Aug 2026).** The lab data tree is scratch-independent: `find
/orcd/data/satra/002/projects/SAILS/rmm -type l -lname '/orcd/scratch/*' | wc -l` → 0, and 0
broken links. The shared checkpoint copy was integrity-checked: file-count parity with the
source, matching md5s, and every `model.safetensors` / OpenTAD `best.pth` (and sampled pyskl
`best_*.pth`) loads.

> All reported mAP / accuracy numbers can also be re-derived from the tracked metrics alone
> (`metrics.json`, `cv_summary.json`, `insights/tables/*.json`) without any checkpoint, so
> analysis/paper work needs only the repo.
