# Artifact & Data Retention Policy

The repo working tree is ~168 GB, but the **tracked source** (code, configs, splits,
annotations, metrics, docs, figures) is only **~57 MB** (`.git` ~72 MB). The rest is model
output and input data that must **not** live in git. This document records where everything
is and what is safe to delete vs. must be kept. Sizes verified Aug 2026.

## Where the data lives (durable vs. scratch)

Shared SAILS data has been **mirrored to the lab's durable project space**, and
[`config.yaml`](../config.yaml) resolves every input path there:

```
/orcd/data/satra/002/projects/SAILS/          # durable lab project space (config source of truth)
  rmm/features/                 # V-JEPA2 TAL features       (~933 MB)
  rmm/classification_clips/     # source classification clips (~4.0 GB)
  rmm/tal_windows_4class/       # TAL window clips + splits   (~37 GB)
  rmm/vjepa2_finetune_clips/    # V-JEPA finetune clips
  rmm/{rmm_sam_numbered,rmm_numbered_target}/
  cache_for_tracking/           # SAM3 pose H5 caches (consumed by crop + pose-pickle steps)
  feature_processing/pipeline_outputs/
```

The original **scratch** copies still exist but are **shared and impermanent** (subject to
cluster cleanup) — treat them as disposable now that the durable mirror exists:

```
/orcd/scratch/bcs/001/sensein/sails/    # 2.0 TB shared lab scratch (many users; original inputs)
/orcd/scratch/bcs/001/brukew/sails/     # 36 GB — tal_windows_4class (duplicate of the lab copy)
```

> Two `config.yaml` paths are **absent everywhere** and non-blocking:
> `classification_clips_cropped` (output-only crop target; nothing reads it) and `rmm/videos`
> (a dead `--videos-root` fallback). See the data-availability table in `REPRODUCE.md`.

## Tiering

| Tier | What | Keep? | Where |
| :--- | :--- | :--- | :--- |
| **A. Source** | code, configs, splits, annotations, metrics (`eval_results/`, `cv_summary.json`), docs, figures | **In git** (~57 MB) | this repo |
| **B. Precious inputs + reported checkpoints** | TAL features (933 MB), classification clips (4.0 GB), TAL window clips (37 GB), SAM3 pose caches; the 12 OpenTAD `best.pth` (1.9 GB) + the reported V-JEPA/pyskl/fusion finals | **Keep, git-ignored** | lab project space (inputs) + repo `exps/`,`runs/`,`work_dirs/` (finals) |
| **C. Regenerable bulk** | full training dumps: all non-final per-run checkpoints, optimizer states, prediction pickles, logs | **Safe to archive/delete** | `v-jepa/runs/` (83 GB), `pyskl/work_dirs/` (43 GB), `OpenTAD/exps/` (40 GB, of which only 1.9 GB is finals) |
| **D. Disposable** | `__pycache__/`, `rendered_videos/*.mp4` (890 MB), smoke-test dirs, local backups | **Delete** | various |

> **Note on `v-jepa/runs/` (83 GB, 59 `.safetensors`):** this is dominated by *experimental*
> runs. Only the reported classifier is precious — the 4-class CV model at
> `v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls/fold_{0,1,2}` (see
> `paths.py --get vjepa_4class_ckpt`). The rest is Tier C.

## Reported-result checkpoints (for the no-retrain path)

```
OpenTAD/exps/sails_rmm/{actionformer,tridet}_vjepa_{binary,balanced}_fold{0,1,2}/gpu1_id*/checkpoint/best.pth
v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls/fold_{0,1,2}/     # V-JEPA classifier finals
pyskl/work_dirs/.../                                                            # PoseC3D / STGCN++ finals
```

## Recommended actions for handoff

1. **Tier B backup — DONE.** Inputs are mirrored to the durable lab project space above and
   `config.yaml` points there; the scratch copies are now redundant. Keep the durable copy;
   scratch may be cleaned at any time.
2. **Archive or delete Tier C.** Everything in `v-jepa/runs/` / `pyskl/work_dirs/` /
   `OpenTAD/exps/` other than the finals above (and the `metrics.json` / `cv_summary.json`
   already in git) is regenerable from code + Tier-B inputs. Reclaims ~160 GB.
3. **Delete Tier D** outright.
4. Everything in Tier A is tracked in git under the scoped `.gitignore`; the artifact
   directories above are git-ignored so they can never be committed by accident.

> All reported mAP / accuracy numbers can be re-derived from Tier A (`metrics.json`,
> `cv_summary.json`, `insights/tables/*.json`) without touching Tier B/C, so analysis/paper
> work needs only the repo.
