# Artifact & Data Retention Policy

The repo working tree is ~168 GB, but **code + tracked text data is only ~72 MB**.
The remaining ~168 GB is model output and input data that should **not** live in git.
This document records where everything is and what is safe to delete vs. must be kept.

## Tiering

| Tier | What | Keep? | Where |
| :--- | :--- | :--- | :--- |
| **A. Source** | code, configs, splits, annotations, metrics, docs, figures | **In git** (~72 MB) | this repo |
| **B. Precious inputs** | V-JEPA TAL features (~1.3 GB), classification clips (~4 GB), final per-fold `best.pth` checkpoints (~2 GB in OpenTAD, plus v-jepa/pyskl finals) | **Keep on disk, git-ignored. Back up.** | `/orcd/scratch/...` + repo `exps/`/`runs/`/`work_dirs/` |
| **C. Regenerable bulk** | full training dumps: per-epoch checkpoints, optimizer states, raw prediction pickles, logs | **Safe to archive/delete** | `runs/` (83 GB), `work_dirs/` (43 GB), `exps/` (~38 GB after finals), `proposal_pickles*/` (1.1 GB) |
| **D. Disposable** | `__pycache__/`, top-level `wandb/`, smoke-test dirs, `rendered_videos/*.mp4` (890 MB) | **Delete** | various |

## Key locations (for the next maintainer)

```
# Precious INPUTS (hard to regenerate -- require GPU feature extraction / clip gen)
/orcd/scratch/bcs/001/sensein/sails/rmm/features/vjepa2_tal_cv_*   # TAL features, ~110 MB/fold
/orcd/scratch/bcs/001/sensein/sails/rmm/classification_clips/      # source clips, ~4 GB

# Final reported-result checkpoints (a small slice of the 166 GB of outputs)
OpenTAD/exps/sails_rmm/<model>_fold{0,1,2}/gpu1_id0/checkpoint/best.pth   # 110-212 MB each
v-jepa/runs/.../fold_{0,1,2}/                                            # V-JEPA classifier finals
pyskl/work_dirs/.../                                                      # PoseC3D / STGCN++ finals
```

## Recommended actions for handoff

1. **Back up Tier B** (features + clips + final `best.pth` per fold) to durable storage
   outside scratch. Scratch is not guaranteed permanent. This is the single most
   important data-preservation step.
2. **Archive or delete Tier C.** Everything in `runs/`/`work_dirs/`/`exps/` other than the
   final `best.pth` (and the `metrics.json` already copied into git) is regenerable from
   code + Tier-B inputs. Reclaims ~160 GB.
3. **Delete Tier D** outright (see `docs/CLEANUP.md` if present).
4. Everything in Tier A is already (or will be) tracked in git under the corrected
   `.gitignore`; the artifact directories above are git-ignored so they can never be
   committed by accident.

> All reported mAP / accuracy numbers can be re-derived from Tier A (`metrics.json`,
> `cv_summary.json`) without touching Tier B/C, so analysis/paper work needs only the repo.
