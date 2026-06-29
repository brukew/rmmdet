# Reproducibility / Handoff TODO

Status as of the handoff cleanup. See `docs/INDEX.md` for the full doc catalog and
`REPRODUCE.md` for the end-to-end pipeline.

## 1. Git Hygiene — DONE

- [x] OpenTAD: registered as a git submodule (`.gitmodules`) pinned to upstream
      `1aa8ca4`; SAILS fork delta captured as a verified patch in `opentad_sails/`.
- [x] `.gitignore`: scoped to artifacts (no longer ignores all `.json`/`.csv`/`.txt`);
      splits/annotations/metrics are tracked, the ~166 GB of run artifacts are not.
- [x] Committed the previously-untracked work (two-stg, tal, dataprep, insights,
      fusion export, docs).

## 2. Path Configuration — PARTIAL

- [x] Central `config.yaml` + `paths.py` loader (auto-detects repo root, env-overridable).
- [ ] Migrate scripts to read from `paths.py` instead of hardcoded `/orcd/...` paths
      (~280 references). **Untested-in-this-env risk** — do this where the running
      environment is available, starting with the top-level entry-point scripts:
  - [ ] `two-stg/run_*` SLURM wrappers + `eval_two_stage_tal.py`
  - [ ] `v-jepa/slurm/*` + `finetune_sails_vjepa2_*.py`
  - [ ] `fusion/` training/export scripts
  - [ ] `tal/` and `dataprep/` scripts
  - [ ] `OpenTAD/configs/_base_/datasets/sails_rmm/*` feature paths

## 3. Environment Files — DONE

- [x] `envs/pyskl.yml`
- [x] `envs/vjepa2.yml`
- [x] `envs/opentad.yml`

## 4. Documentation — DONE

- [x] `REPRODUCE.md`: full pipeline (data prep → pose → splits → train → eval → fusion → TAL).
- [x] `docs/ARTIFACTS.md`: data/artifact retention policy.
- [x] `docs/INDEX.md`: catalog of all docs + entry points.
- [x] README updated to cover the TAL / two-stage work.
- [x] Consolidated scattered docs under `docs/` (writeup, retraining notes, archived updates).

## 5. Data preservation (action for the maintainer)

- [ ] Back up Tier-B inputs (V-JEPA TAL features + clips + final `best.pth` per fold)
      off scratch — see `docs/ARTIFACTS.md`. This is the key data-loss risk.
