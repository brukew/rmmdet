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

## 2. Path Configuration — SCRATCH MIGRATION DONE; repo-path migration optional

- [x] Central `config.yaml` + `paths.py` loader (auto-detects repo root, env-overridable;
      `--export`/`--get` modes for shell scripts).
- [x] **Shared data migrated off soon-to-be-purged scratch → lab durable space**
      (`/orcd/data/satra/002/projects/SAILS`). `config.yaml` repointed; verified 1:1.
  - [x] `tal_windows_4class` (36G) copied to lab + its 14,145 symlinks rewritten relative
        (were absolute into scratch — would have dangled at purge).
  - [x] `classification_clips_cropped`: absent everywhere; regenerable + not a pipeline
        input (crop training crops on-the-fly from `cache_for_tracking`).
  - [x] `rmm/videos`: source already purged; features pre-extracted.
- [x] **All 30 files that referenced `/orcd/scratch/` migrated to read from config**
      (verified: py_compile + bash -n clean, 0 remaining scratch refs):
  - [x] 22 `.py` (v-jepa finetune/extract, two-stg, dataprep, pyskl, scripts, insights)
  - [x] 8 `v-jepa/slurm/*` (source `paths.py --export`)
  - [x] 6 `OpenTAD/configs/_base_/datasets/sails_rmm/*` feature paths (+ patch regenerated)
- [ ] OPTIONAL: migrate the remaining *repo-path* literals (`/orcd/data/satra/001/...`,
      ~250 refs in non-scratch entry points like `two-stg/run_*`, `fusion/`, `tal/`).
      Lower priority — that's durable data space, unaffected by the purge.

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
