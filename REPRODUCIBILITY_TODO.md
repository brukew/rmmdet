y# Reproducibility TODO

## 1. Git Hygiene

- [ ] Fix broken OpenTAD submodule (remove from git index, add `.gitmodules` or track as regular directory)
- [ ] Fix `.gitignore` to stop ignoring all `.json` and `.csv` — scope exclusions to large data files and specific directories only
- [ ] Commit currently untracked files (`updates_1/2/3.md`, `RESOURCES.md`, `tal/tools/`, `scripts/extract_precision_recall.py`, `insights/` subdirs, `writeup/`)

## 2. Path Configuration

- [ ] Create a central `config.yaml` (or `paths.py`) for all dataset and output paths, replacing hardcoded `/orcd/` paths
- [ ] Update V-JEPA2 scripts to read paths from config instead of hardcoded defaults
- [ ] Update pyskl SLURM scripts to read paths from config instead of hardcoded defaults
- [ ] Update fusion training scripts to read paths from config instead of hardcoded defaults
- [ ] Update TAL scripts (`dataprep/tal/`, `tal/`) to read paths from config instead of hardcoded defaults
- [ ] Update clip generation and pose estimation scripts (`dataprep/`) to read paths from config

## 3. Environment Files

- [ ] Create `environment.yml` for `pyskl` (PoseC3D + STGCN++) with pinned versions
- [ ] Create `environment.yml` for `vjepa2` with pinned versions
- [ ] Create `environment.yml` for `OpenTAD` with pinned versions

## 4. Documentation

- [ ] Write `REPRODUCE.md` with full step-by-step pipeline: data prep → pose → splits → train each model → eval → fusion → TAL
