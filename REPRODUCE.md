# Reproducing the actreg pipeline

End-to-end map of the project, in dependency order. Each stage links to the component
README that has the detailed commands. Data/artifact locations are documented in
[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md); large paths live under `/orcd/scratch/...`.

> Three conda envs are used: **`pyskl`** (PoseC3D/STGCN++), **`vjepa2`** (V-JEPA2 +
> two-stage), **`opentad`** (OpenTAD TAL eval). All three are pinned in [`envs/`](envs/).

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

Clips and features live in scratch (see `docs/ARTIFACTS.md`).

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

Full detail: [`tal/README.md`](tal/README.md).

---

## 4. TAL — OpenTAD E2E + binary detectors (`OpenTAD/`)

ActionFormer / TriDet trained on V-JEPA2 features, both **balanced** (E2E multiclass) and
**binary** (single "action" class, used as Stage-1 of the two-stage pipeline).

```bash
cd OpenTAD
# splits -> OpenTAD annotation JSON
python tools/prepare_data/sails_rmm/convert_cv_splits_to_opentad_json.py
# train (per fold)
sbatch slurm/train_actionformer_cv.sh
sbatch slurm/train_tridet_cv.sh
# test -> result_detection.json (best.pth per fold)
sbatch slurm/test_tridet_cv.sh
```

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

---

## Known gaps for the next maintainer

- `OpenTAD/tools/prepare_data/sails_rmm/README.md` is currently empty — the convert step
  is captured in the patch but undocumented; see the script itself.
- Paths are still hardcoded across scripts (see `REPRODUCIBILITY_TODO.md` item 2; a central
  `config.yaml` is the recommended fix).
