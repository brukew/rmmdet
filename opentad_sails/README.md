# OpenTAD SAILS-RMM changes

The `OpenTAD/` directory is a fork of [sming256/OpenTAD](https://github.com/sming256/OpenTAD).
All SAILS-RMM–specific work (dataset configs, ActionFormer/TriDet configs, data-prep
scripts, SLURM scripts, and a handful of model-code edits) lives **on top of** a pinned
upstream commit and is captured here as a single patch so it survives a fresh clone /
submodule checkout.

## Contents

| File | Purpose |
| :--- | :--- |
| `UPSTREAM_COMMIT.txt` | The exact upstream OpenTAD commit the patch applies on top of |
| `sails_changes.patch` | All SAILS changes vs. that commit (modified + new + deleted files) |

The patch covers the SAILS configs (`configs/_base_/datasets/sails_rmm/`,
`configs/actionformer/sails_rmm_*`, `configs/tridet/sails_rmm_*`), model-code edits
(`opentad/models/...`, including a new `align1d/__init__.py` and a two-line
`Align1D_cuda_kernal.cu` fix — `x.type()` → `x.scalar_type()` in the
`AT_DISPATCH_FLOATING_TYPES` calls — so the ROI-align op compiles against the pinned
PyTorch in `envs/opentad.yml`), the SLURM scripts (`slurm/`), the data-prep files
(`tools/prepare_data/sails_rmm/`), and `tools/test.py`.

> Artifacts (`exps/`, `logs/`, `*.out`, `*.err`, `wandb/`) are intentionally **not** in
> the patch — they are git-ignored by OpenTAD and are regenerable. See the repo-level
> artifact policy in `../docs/ARTIFACTS.md`.

## Reconstructing the SAILS fork from a clean checkout

If `OpenTAD/` is set up as a git submodule pinned to the upstream commit:

```bash
cd actreg
git submodule update --init OpenTAD          # checks out the pinned upstream commit
cd OpenTAD
git apply --whitespace=nowarn ../opentad_sails/sails_changes.patch
```

Or, cloning OpenTAD standalone:

```bash
git clone git@github.com:sming256/OpenTAD.git
cd OpenTAD
git checkout "$(head -1 ../opentad_sails/UPSTREAM_COMMIT.txt)"
git apply --whitespace=nowarn ../opentad_sails/sails_changes.patch
```

After applying, build the `align1d` ROI-align CUDA op (needed by **TriDet**; ActionFormer
is anchor-free and does not use it). The patched `.cu` compiles against the pinned
PyTorch, so a from-source build works on a GPU node:

```bash
cd OpenTAD/opentad/models/roi_heads/roi_extractors/align1d
python setup.py build_ext --inplace     # produces Align1D.*.so (run on a GPU node)
```

Then proceed with the data-prep step in `OpenTAD/tools/prepare_data/sails_rmm/README.md`
(run `convert_cv_splits_to_opentad_json.py --fold <f> --task <binary|balanced>` to
generate the annotation JSONs before `tools/test.py`).

## Regenerating this patch (when OpenTAD changes again)

```bash
cd OpenTAD
git add -A                                   # gitignore keeps exps/ etc. out
git diff --cached --binary HEAD > ../opentad_sails/sails_changes.patch
git reset -q                                 # unstage; working tree untouched
git rev-parse HEAD > ../opentad_sails/UPSTREAM_COMMIT.txt
```

> NOTE: this assumes `HEAD` is still the pinned upstream commit. If you have rebased the
> fork onto a newer upstream, update `UPSTREAM_COMMIT.txt` accordingly.
