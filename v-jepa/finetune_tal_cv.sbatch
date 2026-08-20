#!/bin/bash
#SBATCH --job-name=vjepa2_tal_cv
#SBATCH --output=slurm-logs/vjepa2_tal_cv_%j.out
#SBATCH --error=slurm-logs/vjepa2_tal_cv_%j.err
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ============================================================================
# V-JEPA2 TAL Fine-tuning - Cross-Validation (5 classes: 4 RMM + background)
#
# Submit from the repo root:  sbatch v-jepa/finetune_tal_cv.sbatch
# ============================================================================

# --- Portable repo-root resolution (auto-inserted) --------------------------
# Locate the repo root (the directory containing paths.py) so this script runs
# from any clone name/location, under `bash` or `sbatch`. SLURM copies the
# script to a spool dir, so if the script path does not resolve we fall back to
# $SLURM_SUBMIT_DIR then $PWD. Override by exporting REPO_ROOT before launch.
_rmm_find_root() {
  local d="$1"
  while [ -n "$d" ] && [ "$d" != "/" ]; do
    if [ -f "$d/paths.py" ]; then printf '%s\n' "$d"; return 0; fi
    d="$(dirname "$d")"
  done
  return 1
}
if [ -z "${REPO_ROOT:-}" ] || [ ! -f "${REPO_ROOT:-x}/paths.py" ]; then
  REPO_ROOT="$(_rmm_find_root "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" 2>/dev/null && pwd)")" \
    || REPO_ROOT="$(_rmm_find_root "${SLURM_SUBMIT_DIR:-$PWD}")" \
    || REPO_ROOT="$(_rmm_find_root "$PWD")" || true
fi
if [ -z "${REPO_ROOT:-}" ] || [ ! -f "${REPO_ROOT}/paths.py" ]; then
  echo "ERROR: cannot locate repo root (paths.py). cd to the repo or export REPO_ROOT." >&2
  exit 1
fi
export REPO_ROOT
# --- end repo-root resolution ----------------------------------------------

set -eo pipefail

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

# Ensure real-time logging
export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi

cd "${REPO_ROOT}"

# Use vjepa2 env (has torch, numpy, pandas, sklearn, matplotlib)
conda activate vjepa2

# Pull TAL_CLIPS_ROOT (and friends) from config.yaml via paths.py
eval "$(python "${REPO_ROOT}/paths.py" --export)"

# Run V-JEPA2 TAL fine-tuning with background subsampling
python v-jepa/finetune_sails_vjepa2_tal.py \
    --split-mode cv \
    --csv-dir "${REPO_ROOT}/dataprep/tal/splits_cv_4class" \
    --clips-root "${TAL_CLIPS_ROOT:?TAL_CLIPS_ROOT not set by paths.py --export}" \
    --output-root "${REPO_ROOT}/v-jepa/runs/vjepa2_tal_cv_5class_bgsub" \
    --num-epochs 20 \
    --batch-size 1 \
    --accumulation-steps 4 \
    --lr 1e-5 \
    --num-workers 8 \
    --frames-per-clip 32 \
    --topk 2 \
    --bg-subsample 2.0 \
    --enable-crop \
    --wandb-project vjepa-tal \
    --run-prefix vjepa2-tal-cv-bgsub \
    --log-level INFO

echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="

