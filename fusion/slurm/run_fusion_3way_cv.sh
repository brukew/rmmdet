#!/bin/bash -l
#SBATCH -J fusion_3way_cv
#SBATCH -p pi_satra
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o slurm-logs/fusion_3way_cv_%j.out
#SBATCH -e slurm-logs/fusion_3way_cv_%j.err

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
source ~/.bashrc
cd ${REPO_ROOT}
conda activate vjepa2

LOG_DIR=${REPO_ROOT}/fusion_logs
mkdir -p "$LOG_DIR"

# Best V-JEPA2 model (with SAM3 cropping)
VJEPA_ROOT=${REPO_ROOT}/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls

# Best PoseC3D model (non-weighted) - 74.6% Video Top-1
POSEC3D_ROOT=${REPO_ROOT}/pyskl/work_dirs/posec3d/cv/4class_conf04

# Best STGCN++ model (4-stream, non-weighted) - 82.4% Video Top-1
STGCN_ROOT=${REPO_ROOT}/pyskl/work_dirs/stgcnpp/cv/4class_conf04_4stream

OUTPUT_DIR=${REPO_ROOT}/fusion/runs/4class_cv_3way

NUM_CLASSES=4
NUM_FOLDS=3
NUM_EPOCHS=100
LR=0.01
BATCH_SIZE=32
MLP_HIDDEN_DIM=24

echo "============================================================"
echo "3-Way Late Fusion Training: V-JEPA2 + PoseC3D + STGCN++"
echo "============================================================"
echo "V-JEPA2 predictions: $VJEPA_ROOT"
echo "PoseC3D predictions: $POSEC3D_ROOT"
echo "STGCN++ predictions: $STGCN_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Config: classes=$NUM_CLASSES, folds=$NUM_FOLDS, epochs=$NUM_EPOCHS, lr=$LR"
echo "MLP hidden dim: $MLP_HIDDEN_DIM (3 inputs x $NUM_CLASSES = $(($NUM_CLASSES * 3)))"
echo "============================================================"

python fusion/train_fusion_cv.py \
  --vjepa-root "$VJEPA_ROOT" \
  --posec3d-root "$POSEC3D_ROOT" \
  --stgcn-root "$STGCN_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --num-classes "$NUM_CLASSES" \
  --num-folds "$NUM_FOLDS" \
  --num-epochs "$NUM_EPOCHS" \
  --lr "$LR" \
  --batch-size "$BATCH_SIZE" \
  --mlp-hidden-dim "$MLP_HIDDEN_DIM" \
  --log-level INFO \
  --fusion-type three_way

echo "============================================================"
echo "Done! Results saved to: $OUTPUT_DIR"
echo "============================================================"



