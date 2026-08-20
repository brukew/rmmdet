#!/bin/bash -l
# Late Fusion Training: V-JEPA2 + PoseC3D with per-class α vector
#
# Trains a per-class fusion weight (4 parameters for 4-class task).
# Each class can have different modality preference.
#
# Usage:
#   sbatch run_fusion_perclass_cv.sh
#
# Logs: ${REPO_ROOT}/fusion_logs

#SBATCH -J fusion_perclass
#SBATCH -p pi_satra
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o slurm-logs/fusion_perclass_cv_%j.out
#SBATCH -e slurm-logs/fusion_perclass_cv_%j.err

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

if [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi

cd ${REPO_ROOT}

# Use vjepa2 env (has torch, numpy, pandas, sklearn, matplotlib)
conda activate vjepa2

LOG_DIR=${REPO_ROOT}/fusion_logs
mkdir -p "$LOG_DIR"

# ============================================================================
# Paths
# ============================================================================
VJEPA_ROOT=${VJEPA_ROOT:-${REPO_ROOT}/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls}
# Use non-weighted PoseC3D which has full val set coverage
POSEC3D_ROOT=${POSEC3D_ROOT:-${REPO_ROOT}/pyskl/work_dirs/posec3d/cv/4class_conf04}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/fusion/runs/4class_cv_perclass}

# ============================================================================
# Training config
# ============================================================================
NUM_CLASSES=${NUM_CLASSES:-4}
NUM_FOLDS=${NUM_FOLDS:-3}
NUM_EPOCHS=${NUM_EPOCHS:-100}
LR=${LR:-0.01}
BATCH_SIZE=${BATCH_SIZE:-32}
FUSION_TYPE="per_class"

echo "============================================================"
echo "Late Fusion Training: V-JEPA2 + PoseC3D (Per-Class α)"
echo "============================================================"
echo "Fusion type: $FUSION_TYPE (4 parameters)"
echo "V-JEPA2 predictions: $VJEPA_ROOT"
echo "PoseC3D predictions: $POSEC3D_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Config: classes=$NUM_CLASSES, folds=$NUM_FOLDS, epochs=$NUM_EPOCHS, lr=$LR"
echo "============================================================"

python fusion/train_fusion_cv.py \
  --vjepa-root "$VJEPA_ROOT" \
  --posec3d-root "$POSEC3D_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --fusion-type "$FUSION_TYPE" \
  --num-classes "$NUM_CLASSES" \
  --num-folds "$NUM_FOLDS" \
  --num-epochs "$NUM_EPOCHS" \
  --lr "$LR" \
  --batch-size "$BATCH_SIZE" \
  --log-level INFO

echo "============================================================"
echo "Done! Results saved to: $OUTPUT_DIR"
echo "============================================================"

