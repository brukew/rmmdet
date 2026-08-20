#!/bin/bash -l
# 5-Class Late Fusion: V-JEPA2 + STGCN++ (4-stream)
#
# Model configs:
#   - V-JEPA2: f64_lr1e-5_bs1_acc8_ep20_crop (5-class with SAM3 cropping)
#   - STGCN++: 5class_conf04_4stream (non-weighted, has prediction CSVs)
#
# Note: Using non-weighted STGCN++ because weighted 4-stream doesn't have prediction CSVs.
#       Non-weighted 4-stream: 66.5% Clip Top-1, 68.8% Video Top-1
#
# Prerequisites:
#   - V-JEPA2 5-class must have score_classX columns (run rerun_vjepa_eval_5class.sh first)

#SBATCH -J fusion_5cls_stgcn
#SBATCH -p pi_satra
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o slurm-logs/fusion_5class_stgcn_cv_%j.out
#SBATCH -e slurm-logs/fusion_5class_stgcn_cv_%j.err

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

# V-JEPA2 5-class with SAM3 cropping
VJEPA_ROOT=${REPO_ROOT}/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop

# STGCN++ 5-class 4-stream (non-weighted, has prediction CSVs)
STGCN_ROOT=${REPO_ROOT}/pyskl/work_dirs/stgcnpp/cv/5class_conf04_4stream

OUTPUT_DIR=${REPO_ROOT}/fusion/runs/5class_cv_stgcn_mlp

NUM_CLASSES=5
NUM_FOLDS=3
NUM_EPOCHS=100
LR=0.01
BATCH_SIZE=32

echo "============================================================"
echo "5-Class Late Fusion: V-JEPA2 + STGCN++ (MLP)"
echo "============================================================"
echo "Model configs:"
echo "  V-JEPA2: f64_lr1e-5_bs1_acc8_ep20_crop (5-class SAM3 crop)"
echo "  STGCN++: 5class_conf04_4stream (66.5% Clip, 68.8% Video Top-1)"
echo ""
echo "V-JEPA2 predictions: $VJEPA_ROOT"
echo "STGCN++ predictions: $STGCN_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Config: classes=$NUM_CLASSES, folds=$NUM_FOLDS, epochs=$NUM_EPOCHS, lr=$LR"
echo "============================================================"

python fusion/train_fusion_cv.py \
  --vjepa-root "$VJEPA_ROOT" \
  --skeleton-root "$STGCN_ROOT" \
  --skeleton-model-name "STGCN++-4stream" \
  --output-dir "$OUTPUT_DIR" \
  --num-classes "$NUM_CLASSES" \
  --num-folds "$NUM_FOLDS" \
  --num-epochs "$NUM_EPOCHS" \
  --lr "$LR" \
  --batch-size "$BATCH_SIZE" \
  --log-level INFO \
  --fusion-type mlp

echo "============================================================"
echo "Done! Results saved to: $OUTPUT_DIR"
echo "============================================================"


