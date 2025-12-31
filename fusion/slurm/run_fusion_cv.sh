#!/bin/bash -l
# Late Fusion Training: V-JEPA2 + PoseC3D with learned α
#
# Trains a scalar fusion weight on pre-computed predictions.
# Very fast since only 1 parameter is trained.
#
# Usage:
#   sbatch run_fusion_cv.sh
#
# Logs: /orcd/data/satra/001/users/brukew/fusion_logs

#SBATCH -J fusion_cv
#SBATCH -p mit_preemptable
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH --requeue
#SBATCH -o /orcd/data/satra/001/users/brukew/fusion_logs/fusion_cv_%j.out
#SBATCH -e /orcd/data/satra/001/users/brukew/fusion_logs/fusion_cv_%j.err

set -eo pipefail

if [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi

cd /orcd/data/satra/001/users/brukew

# Use vjepa2 env (has torch, numpy, pandas, sklearn, matplotlib)
conda activate vjepa2

LOG_DIR=/orcd/data/satra/001/users/brukew/fusion_logs
mkdir -p "$LOG_DIR"

# ============================================================================
# Paths
# ============================================================================
VJEPA_ROOT=${VJEPA_ROOT:-/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls}
# Use non-weighted PoseC3D which has full val set coverage (205 clips vs 52 for weighted)
POSEC3D_ROOT=${POSEC3D_ROOT:-/orcd/data/satra/001/users/brukew/actreg/pyskl/work_dirs/posec3d/cv/4class_conf04}
OUTPUT_DIR=${OUTPUT_DIR:-/orcd/data/satra/001/users/brukew/actreg/fusion/runs/4class_cv}

# ============================================================================
# Training config
# ============================================================================
NUM_CLASSES=${NUM_CLASSES:-4}
NUM_FOLDS=${NUM_FOLDS:-3}
NUM_EPOCHS=${NUM_EPOCHS:-100}
LR=${LR:-0.01}
BATCH_SIZE=${BATCH_SIZE:-32}

echo "============================================================"
echo "Late Fusion Training: V-JEPA2 + PoseC3D"
echo "============================================================"
echo "V-JEPA2 predictions: $VJEPA_ROOT"
echo "PoseC3D predictions: $POSEC3D_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Config: classes=$NUM_CLASSES, folds=$NUM_FOLDS, epochs=$NUM_EPOCHS, lr=$LR"
echo "============================================================"

python actreg/fusion/train_fusion_cv.py \
  --vjepa-root "$VJEPA_ROOT" \
  --posec3d-root "$POSEC3D_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --num-classes "$NUM_CLASSES" \
  --num-folds "$NUM_FOLDS" \
  --num-epochs "$NUM_EPOCHS" \
  --lr "$LR" \
  --batch-size "$BATCH_SIZE" \
  --log-level INFO

echo "============================================================"
echo "Done! Results saved to: $OUTPUT_DIR"
echo "============================================================"

