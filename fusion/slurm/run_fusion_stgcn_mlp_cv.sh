#!/bin/bash -l
#SBATCH -J fusion_stgcn_mlp_cv
#SBATCH -p pi_satra
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o /orcd/data/satra/001/users/brukew/fusion_logs/fusion_stgcn_mlp_cv_%j.out
#SBATCH -e /orcd/data/satra/001/users/brukew/fusion_logs/fusion_stgcn_mlp_cv_%j.err

set -eo pipefail
source ~/.bashrc
cd /orcd/data/satra/001/users/brukew
conda activate vjepa2

LOG_DIR=/orcd/data/satra/001/users/brukew/fusion_logs
mkdir -p "$LOG_DIR"

# Best V-JEPA2 model (with SAM3 cropping)
VJEPA_ROOT=${VJEPA_ROOT:-/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls}

# Best STGCN++ model (4-stream, non-weighted) - 82.4% Video Top-1
STGCN_ROOT=${STGCN_ROOT:-/orcd/data/satra/001/users/brukew/actreg/pyskl/work_dirs/stgcnpp/cv/4class_conf04_4stream}

OUTPUT_DIR=${OUTPUT_DIR:-/orcd/data/satra/001/users/brukew/actreg/fusion/runs/4class_cv_stgcn_mlp}

NUM_CLASSES=${NUM_CLASSES:-4}
NUM_FOLDS=${NUM_FOLDS:-3}
NUM_EPOCHS=${NUM_EPOCHS:-100}
LR=${LR:-0.01}
BATCH_SIZE=${BATCH_SIZE:-32}

echo "============================================================"
echo "Late Fusion Training: V-JEPA2 + STGCN++ (MLP)"
echo "============================================================"
echo "V-JEPA2 predictions: $VJEPA_ROOT"
echo "STGCN++ predictions: $STGCN_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Config: classes=$NUM_CLASSES, folds=$NUM_FOLDS, epochs=$NUM_EPOCHS, lr=$LR"
echo "============================================================"

python actreg/fusion/train_fusion_cv.py \
  --vjepa-root "$VJEPA_ROOT" \
  --skeleton-root "$STGCN_ROOT" \
  --skeleton-model-name "STGCN++" \
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


