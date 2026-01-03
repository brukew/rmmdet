#!/bin/bash -l
#SBATCH -J fusion_3way_cv
#SBATCH -p pi_satra
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o /orcd/data/satra/001/users/brukew/fusion_logs/fusion_3way_cv_%j.out
#SBATCH -e /orcd/data/satra/001/users/brukew/fusion_logs/fusion_3way_cv_%j.err

set -eo pipefail
source ~/.bashrc
cd /orcd/data/satra/001/users/brukew
conda activate vjepa2

LOG_DIR=/orcd/data/satra/001/users/brukew/fusion_logs
mkdir -p "$LOG_DIR"

# Best V-JEPA2 model (with SAM3 cropping)
VJEPA_ROOT=/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls

# Best PoseC3D model (non-weighted) - 74.6% Video Top-1
POSEC3D_ROOT=/orcd/data/satra/001/users/brukew/actreg/pyskl/work_dirs/posec3d/cv/4class_conf04

# Best STGCN++ model (4-stream, non-weighted) - 82.4% Video Top-1
STGCN_ROOT=/orcd/data/satra/001/users/brukew/actreg/pyskl/work_dirs/stgcnpp/cv/4class_conf04_4stream

OUTPUT_DIR=/orcd/data/satra/001/users/brukew/actreg/fusion/runs/4class_cv_3way

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

python actreg/fusion/train_fusion_cv.py \
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



