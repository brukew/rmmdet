#!/bin/bash -l
# 5-Class Late Fusion: V-JEPA2 + PoseC3D (Weighted)
#
# Model configs:
#   - V-JEPA2: f64_lr1e-5_bs1_acc8_ep20_crop (5-class with SAM3 cropping)
#   - PoseC3D: 5class_conf04_weighted (best PoseC3D for 5-class: 67.0% Clip Top-1)
#
# Prerequisites:
#   - V-JEPA2 5-class must have score_classX columns (run rerun_vjepa_eval_5class.sh first)

#SBATCH -J fusion_5cls_posec3d
#SBATCH -p pi_satra
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o /orcd/data/satra/001/users/brukew/fusion_logs/fusion_5class_posec3d_cv_%j.out
#SBATCH -e /orcd/data/satra/001/users/brukew/fusion_logs/fusion_5class_posec3d_cv_%j.err

set -eo pipefail
source ~/.bashrc
cd /orcd/data/satra/001/users/brukew
conda activate vjepa2

LOG_DIR=/orcd/data/satra/001/users/brukew/fusion_logs
mkdir -p "$LOG_DIR"

# V-JEPA2 5-class with SAM3 cropping
VJEPA_ROOT=/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop

# PoseC3D 5-class weighted (best PoseC3D for 5-class)
POSEC3D_ROOT=/orcd/data/satra/001/users/brukew/actreg/pyskl/work_dirs/posec3d/cv/5class_conf04_weighted

OUTPUT_DIR=/orcd/data/satra/001/users/brukew/actreg/fusion/runs/5class_cv_posec3d_mlp

NUM_CLASSES=5
NUM_FOLDS=3
NUM_EPOCHS=100
LR=0.01
BATCH_SIZE=32

echo "============================================================"
echo "5-Class Late Fusion: V-JEPA2 + PoseC3D (MLP)"
echo "============================================================"
echo "Model configs:"
echo "  V-JEPA2: f64_lr1e-5_bs1_acc8_ep20_crop (5-class SAM3 crop)"
echo "  PoseC3D: 5class_conf04_weighted (67.0% Clip Top-1)"
echo ""
echo "V-JEPA2 predictions: $VJEPA_ROOT"
echo "PoseC3D predictions: $POSEC3D_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Config: classes=$NUM_CLASSES, folds=$NUM_FOLDS, epochs=$NUM_EPOCHS, lr=$LR"
echo "============================================================"

python actreg/fusion/train_fusion_cv.py \
  --vjepa-root "$VJEPA_ROOT" \
  --skeleton-root "$POSEC3D_ROOT" \
  --skeleton-model-name "PoseC3D-Weighted" \
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


