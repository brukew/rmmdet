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
#SBATCH -o /orcd/data/satra/001/users/brukew/fusion_logs/fusion_5class_stgcn_cv_%j.out
#SBATCH -e /orcd/data/satra/001/users/brukew/fusion_logs/fusion_5class_stgcn_cv_%j.err

set -eo pipefail
source ~/.bashrc
cd /orcd/data/satra/001/users/brukew
conda activate vjepa2

LOG_DIR=/orcd/data/satra/001/users/brukew/fusion_logs
mkdir -p "$LOG_DIR"

# V-JEPA2 5-class with SAM3 cropping
VJEPA_ROOT=/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop

# STGCN++ 5-class 4-stream (non-weighted, has prediction CSVs)
STGCN_ROOT=/orcd/data/satra/001/users/brukew/actreg/pyskl/work_dirs/stgcnpp/cv/5class_conf04_4stream

OUTPUT_DIR=/orcd/data/satra/001/users/brukew/actreg/fusion/runs/5class_cv_stgcn_mlp

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

python actreg/fusion/train_fusion_cv.py \
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


