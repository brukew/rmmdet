#!/bin/bash
#SBATCH --job-name=stgcnpp_4cls_j
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/4class_single_j_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/4class_single_j_%j.err

echo "=========================================="
echo "STGCN++ Training: 4-class Single Split (Joint)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

# Configuration
CONFIG="configs/stgcn++/stgcnpp_sails_ntu60p/j.py"
ANN_FILE="data/sails/single/4class_conf04.pkl"
WORK_DIR="work_dirs/stgcnpp/single/4class_conf04_j"
EPOCHS=24
LR=0.01

echo ""
echo "Config: $CONFIG"
echo "Annotation: $ANN_FILE"
echo "Work dir: $WORK_DIR"
echo "Epochs: $EPOCHS"
echo "LR: $LR"
echo ""

# Train
python tools/train.py $CONFIG \
    --work-dir $WORK_DIR \
    --cfg-options \
        data.train.dataset.ann_file=$ANN_FILE \
        data.val.ann_file=$ANN_FILE \
        data.test.ann_file=$ANN_FILE \
        total_epochs=$EPOCHS \
        optimizer.lr=$LR \
    --validate --test-best \
    --launcher none

# Evaluate on test set
BEST_CKPT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="${WORK_DIR}/latest.pth"
fi

echo ""
echo "Evaluating best checkpoint: $BEST_CKPT"

python tools/evaluate_sails.py $CONFIG \
    -C $BEST_CKPT \
    --split test \
    --output-dir ${WORK_DIR}/eval_test \
    --cfg-options \
        data.train.dataset.ann_file=$ANN_FILE \
        data.val.ann_file=$ANN_FILE \
        data.test.ann_file=$ANN_FILE

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="



