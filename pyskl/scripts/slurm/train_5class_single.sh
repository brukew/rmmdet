#!/bin/bash
#SBATCH --job-name=posec3d_5class_single
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/posec3d/5class_single_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/posec3d/5class_single_%j.err

echo "=========================================="
echo "PoseC3D Training: 5-class Single Split"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

# Configuration
CONFIG="configs/posec3d/slowonly_r50_sails_k400p/joint.py"
ANN_FILE="data/sails/single/5class_conf04.pkl"
WORK_DIR="work_dirs/posec3d/single/5class_conf04"
EPOCHS=12
LR=0.00125  # Scaled for 1 GPU
NUM_CLASSES=5

# Class names for 5-class
CLASS_NAMES="hands_flapping jumping one_hand_flap rocking spinning"

echo ""
echo "Config: $CONFIG"
echo "Annotation: $ANN_FILE"
echo "Work dir: $WORK_DIR"
echo "Epochs: $EPOCHS"
echo "LR: $LR"
echo "Num classes: $NUM_CLASSES"
echo ""

# Train (need to override num_classes in model)
python tools/train.py $CONFIG \
    --work-dir $WORK_DIR \
    --cfg-options \
        ann_file=$ANN_FILE \
        data.train.dataset.ann_file=$ANN_FILE \
        data.val.ann_file=$ANN_FILE \
        data.test.ann_file=$ANN_FILE \
        total_epochs=$EPOCHS \
        optimizer.lr=$LR \
        model.cls_head.num_classes=$NUM_CLASSES \
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
    --cfg-options ann_file=$ANN_FILE model.cls_head.num_classes=$NUM_CLASSES \
    --class-names $CLASS_NAMES

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="


