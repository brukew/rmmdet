#!/bin/bash
#SBATCH --job-name=posec3d_5cls_s_w
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/posec3d/5class_single_weighted_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/posec3d/5class_single_weighted_%j.err

echo "=========================================="
echo "PoseC3D Training: 5-class Single Split WITH CLASS WEIGHTING"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

# Configuration
CONFIG="configs/posec3d/slowonly_r50_sails_k400p/joint.py"
ANN_FILE="data/sails/single/5class_conf04.pkl"
WORK_DIR="work_dirs/posec3d/single/5class_conf04_weighted_sqrt"
EPOCHS=12
LR=0.00125
NUM_CLASSES=5

echo ""
echo "Config: $CONFIG"
echo "Annotation: $ANN_FILE"
echo "Work dir: $WORK_DIR"
echo "Epochs: $EPOCHS"
echo "LR: $LR"
echo "Num classes: $NUM_CLASSES"
echo ""

# Train with class weights (sqrt scaling recommended for 5+ classes)
python tools/train_weighted.py $CONFIG \
    --ann-file $ANN_FILE \
    --work-dir $WORK_DIR \
    --total-epochs $EPOCHS \
    --lr $LR \
    --num-classes $NUM_CLASSES \
    --weight-scale sqrt \
    --validate \
    --test-best \
    --launcher none

# Evaluate
BEST_CKPT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="${WORK_DIR}/latest.pth"
fi

echo ""
echo "Evaluating: $BEST_CKPT"

python tools/evaluate_sails.py $CONFIG \
    -C $BEST_CKPT \
    --split test \
    --output-dir ${WORK_DIR}/eval_test \
    --class-names "hands flapping" "jumping" "one hand flap" "rocking" "spinning" \
    --cfg-options \
        data.train.dataset.ann_file=$ANN_FILE \
        data.val.ann_file=$ANN_FILE \
        data.test.ann_file=$ANN_FILE \
        model.cls_head.num_classes=$NUM_CLASSES

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="



