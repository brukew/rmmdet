#!/bin/bash
#SBATCH --job-name=test_regularized
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/test_regularized_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/test_regularized_%j.err

echo "=========================================="
echo "Testing REGULARIZED configs to reduce overfitting"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

# ============================================
# Test 1: PoseC3D Regularized (frozen backbone + dropout)
# ============================================
echo ""
echo "######################################################"
echo "# PoseC3D REGULARIZED (frozen backbone + dropout=0.5)"
echo "######################################################"

python tools/train.py configs/posec3d/slowonly_r50_sails_k400p/joint_regularized.py \
    --work-dir work_dirs/posec3d/single/4class_regularized \
    --validate \
    --launcher none \
    --cfg-options \
        data.train.dataset.ann_file=data/sails/single/4class_conf04.pkl \
        data.val.ann_file=data/sails/single/4class_conf04.pkl \
        data.test.ann_file=data/sails/single/4class_conf04.pkl

# Evaluate
BEST_CKPT=$(ls -t work_dirs/posec3d/single/4class_regularized/best_*.pth 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    python tools/evaluate_sails.py configs/posec3d/slowonly_r50_sails_k400p/joint_regularized.py \
        -C $BEST_CKPT \
        --split test \
        --output-dir work_dirs/posec3d/single/4class_regularized/eval_test \
        --cfg-options \
            data.train.dataset.ann_file=data/sails/single/4class_conf04.pkl \
            data.val.ann_file=data/sails/single/4class_conf04.pkl \
            data.test.ann_file=data/sails/single/4class_conf04.pkl
fi

# ============================================
# Test 2: STGCN++ Regularized (dropout + lower LR)
# ============================================
echo ""
echo "######################################################"
echo "# STGCN++ REGULARIZED (dropout=0.5 + lower LR)"
echo "######################################################"

python tools/train.py configs/stgcn++/stgcnpp_sails_ntu60p/j_regularized.py \
    --work-dir work_dirs/stgcnpp/single/4class_regularized_j \
    --validate \
    --launcher none \
    --cfg-options \
        data.train.dataset.ann_file=data/sails/single/4class_conf04.pkl \
        data.val.ann_file=data/sails/single/4class_conf04.pkl \
        data.test.ann_file=data/sails/single/4class_conf04.pkl

# Evaluate
BEST_CKPT=$(ls -t work_dirs/stgcnpp/single/4class_regularized_j/best_*.pth 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    python tools/evaluate_sails.py configs/stgcn++/stgcnpp_sails_ntu60p/j_regularized.py \
        -C $BEST_CKPT \
        --split test \
        --output-dir work_dirs/stgcnpp/single/4class_regularized_j/eval_test \
        --cfg-options \
            data.train.dataset.ann_file=data/sails/single/4class_conf04.pkl \
            data.val.ann_file=data/sails/single/4class_conf04.pkl \
            data.test.ann_file=data/sails/single/4class_conf04.pkl
fi

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

echo ""
echo "SUMMARY: Compare these metrics to baseline runs:"
echo "  - PoseC3D Regularized: work_dirs/posec3d/single/4class_regularized/"
echo "  - STGCN++ Regularized: work_dirs/stgcnpp/single/4class_regularized_j/"
echo ""
echo "Key changes:"
echo "  PoseC3D: frozen_stages=4, dropout=0.5, weight_decay=0.001, epochs=10"
echo "  STGCN++: dropout=0.5, lr=0.005, weight_decay=0.001, epochs=16, RandomRot aug"








