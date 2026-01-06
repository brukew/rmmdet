#!/bin/bash
#SBATCH --job-name=stgcnpp_tal_ce_bal
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/tal/stgcnpp_tal_5class_cv_ce_balanced_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/tal/stgcnpp_tal_5class_cv_ce_balanced_%j.err

# ============================================================================
# STGCN++ TAL Training: 5-class CV 4-Stream with Cross-Entropy + Balanced Sampling
#
# Features:
#   - Cross-Entropy loss
#   - Inverse-frequency class weights (computed automatically)
#   - Background subsampling (10%) to reduce class imbalance
#   - Upsampling 'rocking' and 'spinning' to 'jumping' levels
#   - Early stopping (patience=5)
#   - All 4 modalities: joint (j), bone (b), joint motion (jm), bone motion (bm)
#
# Runs on all 3 folds for each modality.
# ============================================================================

set -eo pipefail

echo "=========================================="
echo "STGCN++ TAL Training: 5-class CV 4-Stream with CE + Balanced Sampling"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Ensure logs directory exists
mkdir -p /orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/tal

# Ensure real-time logging
export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

# Configuration
CONFIG_DIR="configs/stgcn++/stgcnpp_sails_ntu60p"
BASE_ANN_DIR="data/sails/tal/cv_4class/5class_windows_conf04"
BASE_WORK_DIR="work_dirs/stgcnpp/tal/cv_4class_5class_ce_balanced"
EPOCHS=24

# Class probabilities for sampling:
# 0: hands flapping (500) -> 1.0
# 1: jumping (237) -> 1.0 (target for upsampling)
# 2: rocking (123) -> 237/123 = 1.93
# 3: spinning (26) -> 237/26 = 9.12
# 4: background (7373) -> 0.1 (subsample to 10%)
CLASS_PROB="[1.0, 1.0, 1.93, 9.12, 0.1]"

# Modalities (CE configs, not focal)
MODALITIES="j b jm bm"

echo ""
echo "Config dir: $CONFIG_DIR"
echo "Annotation dir: $BASE_ANN_DIR"
echo "Work dir: $BASE_WORK_DIR"
echo "Epochs: $EPOCHS (with early stopping patience=5)"
echo "Class Probabilities: $CLASS_PROB"
echo "Loss: Cross-Entropy + inverse class weights"
echo "Modalities: $MODALITIES"
echo ""

# Train each modality across all 3 folds
for MODALITY in $MODALITIES; do
    # Use CE configs (e.g., j_tal_5class.py, NOT j_tal_5class_focal.py)
    CONFIG="${CONFIG_DIR}/${MODALITY}_tal_5class.py"
    
    echo ""
    echo "======================================================"
    echo "MODALITY: $MODALITY"
    echo "======================================================"
    
    for FOLD in 0 1 2; do
        ANN_FILE="${BASE_ANN_DIR}/fold${FOLD}.pkl"
        WORK_DIR="${BASE_WORK_DIR}/${MODALITY}/fold${FOLD}"
        
        echo ""
        echo "------------------------------------------------------"
        echo "MODALITY: $MODALITY | FOLD: $FOLD"
        echo "------------------------------------------------------"
        echo "Config: $CONFIG"
        echo "Annotation: $ANN_FILE"
        echo "Work dir: $WORK_DIR"
        
        # Check if pickle exists
        if [ ! -f "$ANN_FILE" ]; then
            echo "ERROR: Annotation file not found: $ANN_FILE"
            continue
        fi
        
        # Train with CE loss, class weights, and balanced sampling
        python tools/train_weighted.py $CONFIG \
            --ann-file $ANN_FILE \
            --work-dir $WORK_DIR \
            --total-epochs $EPOCHS \
            --class-prob "$CLASS_PROB" \
            --validate \
            --launcher none
        
        # Evaluate on val set
        BEST_CKPT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)
        if [ -z "$BEST_CKPT" ]; then
            BEST_CKPT="${WORK_DIR}/latest.pth"
        fi
        
        echo ""
        echo "Evaluating $MODALITY fold ${FOLD}: $BEST_CKPT"
        
        python tools/evaluate_sails.py $CONFIG \
            -C $BEST_CKPT \
            --split val \
            --output-dir ${WORK_DIR}/eval_val \
            --task tal \
            --cfg-options \
                data.train.dataset.ann_file=$ANN_FILE \
                data.val.ann_file=$ANN_FILE \
                model.cls_head.num_classes=5
    done
done

# Compute CV Summary using standalone script
python tools/summarize_cv_results.py \
    --work-dir $BASE_WORK_DIR \
    --model stgcnpp \
    --task tal \
    --loss ce \
    --modalities $MODALITIES \
    --metadata "class_prob=$CLASS_PROB" "patience=5"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

