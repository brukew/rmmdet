#!/usr/bin/env bash
# SLURM job: run two-stage TAL evaluation with 5-class classifier (4 RMM + BG).
# Usage: sbatch --export=FOLD=0 two-stg/run_5class_fold_gpu.sh

#SBATCH --job-name=two_stg_5cls
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/two_stage_5cls_fold%x_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/two_stage_5cls_fold%x_%j.err

set -e

FOLD="${FOLD:?FOLD env var required (0, 1, or 2)}"

echo "Two-Stage TAL Eval (5-class) — Fold ${FOLD}"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  $(date)"

mkdir -p /orcd/data/satra/001/users/brukew/actreg/two-stg/logs
export PYTHONUNBUFFERED=1

if [[ -f ~/.bashrc ]]; then
  source ~/.bashrc
fi

cd /orcd/data/satra/001/users/brukew/actreg
source ~/miniconda3/etc/profile.d/conda.sh

echo "[1/2] Classifying proposals with V-JEPA2 5-class (4 RMM + BG)"
conda activate vjepa2

OPENTAD_EXPS="/orcd/data/satra/001/users/brukew/actreg/OpenTAD/exps/sails_rmm"
# Use the 5-class checkpoint (includes background as class 4)
VJEPA_CKPT="/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_tal_cv_5class_balanced"
OUTPUT_ROOT="/orcd/data/satra/001/users/brukew/actreg/two-stg/eval_results_5class"

python two-stg/eval_two_stage_tal.py \
    --fold "$FOLD" \
    --detection-json "${OPENTAD_EXPS}/actionformer_vjepa_binary_fold${FOLD}/gpu1_id99/result_detection.json" \
    --checkpoint-dir "${VJEPA_CKPT}/fold_${FOLD}" \
    --output-dir "${OUTPUT_ROOT}/fold${FOLD}" \
    --num-classes 5 \
    --skip-opentad-eval

echo "[2/2] Evaluating predictions with OpenTAD"
conda activate opentad
python two-stg/opentad_eval.py \
    --fold "$FOLD" \
    --predictions-csv "${OUTPUT_ROOT}/fold${FOLD}/predictions.csv" \
    --output-dir "${OUTPUT_ROOT}/fold${FOLD}"

echo "Fold ${FOLD} (5-class) done. $(date)"
