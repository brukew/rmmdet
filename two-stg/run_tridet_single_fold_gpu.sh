#!/usr/bin/env bash
# Run 5: TriDet + V-JEPA2 4-class two-stage TAL.
# Mirrors run_single_fold_gpu.sh but uses TriDet detections.
#
# Usage: sbatch --export=FOLD=0 two-stg/run_tridet_single_fold_gpu.sh

#SBATCH --job-name=r5_tridet
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/run5_tridet_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/run5_tridet_%j.err

set -e

FOLD="${FOLD:?FOLD env var required (0, 1, or 2)}"

echo "Run 5: TriDet + V-JEPA2 4-class — Fold ${FOLD}"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  $(date)"

mkdir -p /orcd/data/satra/001/users/brukew/actreg/two-stg/logs
export PYTHONUNBUFFERED=1

if [[ -f ~/.bashrc ]]; then
  source ~/.bashrc
fi

cd /orcd/data/satra/001/users/brukew/actreg
source ~/miniconda3/etc/profile.d/conda.sh

OPENTAD_EXPS="OpenTAD/exps/sails_rmm"
VJEPA_CKPT="v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls"
OUTPUT_ROOT="two-stg/eval_results_tridet"

echo "[1/2] Classifying proposals"
conda activate vjepa2

python two-stg/eval_two_stage_tal.py \
  --fold "$FOLD" \
  --detection-json "${OPENTAD_EXPS}/tridet_vjepa_binary_fold${FOLD}/gpu1_id99/result_detection.json" \
  --checkpoint-dir "${VJEPA_CKPT}/fold_${FOLD}" \
  --output-dir "${OUTPUT_ROOT}/fold${FOLD}" \
  --skip-opentad-eval

echo "[2/2] Evaluating predictions with OpenTAD"
conda activate opentad
python two-stg/opentad_eval.py \
  --fold "$FOLD" \
  --predictions-csv "${OUTPUT_ROOT}/fold${FOLD}/predictions.csv" \
  --output-dir "${OUTPUT_ROOT}/fold${FOLD}"

echo "Fold ${FOLD} done. $(date)"
