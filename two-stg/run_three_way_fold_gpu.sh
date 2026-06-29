#!/usr/bin/env bash
# SLURM job: two-stage TAL with Stage-2 = live V-JEPA2 + exported 3-way MLP fusion
# (PoseC3D/STGCN++ via skeleton_index.csv; see fusion/export_three_way_deploy_checkpoint.py).
#
# Prerequisite:
#   python fusion/export_three_way_deploy_checkpoint.py --all-folds
#
# Usage:
#   cd /path/to/actreg && sbatch --export=FOLD=0 two-stg/run_three_way_fold_gpu.sh
#
# Optional env:
#   OUTPUT_ROOT  — default: two-stg/eval_results_3way
#   FUSION_BUNDLE_DIR — override per-fold fusion bundle
#   BATCH_SIZE — classifier batch size (default 32)
#
# Partition: mit_normal_gpu (same family as run_two_stage_cv_gpu.sh). Override at submit:
#   sbatch --partition=OTHER_PART ... (if your site supports it)

#SBATCH --job-name=two_stg_3way
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/two_stage_3way_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/two_stage_3way_%j.err

set -e

FOLD="${FOLD:?FOLD env var required (0, 1, or 2)}"

echo "Two-Stage TAL Eval (3-way fusion) — Fold ${FOLD}"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-unset}  $(date)"

mkdir -p /orcd/data/satra/001/users/brukew/actreg/two-stg/logs
export PYTHONUNBUFFERED=1

if [[ -f ~/.bashrc ]]; then
  source ~/.bashrc
fi

cd /orcd/data/satra/001/users/brukew/actreg
source ~/miniconda3/etc/profile.d/conda.sh

BUNDLE="${FUSION_BUNDLE_DIR:-${PWD}/two-stg/fusion_checkpoints/three_way/fold_${FOLD}}"
if [[ ! -f "${BUNDLE}/mlp_state.pt" ]]; then
  echo "ERROR: Missing fusion bundle: ${BUNDLE}/mlp_state.pt"
  echo "Run: python fusion/export_three_way_deploy_checkpoint.py --fold ${FOLD}"
  exit 1
fi

OPENTAD_EXPS="${PWD}/OpenTAD/exps/sails_rmm"
VJEPA_CKPT="${PWD}/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/two-stg/eval_results_3way}"
OUT_DIR="${OUTPUT_ROOT}/fold${FOLD}"

echo "[1/3] Classifying proposals (V-JEPA2 RGB + 3-way MLP on GPU)"
conda activate vjepa2

BATCH_SIZE="${BATCH_SIZE:-32}"

EVAL_ARGS=(
  --fold "$FOLD"
  --detection-json "${OPENTAD_EXPS}/actionformer_vjepa_binary_fold${FOLD}/gpu1_id99/result_detection.json"
  --checkpoint-dir "${VJEPA_CKPT}/fold_${FOLD}"
  --output-dir "$OUT_DIR"
  --classifier-backend three_way
  --fusion-bundle-dir "$BUNDLE"
  --device cuda
  --batch-size "$BATCH_SIZE"
  --skip-opentad-eval
)
python two-stg/eval_two_stage_tal.py "${EVAL_ARGS[@]}"

echo "[2/3] OpenTAD native evaluation"
conda activate opentad
python two-stg/opentad_eval.py \
  --fold "$FOLD" \
  --predictions-csv "${OUT_DIR}/predictions.csv" \
  --output-dir "$OUT_DIR"

echo "[3/3] info.json metadata"
conda activate vjepa2
INFO_ARGS=(
  --fold "$FOLD"
  --output-dir "$OUT_DIR"
  --stage2-backend three_way
  --fusion-bundle-dir "$BUNDLE"
)
python two-stg/write_info_json.py "${INFO_ARGS[@]}"

echo "Fold ${FOLD} (3-way) done. Results: ${OUT_DIR}  $(date)"
