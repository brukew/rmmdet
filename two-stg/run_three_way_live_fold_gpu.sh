#!/usr/bin/env bash
# Full pipeline: proposal pickle + skeleton inference (or reuse) + decontaminated 3-way TAL eval.
#
# Prerequisite:
#   python fusion/export_three_way_deploy_checkpoint.py --all-folds
#
# Either precompute skeleton scores (recommended separate GPU job):
#   sbatch --export=FOLD=N two-stg/run_proposal_skeleton_inference.sh
#   (set SKIP_SKELETON_INF=1 below if scores already exist)
#
# Usage:
#   cd actreg && sbatch --export=FOLD=0 two-stg/run_three_way_live_fold_gpu.sh
#
# Env:
#   OUTPUT_ROOT     default: two-stg/eval_results_3way_live
#   FUSION_BUNDLE_DIR
#   BATCH_SIZE
#   PICKLE_DIR, SCORE_DIR  (same as run_proposal_skeleton_inference.sh)
#   SKIP_SKELETON_INF  if 1, skip pickle build + pyskl inference (expect CSVs under SCORE_DIR)
#   SKIP_BUILD_ONLY    if 1, only skip pickle build (still run pyskl if SKIP_SKELETON_INF=0)

#SBATCH --job-name=two_stg_3way_live
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/two_stage_3way_live_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/two_stage_3way_live_%j.err

# No `set -u`: conda activate/deactivate hooks use unset vars (e.g. CONDA_BACKUP_GXX).
set -eo pipefail

FOLD="${FOLD:?FOLD env var required (0, 1, or 2)}"

echo "Two-Stage TAL (3-way LIVE skeleton scores) — Fold ${FOLD}"
echo "Job ID: ${SLURM_JOB_ID:-local}  Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-unset}  $(date)"

mkdir -p /orcd/data/satra/001/users/brukew/actreg/two-stg/logs
export PYTHONUNBUFFERED=1

if [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.bashrc"
fi

cd /orcd/data/satra/001/users/brukew/actreg
source "${HOME}/miniconda3/etc/profile.d/conda.sh"

BUNDLE="${FUSION_BUNDLE_DIR:-${PWD}/two-stg/fusion_checkpoints/three_way/fold_${FOLD}}"
if [[ ! -f "${BUNDLE}/mlp_state.pt" ]]; then
  echo "ERROR: Missing fusion bundle: ${BUNDLE}/mlp_state.pt"
  exit 1
fi

PICKLE_DIR="${PICKLE_DIR:-${PWD}/two-stg/proposal_pickles}"
SCORE_DIR="${SCORE_DIR:-${PWD}/two-stg/proposal_scores}"
POSE_CSV="${SCORE_DIR}/posec3d/fold${FOLD}/predictions_clip.csv"
STGCN_CSV="${SCORE_DIR}/stgcnpp_fused_4stream/fold${FOLD}/predictions_clip.csv"

if [[ "${SKIP_SKELETON_INF:-0}" != "1" ]]; then
  export FOLD
  export PICKLE_DIR
  export SCORE_DIR
  if [[ "${SKIP_BUILD_ONLY:-0}" == "1" ]]; then
    export SKIP_BUILD=1
  else
    export SKIP_BUILD=0
  fi
  echo "[Skeleton] Running inline proposal skeleton inference (same as run_proposal_skeleton_inference.sh)"
  bash two-stg/run_proposal_skeleton_inference.sh
else
  echo "[Skeleton] SKIP_SKELETON_INF=1 — using existing CSVs"
fi

if [[ ! -f "${POSE_CSV}" || ! -f "${STGCN_CSV}" ]]; then
  echo "ERROR: Missing proposal score CSVs."
  echo "  Expected: ${POSE_CSV}"
  echo "            ${STGCN_CSV}"
  echo "Run: sbatch --export=FOLD=${FOLD} two-stg/run_proposal_skeleton_inference.sh"
  exit 1
fi

OPENTAD_EXPS="${PWD}/OpenTAD/exps/sails_rmm"
VJEPA_CKPT="${PWD}/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/two-stg/eval_results_3way_live}"
OUT_DIR="${OUTPUT_ROOT}/fold${FOLD}"

echo "[Eval] V-JEPA2 + 3-way MLP + live proposal skeleton scores -> ${OUT_DIR}"
conda activate vjepa2
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_ARGS=(
  --fold "$FOLD"
  --detection-json "${OPENTAD_EXPS}/actionformer_vjepa_binary_fold${FOLD}/gpu1_id99/result_detection.json"
  --checkpoint-dir "${VJEPA_CKPT}/fold_${FOLD}"
  --output-dir "$OUT_DIR"
  --classifier-backend three_way
  --fusion-bundle-dir "$BUNDLE"
  --posec3d-proposal-scores "$POSE_CSV"
  --stgcn-proposal-scores "$STGCN_CSV"
  --device cuda
  --batch-size "$BATCH_SIZE"
  --skip-opentad-eval
)
python two-stg/eval_two_stage_tal.py "${EVAL_ARGS[@]}"

echo "[OpenTAD]"
conda activate opentad
python two-stg/opentad_eval.py \
  --fold "$FOLD" \
  --predictions-csv "${OUT_DIR}/predictions.csv" \
  --output-dir "$OUT_DIR"

echo "[info.json]"
conda activate vjepa2
python two-stg/write_info_json.py \
  --fold "$FOLD" \
  --output-dir "$OUT_DIR" \
  --stage2-backend three_way \
  --fusion-bundle-dir "$BUNDLE" \
  --posec3d-proposal-scores "$POSE_CSV" \
  --stgcn-proposal-scores "$STGCN_CSV"

echo "Fold ${FOLD} (3-way live) done. ${OUT_DIR}  $(date)"
