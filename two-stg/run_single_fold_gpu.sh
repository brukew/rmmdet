#!/usr/bin/env bash
# SLURM job: run two-stage TAL evaluation for a SINGLE fold.
# Usage: sbatch --export=FOLD=1 two-stg/run_single_fold_gpu.sh

#SBATCH --job-name=two_stg_fold
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-logs/two_stage_fold%x_%j.out
#SBATCH --error=slurm-logs/two_stage_fold%x_%j.err

# --- Portable repo-root resolution (auto-inserted) --------------------------
# Locate the repo root (the directory containing paths.py) so this script runs
# from any clone name/location, under `bash` or `sbatch`. SLURM copies the
# script to a spool dir, so if the script path does not resolve we fall back to
# $SLURM_SUBMIT_DIR then $PWD. Override by exporting REPO_ROOT before launch.
_rmm_find_root() {
  local d="$1"
  while [ -n "$d" ] && [ "$d" != "/" ]; do
    if [ -f "$d/paths.py" ]; then printf '%s\n' "$d"; return 0; fi
    d="$(dirname "$d")"
  done
  return 1
}
if [ -z "${REPO_ROOT:-}" ] || [ ! -f "${REPO_ROOT:-x}/paths.py" ]; then
  REPO_ROOT="$(_rmm_find_root "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" 2>/dev/null && pwd)")" \
    || REPO_ROOT="$(_rmm_find_root "${SLURM_SUBMIT_DIR:-$PWD}")" \
    || REPO_ROOT="$(_rmm_find_root "$PWD")" || true
fi
if [ -z "${REPO_ROOT:-}" ] || [ ! -f "${REPO_ROOT}/paths.py" ]; then
  echo "ERROR: cannot locate repo root (paths.py). cd to the repo or export REPO_ROOT." >&2
  exit 1
fi
export REPO_ROOT
# --- end repo-root resolution ----------------------------------------------

set -e

FOLD="${FOLD:?FOLD env var required (0, 1, or 2)}"

echo "Two-Stage TAL Eval — Fold ${FOLD}"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  $(date)"

mkdir -p ${REPO_ROOT}/two-stg/logs
export PYTHONUNBUFFERED=1

if [[ -f ~/.bashrc ]]; then
  source ~/.bashrc
fi

cd ${REPO_ROOT}
source ~/miniconda3/etc/profile.d/conda.sh

# CLASSIFIER_BACKEND=vjepa2 (default) | three_way
# For three_way: export fusion first:
#   python fusion/export_three_way_deploy_checkpoint.py --all-folds
# Optional: OUTPUT_ROOT, FUSION_BUNDLE_DIR
CLASSIFIER_BACKEND="${CLASSIFIER_BACKEND:-vjepa2}"

echo "[1/2] Classifying proposals (backend=${CLASSIFIER_BACKEND})"
conda activate vjepa2

OPENTAD_EXPS="${REPO_ROOT}/OpenTAD/exps/sails_rmm"
VJEPA_CKPT="${REPO_ROOT}/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls"
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ "$CLASSIFIER_BACKEND" == "three_way" ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/two-stg/eval_results_3way"
  else
    OUTPUT_ROOT="${REPO_ROOT}/two-stg/eval_results"
  fi
fi

EVAL_ARGS=(
  --fold "$FOLD"
  --detection-json "${OPENTAD_EXPS}/actionformer_vjepa_binary_fold${FOLD}/gpu1_id99/result_detection.json"
  --checkpoint-dir "${VJEPA_CKPT}/fold_${FOLD}"
  --output-dir "${OUTPUT_ROOT}/fold${FOLD}"
  --skip-opentad-eval
)
if [[ "$CLASSIFIER_BACKEND" == "three_way" ]]; then
  EVAL_ARGS+=(--classifier-backend three_way)
  if [[ -n "${FUSION_BUNDLE_DIR:-}" ]]; then
    EVAL_ARGS+=(--fusion-bundle-dir "${FUSION_BUNDLE_DIR}")
  fi
fi

python two-stg/eval_two_stage_tal.py "${EVAL_ARGS[@]}"

echo "[2/2] Evaluating predictions with OpenTAD"
conda activate opentad
python two-stg/opentad_eval.py \
    --fold "$FOLD" \
    --predictions-csv "${OUTPUT_ROOT}/fold${FOLD}/predictions.csv" \
    --output-dir "${OUTPUT_ROOT}/fold${FOLD}"

echo "Fold ${FOLD} done. $(date)"
