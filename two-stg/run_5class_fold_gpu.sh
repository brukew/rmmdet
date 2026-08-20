#!/usr/bin/env bash
# SLURM job: run two-stage TAL evaluation with 5-class classifier (4 RMM + BG).
# Usage: sbatch --export=FOLD=0 two-stg/run_5class_fold_gpu.sh

#SBATCH --job-name=two_stg_5cls
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-logs/two_stage_5cls_fold%x_%j.out
#SBATCH --error=slurm-logs/two_stage_5cls_fold%x_%j.err

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

echo "Two-Stage TAL Eval (5-class) — Fold ${FOLD}"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  $(date)"

mkdir -p ${REPO_ROOT}/two-stg/logs
export PYTHONUNBUFFERED=1

if [[ -f ~/.bashrc ]]; then
  source ~/.bashrc
fi

cd ${REPO_ROOT}
source ~/miniconda3/etc/profile.d/conda.sh

echo "[1/2] Classifying proposals with V-JEPA2 5-class (4 RMM + BG)"
conda activate vjepa2

OPENTAD_EXPS="${REPO_ROOT}/OpenTAD/exps/sails_rmm"
# Use the 5-class checkpoint (includes background as class 4)
VJEPA_CKPT="${REPO_ROOT}/v-jepa/runs/vjepa2_tal_cv_5class_balanced"
OUTPUT_ROOT="${REPO_ROOT}/two-stg/eval_results_5class"

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
