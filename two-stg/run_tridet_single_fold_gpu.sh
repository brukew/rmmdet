#!/usr/bin/env bash
# Run 5: TriDet + V-JEPA2 4-class two-stage TAL.
# Mirrors run_single_fold_gpu.sh but uses TriDet detections.
#
# Usage: sbatch --export=FOLD=0 two-stg/run_tridet_single_fold_gpu.sh

#SBATCH --job-name=r5_tridet
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-logs/run5_tridet_%j.out
#SBATCH --error=slurm-logs/run5_tridet_%j.err

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

echo "Run 5: TriDet + V-JEPA2 4-class — Fold ${FOLD}"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  $(date)"

mkdir -p ${REPO_ROOT}/two-stg/logs
export PYTHONUNBUFFERED=1

if [[ -f ~/.bashrc ]]; then
  source ~/.bashrc
fi

cd ${REPO_ROOT}
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
