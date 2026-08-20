#!/usr/bin/env bash
# SLURM job: convert finished two-stage predictions into OpenTAD-native outputs.
# Usage:
#   sbatch --export=FOLDS=1,2 two-stg/run_opentad_postprocess.sh

#SBATCH --job-name=two_stg_otad
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-logs/two_stage_opentad_%j.out
#SBATCH --error=slurm-logs/two_stage_opentad_%j.err

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

set -euo pipefail

FOLDS="${FOLDS:?FOLDS env var required, e.g. 1,2 or 0}"
SOURCE_JOB_IDS="${SOURCE_JOB_IDS:-}"

echo "Two-Stage OpenTAD Postprocess"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  $(date)"
echo "Folds: $FOLDS"

mkdir -p ${REPO_ROOT}/two-stg/logs

if [[ -f ~/.bashrc ]]; then
  set +u
  source ~/.bashrc
  set -u
fi

cd ${REPO_ROOT}
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opentad
set -u

OUTPUT_ROOT="${REPO_ROOT}/two-stg/eval_results"

IFS=',' read -r -a fold_array <<< "$FOLDS"
for fold in "${fold_array[@]}"; do
  pred_csv="${OUTPUT_ROOT}/fold${fold}/predictions.csv"
  out_dir="${OUTPUT_ROOT}/fold${fold}"

  if [[ ! -f "$pred_csv" ]]; then
    echo "Missing predictions for fold ${fold}: ${pred_csv}"
    exit 1
  fi

  echo "Running OpenTAD eval for fold ${fold}"
  python two-stg/opentad_eval.py \
    --fold "$fold" \
    --predictions-csv "$pred_csv" \
    --output-dir "$out_dir"

  python two-stg/write_info_json.py \
    --fold "$fold" \
    --output-dir "$out_dir" \
    --source-job-ids "$SOURCE_JOB_IDS" \
    --postprocess-job-id "${SLURM_JOB_ID:-}" \
    --postprocess-dependency "${SLURM_JOB_DEPENDENCY:-}"
done

echo "OpenTAD postprocess done. $(date)"
