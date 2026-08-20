#!/usr/bin/env bash
# SLURM job: run two-stage TAL evaluation (binary ActionFormer + V-JEPA2 classification) on a GPU node.
# Usage: sbatch two-stg/run_two_stage_cv_gpu.sh
# (submit from actreg repo root, or use absolute path to script)

#SBATCH --job-name=two_stg_tal
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3:20:00
#SBATCH --output=slurm-logs/two_stage_cv_%j.out
#SBATCH --error=slurm-logs/two_stage_cv_%j.err

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

echo "=============================================="
echo "Two-Stage TAL Eval (Binary AF + V-JEPA2 4-class)"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  $(date)"
echo "=============================================="

mkdir -p ${REPO_ROOT}/two-stg/logs
export PYTHONUNBUFFERED=1

if [[ -f ~/.bashrc ]]; then
  source ~/.bashrc
fi

cd ${REPO_ROOT}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2

bash two-stg/run_two_stage_cv.sh

echo "Done. $(date)"
