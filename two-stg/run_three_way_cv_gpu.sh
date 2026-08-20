#!/usr/bin/env bash
# Submit all 3 folds for two-stage TAL with 3-way fusion Stage-2 (GPU jobs).
# Usage (from actreg root):
#   bash two-stg/run_three_way_cv_gpu.sh
#
# Prerequisite:
#   python fusion/export_three_way_deploy_checkpoint.py --all-folds

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

cd ${REPO_ROOT}

echo "Submitting 3-way fusion two-stage TAL jobs (one GPU job per fold)..."

for fold in 0 1 2; do
  BUNDLE="${PWD}/two-stg/fusion_checkpoints/three_way/fold_${fold}"
  if [[ ! -f "${BUNDLE}/mlp_state.pt" ]]; then
    echo "ERROR: Missing ${BUNDLE}/mlp_state.pt — run export script first."
    exit 1
  fi
  echo "Submitting fold ${fold}..."
  sbatch --export=FOLD="${fold}" two-stg/run_three_way_fold_gpu.sh
done

echo ""
echo "All 3 folds submitted. Monitor: squeue -u \$USER"
echo "Logs: two-stg/logs/two_stage_3way_fold*.out"
echo "Results: two-stg/eval_results_3way/fold{0,1,2}/"
