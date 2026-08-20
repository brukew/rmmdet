#!/usr/bin/env bash
# Submit all 3 folds: full 3-way pipeline with live proposal skeleton scores + decontaminated MLP.
#
# Each job is long (pickle + 5 skeleton runs + V-JEPA2 eval). If pickles already
# exist under two-stg/proposal_pickles/, export SKIP_BUILD_ONLY=1 to skip rebuild.
#
# Usage (from actreg root):
#   SKIP_BUILD_ONLY=1 bash two-stg/run_three_way_live_cv_gpu.sh
#   bash two-stg/run_three_way_live_cv_gpu.sh   # rebuilds pickles each job (slow)

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

cd ${REPO_ROOT}

for fold in 0 1 2; do
  BUNDLE="${PWD}/two-stg/fusion_checkpoints/three_way/fold_${fold}"
  if [[ ! -f "${BUNDLE}/mlp_state.pt" ]]; then
    echo "ERROR: Missing ${BUNDLE}/mlp_state.pt — run fusion/export_three_way_deploy_checkpoint.py --all-folds"
    exit 1
  fi
  echo "Submitting 3-way LIVE fold ${fold}..."
  if [[ "${SKIP_BUILD_ONLY:-0}" == "1" ]]; then
    sbatch --export=FOLD="${fold}",SKIP_BUILD_ONLY=1 two-stg/run_three_way_live_fold_gpu.sh
  else
    sbatch --export=FOLD="${fold}" two-stg/run_three_way_live_fold_gpu.sh
  fi
done

echo "Submitted 3 jobs. Monitor: squeue -u \$USER"
