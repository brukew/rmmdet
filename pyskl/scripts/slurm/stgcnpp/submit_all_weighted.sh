#!/bin/bash
# Submit all STGCN++ weighted training jobs

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

echo "Submitting STGCN++ weighted 4-stream training jobs..."

cd "${REPO_ROOT}"
_SUBMIT_DIR="pyskl/scripts/slurm/stgcnpp"

# Single splits
echo "Submitting 4-class single weighted..."
sbatch "${_SUBMIT_DIR}/train_4class_single_4stream_weighted.sh"

echo "Submitting 5-class single weighted..."
sbatch "${_SUBMIT_DIR}/train_5class_single_4stream_weighted.sh"

# CV splits
echo "Submitting 4-class CV weighted..."
sbatch "${_SUBMIT_DIR}/train_4class_cv_4stream_weighted.sh"

echo "Submitting 5-class CV weighted..."
sbatch "${_SUBMIT_DIR}/train_5class_cv_4stream_weighted.sh"

echo ""
echo "All weighted jobs submitted! Check status with: squeue -u \$USER"

