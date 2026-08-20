#!/bin/bash
#
# Submit all 4 PoseC3D training jobs to SLURM
#
# Usage:
#   bash scripts/slurm/submit_all.sh
#

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"   # submit from repo root so child slurm-logs/ resolves

echo "=============================================="
echo "Submitting PoseC3D Training Jobs"
echo "=============================================="
echo ""

# Submit each job
echo "1. 4-class single split..."
JOB1=$(sbatch --parsable $SCRIPT_DIR/train_4class_single.sh)
echo "   Job ID: $JOB1"

echo "2. 4-class CV (3 folds)..."
JOB2=$(sbatch --parsable $SCRIPT_DIR/train_4class_cv.sh)
echo "   Job ID: $JOB2"

echo "3. 5-class single split..."
JOB3=$(sbatch --parsable $SCRIPT_DIR/train_5class_single.sh)
echo "   Job ID: $JOB3"

echo "4. 5-class CV (3 folds)..."
JOB4=$(sbatch --parsable $SCRIPT_DIR/train_5class_cv.sh)
echo "   Job ID: $JOB4"

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo ""
echo "Logs will be saved to:"
echo "  ${REPO_ROOT}/pyskl_logs/posec3d/"
echo ""
echo "Results will be in:"
echo "  work_dirs/posec3d/single/4class_conf04/"
echo "  work_dirs/posec3d/cv/4class_conf04/"
echo "  work_dirs/posec3d/single/5class_conf04/"
echo "  work_dirs/posec3d/cv/5class_conf04/"

