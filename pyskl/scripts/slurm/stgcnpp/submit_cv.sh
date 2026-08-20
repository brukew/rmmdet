#!/bin/bash
# Submit cv STGCN++ 4-stream training jobs for SAILS
# Usage: bash scripts/slurm/stgcnpp/submit_cv.sh [--dry-run]

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

cd "${REPO_ROOT}"

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

# Create log directory
mkdir -p ${REPO_ROOT}/pyskl_logs/stgcnpp

echo "=========================================="
echo "STGCN++ SAILS 4-Stream CV Training"
echo "Date: $(date)"
echo "=========================================="
echo ""
echo "Each job trains all 4 modalities (j, b, jm, bm) and computes fusion."
echo "Fusion weights: 2*j + 2*b + 1*jm + 1*bm"
echo ""

SCRIPTS=(
    "pyskl/scripts/slurm/stgcnpp/train_4class_cv_4stream.sh"
    "pyskl/scripts/slurm/stgcnpp/train_5class_cv_4stream.sh"
)

JOB_IDS=()

for SCRIPT in "${SCRIPTS[@]}"; do
    echo "Submitting: $SCRIPT"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would run: sbatch $SCRIPT"
    else
        JOB_ID=$(sbatch $SCRIPT | awk '{print $4}')
        JOB_IDS+=($JOB_ID)
        echo "  Submitted job: $JOB_ID"
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total jobs: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Training per job:"
echo "  - CV (3-fold):  4 modalities × 3 folds × 24 epochs"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Logs at: ${REPO_ROOT}/pyskl_logs/stgcnpp/"
echo "=========================================="
