#!/bin/bash
#SBATCH --job-name=cut_tal_windows
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-logs/cut_tal_windows_%j.out
#SBATCH --error=slurm-logs/cut_tal_windows_%j.err

# ============================================================================
# Cut TAL window clips from full videos
# Uses create_clip_segments.py with window_id -> segment_id mapping
#
# Submit from the repo root:  sbatch dataprep/tal/cut_tal_windows.sbatch
#
# The TAL window clips already exist at $TAL_CLIPS_ROOT; this only needs to be
# re-run if the TAL window splits change.
# ============================================================================

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

set -eo pipefail
source ~/.bashrc
cd "${REPO_ROOT}"
conda activate dataprep

# Pull TAL_CLIPS_ROOT (and friends) from config.yaml via paths.py
eval "$(python "${REPO_ROOT}/paths.py" --export)"

# Disable Python output buffering so logs appear in real-time
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=========================================="

# Paths
SCRIPT_DIR="${REPO_ROOT}/dataprep/clip_gen"
SPLITS_DIR="${REPO_ROOT}/dataprep/tal/splits_cv_4class"
OUTPUT_DIR="${TAL_CLIPS_ROOT:?TAL_CLIPS_ROOT not set by paths.py --export}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Input CSVs:"
echo "  - ${SPLITS_DIR}/fold_0_train_windows.csv"
echo "  - ${SPLITS_DIR}/fold_0_val_windows.csv"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Run clip generation
# - Using fold_0 train+val with dedup covers all unique windows
# - --no-include-end-second: TAL windows are already half-open [start, end)
# - --jobs 12: Use 12 parallel ffmpeg workers (leaving headroom for overhead)
# - --dedup-by-segment: Avoid re-encoding the same window_id across CSVs

cd "${SCRIPT_DIR}"

python create_clip_segments.py \
    --csv-files "${SPLITS_DIR}/fold_0_train_windows.csv" \
                "${SPLITS_DIR}/fold_0_val_windows.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --no-include-end-second \
    --jobs 12 \
    --dedup-by-segment \
    --codec h264

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

# Show summary
echo ""
echo "Output summary:"
ls -lh "${OUTPUT_DIR}" | head -20
echo ""
echo "Total clips generated:"
find "${OUTPUT_DIR}" -name "*.mp4" | wc -l

