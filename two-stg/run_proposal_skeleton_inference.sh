#!/usr/bin/env bash
# Build proposal pickle (CPU) + run PoseC3D and STGCN++ (4-stream) inference on GPU.
#
# Prereqs: conda envs ``vjepa2`` (pickle build) and ``pyskl`` (mmcv / evaluate_sails).
#
# Usage (from actreg root):
#   sbatch --export=FOLD=0 two-stg/run_proposal_skeleton_inference.sh
#
# Env overrides:
#   PICKLE_DIR   default: two-stg/proposal_pickles
#   SCORE_DIR    default: two-stg/proposal_scores
#   SKIP_BUILD   if set to 1, skip build_proposal_poses.py (reuse existing pickle)

#SBATCH --job-name=prop_skel_inf
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-logs/proposal_skeleton_%j.out
#SBATCH --error=slurm-logs/proposal_skeleton_%j.err

# No `set -u`: conda hooks may reference unset variables during activate/deactivate.
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

FOLD="${FOLD:?Set FOLD=0|1|2}"

echo "Proposal skeleton inference — fold ${FOLD}  Job ${SLURM_JOB_ID:-local}  $(date)"

mkdir -p ${REPO_ROOT}/two-stg/logs
export PYTHONUNBUFFERED=1

if [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.bashrc"
fi

cd ${REPO_ROOT}
source "${HOME}/miniconda3/etc/profile.d/conda.sh"

PICKLE_DIR="${PICKLE_DIR:-${PWD}/two-stg/proposal_pickles}"
SCORE_DIR="${SCORE_DIR:-${PWD}/two-stg/proposal_scores}"
PKL="${PICKLE_DIR}/proposals_fold${FOLD}.pkl"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "[1/7] Build proposal pickle (vjepa2 env, h5py)"
  conda activate vjepa2
  python two-stg/build_proposal_poses.py --fold "${FOLD}" --output-dir "${PICKLE_DIR}"
else
  echo "[1/7] SKIP_BUILD=1 — using existing ${PKL}"
fi

if [[ ! -f "${PKL}" ]]; then
  echo "ERROR: Missing pickle ${PKL}"
  exit 1
fi

conda activate pyskl
cd "${PWD}/pyskl"

best_ckpt() {
  local pattern="$1"
  # shellcheck disable=SC2086
  ls ${pattern} 2>/dev/null | head -1
}

POSE_CKPT=$(best_ckpt "work_dirs/posec3d/cv/4class_conf04/fold${FOLD}/best_*.pth")
if [[ -z "${POSE_CKPT}" ]]; then
  echo "ERROR: No PoseC3D checkpoint in fold${FOLD}"
  exit 1
fi

POUT="${SCORE_DIR}/posec3d/fold${FOLD}"
mkdir -p "${POUT}"
echo "[2/7] PoseC3D inference -> ${POUT}"
python tools/evaluate_sails.py configs/posec3d/slowonly_r50_sails_k400p/joint.py \
  -C "${POSE_CKPT}" \
  --split val \
  --output-dir "${POUT}" \
  --device cuda:0 \
  --cfg-options data.val.ann_file="${PKL}"

for mod in j b jm bm; do
  ST_CKPT=$(best_ckpt "work_dirs/stgcnpp/cv/4class_conf04_${mod}/fold${FOLD}/best_*.pth")
  if [[ -z "${ST_CKPT}" ]]; then
    echo "ERROR: No STGCN++ ${mod} checkpoint"
    exit 1
  fi
  SOUT="${SCORE_DIR}/stgcnpp_${mod}/fold${FOLD}"
  mkdir -p "${SOUT}"
  echo "[STGCN++ ${mod}] ${ST_CKPT} -> ${SOUT}"
  python tools/evaluate_sails.py "configs/stgcn++/stgcnpp_sails_ntu60p/${mod}.py" \
    -C "${ST_CKPT}" \
    --split val \
    --output-dir "${SOUT}" \
    --device cuda:0 \
    --cfg-options data.val.ann_file="${PKL}"
done

cd ${REPO_ROOT}
FUSED="${SCORE_DIR}/stgcnpp_fused_4stream/fold${FOLD}/predictions_clip.csv"
mkdir -p "$(dirname "${FUSED}")"
echo "[7/7] Fuse STGCN++ 4-stream -> ${FUSED}"
conda activate vjepa2
python two-stg/fuse_stgcn_4stream.py \
  --j-csv "${SCORE_DIR}/stgcnpp_j/fold${FOLD}/predictions_clip.csv" \
  --b-csv "${SCORE_DIR}/stgcnpp_b/fold${FOLD}/predictions_clip.csv" \
  --jm-csv "${SCORE_DIR}/stgcnpp_jm/fold${FOLD}/predictions_clip.csv" \
  --bm-csv "${SCORE_DIR}/stgcnpp_bm/fold${FOLD}/predictions_clip.csv" \
  --output "${FUSED}"

echo "Done fold ${FOLD}. PoseC3D: ${POUT}/predictions_clip.csv  STGCN fused: ${FUSED}  $(date)"
