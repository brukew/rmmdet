#!/usr/bin/env bash
# Full pipeline: TriDet proposals + skeleton inference + 3-way fusion.
# Mirrors run_three_way_live_fold_gpu.sh but uses TriDet detections.
#
# Usage:
#   cd actreg && sbatch --export=FOLD=0 two-stg/run_tridet_three_way_live_fold_gpu.sh

#SBATCH --job-name=r5c_3way
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/run5c_3way_live_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/run5c_3way_live_%j.err

set -eo pipefail

FOLD="${FOLD:?FOLD env var required (0, 1, or 2)}"

echo "Run 5c: TriDet + 3-way LIVE skeleton scores — Fold ${FOLD}"
echo "Job ID: ${SLURM_JOB_ID:-local}  Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-unset}  $(date)"

mkdir -p /orcd/data/satra/001/users/brukew/actreg/two-stg/logs
export PYTHONUNBUFFERED=1

if [[ -f "${HOME}/.bashrc" ]]; then
  source "${HOME}/.bashrc"
fi

cd /orcd/data/satra/001/users/brukew/actreg
source "${HOME}/miniconda3/etc/profile.d/conda.sh"

OPENTAD_EXPS="${PWD}/OpenTAD/exps/sails_rmm"
DETECTION_JSON="${OPENTAD_EXPS}/tridet_vjepa_binary_fold${FOLD}/gpu1_id99/result_detection.json"
VJEPA_CKPT="${PWD}/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls"
BUNDLE="${PWD}/two-stg/fusion_checkpoints/three_way/fold_${FOLD}"
OUTPUT_ROOT="${PWD}/two-stg/eval_results_tridet_3way_live"
OUT_DIR="${OUTPUT_ROOT}/fold${FOLD}"

PICKLE_DIR="${PWD}/two-stg/proposal_pickles_tridet"
SCORE_DIR="${PWD}/two-stg/proposal_scores_tridet"

if [[ ! -f "${DETECTION_JSON}" ]]; then
  echo "ERROR: Missing detection JSON: ${DETECTION_JSON}"
  exit 1
fi
if [[ ! -f "${BUNDLE}/mlp_state.pt" ]]; then
  echo "ERROR: Missing fusion bundle: ${BUNDLE}/mlp_state.pt"
  exit 1
fi

echo "Detection JSON: ${DETECTION_JSON}"
echo "Fusion bundle: ${BUNDLE}"
echo "Output: ${OUT_DIR}"

# --- Step 1: Build proposal pickles from TriDet detections ---
echo "[1/7] Build proposal pickle"
conda activate vjepa2
python two-stg/build_proposal_poses.py \
  --fold "${FOLD}" \
  --output-dir "${PICKLE_DIR}" \
  --detection-json "${DETECTION_JSON}"

PKL="${PICKLE_DIR}/proposals_fold${FOLD}.pkl"
if [[ ! -f "${PKL}" ]]; then
  echo "ERROR: Missing pickle ${PKL}"
  exit 1
fi

# --- Steps 2-7: Skeleton inference (PoseC3D + STGCN++ 4-stream) ---
conda activate pyskl
cd "${PWD}/pyskl"

best_ckpt() {
  local pattern="$1"
  ls ${pattern} 2>/dev/null | head -1
}

POSE_CKPT=$(best_ckpt "work_dirs/posec3d/cv/4class_conf04/fold${FOLD}/best_*.pth")
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
  SOUT="${SCORE_DIR}/stgcnpp_${mod}/fold${FOLD}"
  mkdir -p "${SOUT}"
  echo "[STGCN++ ${mod}] -> ${SOUT}"
  python tools/evaluate_sails.py "configs/stgcn++/stgcnpp_sails_ntu60p/${mod}.py" \
    -C "${ST_CKPT}" \
    --split val \
    --output-dir "${SOUT}" \
    --device cuda:0 \
    --cfg-options data.val.ann_file="${PKL}"
done

cd /orcd/data/satra/001/users/brukew/actreg
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

POSE_CSV="${SCORE_DIR}/posec3d/fold${FOLD}/predictions_clip.csv"

# --- Stage 2: 3-way fusion eval ---
echo "[Eval] V-JEPA2 + 3-way MLP + live proposal skeleton scores -> ${OUT_DIR}"
conda activate vjepa2
python two-stg/eval_two_stage_tal.py \
  --fold "$FOLD" \
  --detection-json "${DETECTION_JSON}" \
  --checkpoint-dir "${VJEPA_CKPT}/fold_${FOLD}" \
  --output-dir "$OUT_DIR" \
  --classifier-backend three_way \
  --fusion-bundle-dir "$BUNDLE" \
  --posec3d-proposal-scores "$POSE_CSV" \
  --stgcn-proposal-scores "$FUSED" \
  --device cuda \
  --batch-size 16 \
  --skip-opentad-eval

echo "[OpenTAD eval]"
conda activate opentad
python two-stg/opentad_eval.py \
  --fold "$FOLD" \
  --predictions-csv "${OUT_DIR}/predictions.csv" \
  --output-dir "$OUT_DIR"

echo "Run 5c fold ${FOLD} done. ${OUT_DIR}  $(date)"
