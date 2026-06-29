#!/usr/bin/env bash
# Run two-stage TAL evaluation across all 3 folds and aggregate CV results.
# Execute from actreg repo root so that two-stg/ and OpenTAD/ paths resolve.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTREG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ACTREG_ROOT"

source ~/miniconda3/etc/profile.d/conda.sh

# CLASSIFIER_BACKEND=vjepa2 | three_way  (default vjepa2)
# OUTPUT_ROOT optional; three_way defaults to two-stg/eval_results_3way
CLASSIFIER_BACKEND="${CLASSIFIER_BACKEND:-vjepa2}"

OPENTAD_EXPS="${ACTREG_ROOT}/OpenTAD/exps/sails_rmm"
VJEPA_CKPT="${ACTREG_ROOT}/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls"
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ "$CLASSIFIER_BACKEND" == "three_way" ]]; then
    OUTPUT_ROOT="${ACTREG_ROOT}/two-stg/eval_results_3way"
  else
    OUTPUT_ROOT="${ACTREG_ROOT}/two-stg/eval_results"
  fi
fi

for fold in 0 1 2; do
  DETECTION_JSON="${OPENTAD_EXPS}/actionformer_vjepa_binary_fold${fold}/gpu1_id99/result_detection.json"
  CHECKPOINT_DIR="${VJEPA_CKPT}/fold_${fold}"
  OUT_DIR="${OUTPUT_ROOT}/fold${fold}"

  if [[ ! -f "$DETECTION_JSON" ]]; then
    echo "Missing detection JSON: $DETECTION_JSON"
    exit 1
  fi
  if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Missing checkpoint: $CHECKPOINT_DIR"
    exit 1
  fi
  if [[ "$CLASSIFIER_BACKEND" == "three_way" ]]; then
    BUNDLE="${FUSION_BUNDLE_DIR:-${ACTREG_ROOT}/two-stg/fusion_checkpoints/three_way/fold_${fold}}"
    if [[ ! -f "${BUNDLE}/mlp_state.pt" ]]; then
      echo "Missing 3-way fusion bundle: ${BUNDLE}/mlp_state.pt"
      echo "Run: python fusion/export_three_way_deploy_checkpoint.py --all-folds"
      exit 1
    fi
  fi

  echo "=== Fold $fold ==="
  echo "[1/2] Classifying proposals (backend=${CLASSIFIER_BACKEND})"
  conda activate vjepa2
  EVAL_ARGS=(
    --fold "$fold"
    --detection-json "$DETECTION_JSON"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --output-dir "$OUT_DIR"
    --skip-opentad-eval
  )
  if [[ "$CLASSIFIER_BACKEND" == "three_way" ]]; then
    EVAL_ARGS+=(--classifier-backend three_way)
    if [[ -n "${FUSION_BUNDLE_DIR:-}" ]]; then
      EVAL_ARGS+=(--fusion-bundle-dir "${FUSION_BUNDLE_DIR}")
    fi
  fi
  python two-stg/eval_two_stage_tal.py "${EVAL_ARGS[@]}"

  echo "[2/2] Evaluating predictions with OpenTAD env"
  conda activate opentad
  python two-stg/opentad_eval.py \
    --fold "$fold" \
    --predictions-csv "${OUT_DIR}/predictions.csv" \
    --output-dir "$OUT_DIR"
done

echo "=== Aggregating CV results ==="
conda activate opentad
python two-stg/aggregate_cv_results.py --input-dir "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/cv_summary.json"

echo "Done. CV summary: $OUTPUT_ROOT/cv_summary.json"
