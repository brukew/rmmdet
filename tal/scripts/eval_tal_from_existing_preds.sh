#!/bin/bash
#SBATCH --job-name=eval_tal_csv
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/tal_eval/eval_tal_from_existing_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/tal_eval/eval_tal_from_existing_%j.err

# ============================================================================
# TAL Evaluation from Existing Prediction CSVs
#
# Evaluates all TAL models using existing predictions_clip.csv files:
# - PoseC3D (CE and Focal loss)
# - STGCN++ 4-stream (CE and Focal loss)
# - V-JEPA
#
# No inference required - uses CSV outputs from training runs.
# ============================================================================

set -eo pipefail

echo "=========================================="
echo "TAL Evaluation from Existing CSVs"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Ensure logs directory exists
mkdir -p /orcd/data/satra/001/users/brukew/pyskl_logs/tal_eval

export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

# Paths
TAL_DIR="/orcd/data/satra/001/users/brukew/actreg/tal"
PYSKL_WORK="/orcd/data/satra/001/users/brukew/actreg/pyskl/work_dirs"
VJEPA_RUNS="/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs"
OUT_DIR="${TAL_DIR}/eval_results"
SPLITS_ROOT="/orcd/data/satra/001/users/brukew/actreg/dataprep/splits"
WINDOW_CSV_DIR="/orcd/data/satra/001/users/brukew/actreg/dataprep/tal/splits_cv_4class"

cd "${TAL_DIR}"

# ============================================================================
# Helper function to evaluate a single model
# ============================================================================
evaluate_model() {
    local MODEL_NAME=$1
    local INPUT_CSV=$2
    local OUT_SUBDIR=$3
    local FOLD=$4
    local IS_VJEPA=${5:-false}  # V-JEPA doesn't need window CSV mapping
    
    local FOLD_OUT="${OUT_DIR}/${OUT_SUBDIR}/fold${FOLD}"
    mkdir -p "${FOLD_OUT}"
    
    # Get window CSV for video_key mapping (pyskl format needs this)
    local WINDOW_CSV="${WINDOW_CSV_DIR}/fold_${FOLD}_val_windows.csv"
    
    # Convert to TAL format
    local TAL_CSV="${FOLD_OUT}/tal_format_preds.csv"
    if [ ! -f "${TAL_CSV}" ]; then
        echo "  Converting to TAL format..."
        if [ "${IS_VJEPA}" = true ]; then
            # V-JEPA already has correct video_key format
            python convert_preds_to_tal_format.py \
                -i "${INPUT_CSV}" \
                -o "${TAL_CSV}"
        else
            # pyskl needs window CSV for video_key mapping
            python convert_preds_to_tal_format.py \
                -i "${INPUT_CSV}" \
                -o "${TAL_CSV}" \
                --window-csv "${WINDOW_CSV}"
        fi
    else
        echo "  TAL format CSV exists, skipping conversion"
    fi
    
    # Run TAL evaluation
    if [ ! -f "${FOLD_OUT}/metrics.json" ]; then
        echo "  Running TAL evaluation..."
        python eval_tal_from_window_preds.py \
            --window-preds "${TAL_CSV}" \
            --splits-root "${SPLITS_ROOT}" \
            --task 4class \
            --out-dir "${FOLD_OUT}" \
            --smooth-k 3 \
            --thr 0.5 \
            --merge-gap-sec 1.0
    else
        echo "  metrics.json exists, skipping evaluation"
    fi
}

# ============================================================================
# Helper function to evaluate STGCN++ 4-stream fusion
# ============================================================================
evaluate_stgcn_4stream() {
    local MODEL_NAME=$1
    local BASE_PATH=$2
    local OUT_SUBDIR=$3
    local FOLD=$4
    
    local FOLD_OUT="${OUT_DIR}/${OUT_SUBDIR}/fold${FOLD}"
    mkdir -p "${FOLD_OUT}"
    
    # Check if all modalities exist
    local MODALITIES="j b jm bm"
    local MODALITY_CSVS=""
    local ALL_EXIST=true
    
    for MOD in $MODALITIES; do
        local MOD_CSV="${BASE_PATH}/${MOD}/fold${FOLD}/eval_val/predictions_clip.csv"
        if [ -f "${MOD_CSV}" ]; then
            MODALITY_CSVS="${MODALITY_CSVS} ${MOD_CSV}"
        else
            echo "  Warning: Missing ${MOD} CSV for fold ${FOLD}: ${MOD_CSV}"
            ALL_EXIST=false
        fi
    done
    
    if [ "$ALL_EXIST" = false ]; then
        echo "  Skipping ${MODEL_NAME} fold ${FOLD} - missing modality files"
        return
    fi
    
    # Get window CSV for video_key mapping
    local WINDOW_CSV="${WINDOW_CSV_DIR}/fold_${FOLD}_val_windows.csv"
    
    # Convert and fuse to TAL format
    local FUSED_CSV="${FOLD_OUT}/fused_tal_format_preds.csv"
    if [ ! -f "${FUSED_CSV}" ]; then
        echo "  Converting and fusing 4 modalities..."
        python convert_preds_to_tal_format.py \
            -i ${MODALITY_CSVS} \
            -o "${FUSED_CSV}" \
            --fuse \
            --window-csv "${WINDOW_CSV}"
    else
        echo "  Fused TAL format CSV exists, skipping conversion"
    fi
    
    # Run TAL evaluation
    if [ ! -f "${FOLD_OUT}/metrics.json" ]; then
        echo "  Running TAL evaluation on fused predictions..."
        python eval_tal_from_window_preds.py \
            --window-preds "${FUSED_CSV}" \
            --splits-root "${SPLITS_ROOT}" \
            --task 4class \
            --out-dir "${FOLD_OUT}" \
            --smooth-k 3 \
            --thr 0.5 \
            --merge-gap-sec 1.0
    else
        echo "  metrics.json exists, skipping evaluation"
    fi
}

# ============================================================================
# PoseC3D CE
# ============================================================================
echo ""
echo "======================================================"
echo "EVALUATING: PoseC3D (CE Loss)"
echo "======================================================"

for FOLD in 0 1; do
    INPUT_CSV="${PYSKL_WORK}/posec3d/tal/cv_4class_5class_bgsub/fold${FOLD}/eval_val/predictions_clip.csv"
    if [ -f "${INPUT_CSV}" ]; then
        echo ""
        echo "--- PoseC3D CE Fold ${FOLD} ---"
        evaluate_model "PoseC3D-CE" "${INPUT_CSV}" "posec3d_ce" "${FOLD}"
    else
        echo "--- PoseC3D CE Fold ${FOLD}: CSV not found, skipping ---"
    fi
done

# ============================================================================
# PoseC3D Focal
# ============================================================================
echo ""
echo "======================================================"
echo "EVALUATING: PoseC3D (Focal Loss)"
echo "======================================================"

for FOLD in 0 1 2; do
    INPUT_CSV="${PYSKL_WORK}/posec3d/tal/cv_4class_5class_focal/fold${FOLD}/eval_val/predictions_clip.csv"
    if [ -f "${INPUT_CSV}" ]; then
        echo ""
        echo "--- PoseC3D Focal Fold ${FOLD} ---"
        evaluate_model "PoseC3D-Focal" "${INPUT_CSV}" "posec3d_focal" "${FOLD}"
    else
        echo "--- PoseC3D Focal Fold ${FOLD}: CSV not found, skipping ---"
    fi
done

# ============================================================================
# STGCN++ CE 4-Stream
# ============================================================================
echo ""
echo "======================================================"
echo "EVALUATING: STGCN++ 4-Stream (CE Loss)"
echo "======================================================"

for FOLD in 0 1; do
    echo ""
    echo "--- STGCN++ CE 4-Stream Fold ${FOLD} ---"
    evaluate_stgcn_4stream \
        "STGCN++-CE" \
        "${PYSKL_WORK}/stgcnpp/tal/cv_4class_5class_bgsub" \
        "stgcnpp_ce_4stream" \
        "${FOLD}"
done

# ============================================================================
# STGCN++ Focal 4-Stream
# ============================================================================
echo ""
echo "======================================================"
echo "EVALUATING: STGCN++ 4-Stream (Focal Loss)"
echo "======================================================"

for FOLD in 0 1 2; do
    echo ""
    echo "--- STGCN++ Focal 4-Stream Fold ${FOLD} ---"
    evaluate_stgcn_4stream \
        "STGCN++-Focal" \
        "${PYSKL_WORK}/stgcnpp/tal/cv_4class_5class_focal" \
        "stgcnpp_focal_4stream" \
        "${FOLD}"
done

# ============================================================================
# V-JEPA
# ============================================================================
echo ""
echo "======================================================"
echo "EVALUATING: V-JEPA"
echo "======================================================"

for FOLD in 0 1; do
    INPUT_CSV="${VJEPA_RUNS}/vjepa2_tal_cv_5class_bgsub/fold_${FOLD}/window_level_preds.csv"
    if [ -f "${INPUT_CSV}" ]; then
        echo ""
        echo "--- V-JEPA Fold ${FOLD} ---"
        evaluate_model "V-JEPA" "${INPUT_CSV}" "vjepa" "${FOLD}" true
    else
        echo "--- V-JEPA Fold ${FOLD}: CSV not found, skipping ---"
    fi
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "======================================================"
echo "Computing Summary"
echo "======================================================"

python -c "
import json
from pathlib import Path
import numpy as np

out_dir = Path('${OUT_DIR}')
summary = {}

# Models to aggregate
models = [
    ('posec3d_ce', 'PoseC3D (CE)', [0, 1]),
    ('posec3d_focal', 'PoseC3D (Focal)', [0, 1, 2]),
    ('stgcnpp_ce_4stream', 'STGCN++ 4-Stream (CE)', [0, 1]),
    ('stgcnpp_focal_4stream', 'STGCN++ 4-Stream (Focal)', [0, 1, 2]),
    ('vjepa', 'V-JEPA', [0, 1]),
]

print('\\n' + '='*60)
print('TAL Evaluation Summary (mAP @ tIoU thresholds)')
print('='*60)

for model_dir, model_name, folds in models:
    model_path = out_dir / model_dir
    fold_metrics = []
    
    for fold in folds:
        metrics_file = model_path / f'fold{fold}' / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                m = json.load(f)
                m['fold'] = fold
                fold_metrics.append(m)
    
    if fold_metrics:
        # Compute mean and std across folds
        model_summary = {'model': model_name, 'n_folds': len(fold_metrics)}
        
        for key in ['mAP@0.3', 'mAP@0.5', 'mAP@0.7', 'avg_mAP']:
            vals = [f.get(key, 0) for f in fold_metrics if key in f]
            if vals:
                model_summary[f'{key}_mean'] = float(np.mean(vals))
                model_summary[f'{key}_std'] = float(np.std(vals))
        
        summary[model_dir] = model_summary
        
        map05 = model_summary.get('mAP@0.5_mean', 0)
        map05_std = model_summary.get('mAP@0.5_std', 0)
        avg_map = model_summary.get('avg_mAP_mean', 0)
        avg_map_std = model_summary.get('avg_mAP_std', 0)
        
        print(f'\\n{model_name} ({len(fold_metrics)} folds):')
        print(f'  mAP@0.5: {map05:.3f} +/- {map05_std:.3f}')
        print(f'  avg_mAP: {avg_map:.3f} +/- {avg_map_std:.3f}')
    else:
        print(f'\\n{model_name}: No results found')

# Save summary
summary_file = out_dir / 'tal_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\\nSummary saved to: {summary_file}')
"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "Results saved to: ${OUT_DIR}"
echo "=========================================="

