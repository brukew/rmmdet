#!/bin/bash
#SBATCH --job-name=eval_tal_ce
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/tal_eval/eval_tal_ce_models_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/tal_eval/eval_tal_ce_models_%j.err

# ============================================================================
# TAL Evaluation: CE-Loss Models (PoseC3D + STGCN++ 4-stream)
#
# Evaluates previously trained TAL models using mAP@tIoU {0.3, 0.5, 0.7}
# ============================================================================

set -eo pipefail

echo "=========================================="
echo "TAL Evaluation: CE-Loss Models"
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
PYSKL_DIR="/orcd/data/satra/001/users/brukew/actreg/pyskl"
TAL_DIR="/orcd/data/satra/001/users/brukew/actreg/tal"
DATAPREP_DIR="/orcd/data/satra/001/users/brukew/actreg/dataprep/tal"
ANN_DIR="${PYSKL_DIR}/data/sails/tal/cv_4class/5class_windows_conf04"
OUT_DIR="${TAL_DIR}/eval_results/ce_models"

# Model directories
POSEC3D_WORK="${PYSKL_DIR}/work_dirs/posec3d/tal/cv_4class_5class_bgsub"
STGCNPP_WORK="${PYSKL_DIR}/work_dirs/stgcnpp/tal/cv_4class_5class_bgsub"

# Configs
POSEC3D_CONFIG="${PYSKL_DIR}/configs/posec3d/slowonly_r50_sails_k400p/joint_tal_5class.py"
STGCNPP_J_CONFIG="${PYSKL_DIR}/configs/stgcn++/stgcnpp_sails_ntu60p/j_tal_5class.py"
STGCNPP_B_CONFIG="${PYSKL_DIR}/configs/stgcn++/stgcnpp_sails_ntu60p/b_tal_5class.py"
STGCNPP_JM_CONFIG="${PYSKL_DIR}/configs/stgcn++/stgcnpp_sails_ntu60p/jm_tal_5class.py"
STGCNPP_BM_CONFIG="${PYSKL_DIR}/configs/stgcn++/stgcnpp_sails_ntu60p/bm_tal_5class.py"

mkdir -p "${OUT_DIR}"

cd "${PYSKL_DIR}"

# ============================================================================
# PoseC3D Evaluation
# ============================================================================
echo ""
echo "======================================================"
echo "EVALUATING: PoseC3D"
echo "======================================================"

for FOLD in 0 1; do
    echo ""
    echo "--- PoseC3D Fold ${FOLD} ---"
    
    FOLD_WORK="${POSEC3D_WORK}/fold${FOLD}"
    FOLD_OUT="${OUT_DIR}/posec3d/fold${FOLD}"
    WINDOW_CSV="${DATAPREP_DIR}/splits_cv_4class/fold_${FOLD}_val_windows.csv"
    ANN_PKL="${ANN_DIR}/fold${FOLD}.pkl"
    
    mkdir -p "${FOLD_OUT}"
    
    # Find best checkpoint
    BEST_CKPT=$(ls -t ${FOLD_WORK}/best_*.pth 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT="${FOLD_WORK}/latest.pth"
    fi
    echo "Checkpoint: $BEST_CKPT"
    
    # 1. Run test.py to get result.pkl
    if [ ! -f "${FOLD_OUT}/result.pkl" ]; then
        echo "Running test.py..."
        python tools/test.py "${POSEC3D_CONFIG}" \
            -C "${BEST_CKPT}" \
            --out "${FOLD_OUT}/result.pkl" \
            --launcher none \
            --cfg-options \
                data.test.ann_file="${ANN_PKL}" \
                data.val.ann_file="${ANN_PKL}"
    else
        echo "result.pkl already exists, skipping test.py"
    fi
    
    # 2. Export to window_preds.csv
    if [ ! -f "${FOLD_OUT}/window_preds.csv" ]; then
        echo "Exporting to window_preds.csv..."
        cd "${TAL_DIR}"
        python export_pyskl_window_preds.py \
            --ann-pkl "${ANN_PKL}" \
            --scores-pkl "${FOLD_OUT}/result.pkl" \
            --window-csv "${WINDOW_CSV}" \
            --out-csv "${FOLD_OUT}/window_preds.csv"
        cd "${PYSKL_DIR}"
    else
        echo "window_preds.csv already exists, skipping export"
    fi
    
    # 3. Run TAL evaluation
    if [ ! -f "${FOLD_OUT}/metrics.json" ]; then
        echo "Running TAL evaluation..."
        cd "${TAL_DIR}"
        python eval_tal_from_window_preds.py \
            --window-preds "${FOLD_OUT}/window_preds.csv" \
            --splits-root "${DATAPREP_DIR}/../splits" \
            --task 4class \
            --out-dir "${FOLD_OUT}"
        cd "${PYSKL_DIR}"
    else
        echo "metrics.json already exists, skipping TAL eval"
    fi
    
    echo "PoseC3D Fold ${FOLD} done!"
done

# ============================================================================
# STGCN++ 4-Stream Evaluation
# ============================================================================
echo ""
echo "======================================================"
echo "EVALUATING: STGCN++ 4-Stream"
echo "======================================================"

for FOLD in 0 1; do
    echo ""
    echo "--- STGCN++ 4-Stream Fold ${FOLD} ---"
    
    FOLD_OUT="${OUT_DIR}/stgcnpp_4stream/fold${FOLD}"
    WINDOW_CSV="${DATAPREP_DIR}/splits_cv_4class/fold_${FOLD}_val_windows.csv"
    ANN_PKL="${ANN_DIR}/fold${FOLD}.pkl"
    
    mkdir -p "${FOLD_OUT}"
    
    # Run test.py for each modality
    for MODALITY in j b jm bm; do
        MOD_WORK="${STGCNPP_WORK}/${MODALITY}/fold${FOLD}"
        MOD_OUT="${FOLD_OUT}/${MODALITY}_result.pkl"
        
        case $MODALITY in
            j)  CONFIG="${STGCNPP_J_CONFIG}" ;;
            b)  CONFIG="${STGCNPP_B_CONFIG}" ;;
            jm) CONFIG="${STGCNPP_JM_CONFIG}" ;;
            bm) CONFIG="${STGCNPP_BM_CONFIG}" ;;
        esac
        
        # Find best checkpoint
        BEST_CKPT=$(ls -t ${MOD_WORK}/best_*.pth 2>/dev/null | head -1)
        if [ -z "$BEST_CKPT" ]; then
            BEST_CKPT="${MOD_WORK}/latest.pth"
        fi
        
        if [ ! -f "${MOD_OUT}" ]; then
            echo "Running test.py for ${MODALITY}..."
            echo "  Checkpoint: $BEST_CKPT"
            python tools/test.py "${CONFIG}" \
                -C "${BEST_CKPT}" \
                --out "${MOD_OUT}" \
                --launcher none \
                --cfg-options \
                    data.test.ann_file="${ANN_PKL}" \
                    data.val.ann_file="${ANN_PKL}"
        else
            echo "${MODALITY}_result.pkl already exists, skipping"
        fi
    done
    
    # Fuse scores and export to window_preds.csv
    if [ ! -f "${FOLD_OUT}/window_preds.csv" ]; then
        echo "Fusing 4-stream scores and exporting..."
        cd "${TAL_DIR}"
        python -c "
import pickle
import numpy as np
from pathlib import Path

fold_out = Path('${FOLD_OUT}')
modalities = ['j', 'b', 'jm', 'bm']

# Load all modality scores
all_scores = []
for mod in modalities:
    pkl_path = fold_out / f'{mod}_result.pkl'
    with open(pkl_path, 'rb') as f:
        scores = pickle.load(f)
    if isinstance(scores, list):
        scores = np.array(scores)
    all_scores.append(scores)
    print(f'  {mod}: {scores.shape}')

# Average (4-stream fusion)
fused = np.mean(all_scores, axis=0)
print(f'  Fused: {fused.shape}')

# Save fused result
with open(fold_out / 'fused_result.pkl', 'wb') as f:
    pickle.dump(fused, f)
print('  Saved fused_result.pkl')
"
        
        # Export fused to window_preds.csv
        python export_pyskl_window_preds.py \
            --ann-pkl "${ANN_PKL}" \
            --scores-pkl "${FOLD_OUT}/fused_result.pkl" \
            --window-csv "${WINDOW_CSV}" \
            --out-csv "${FOLD_OUT}/window_preds.csv"
        cd "${PYSKL_DIR}"
    else
        echo "window_preds.csv already exists, skipping fusion"
    fi
    
    # Run TAL evaluation
    if [ ! -f "${FOLD_OUT}/metrics.json" ]; then
        echo "Running TAL evaluation..."
        cd "${TAL_DIR}"
        python eval_tal_from_window_preds.py \
            --window-preds "${FOLD_OUT}/window_preds.csv" \
            --splits-root "${DATAPREP_DIR}/../splits" \
            --task 4class \
            --out-dir "${FOLD_OUT}"
        cd "${PYSKL_DIR}"
    else
        echo "metrics.json already exists, skipping TAL eval"
    fi
    
    echo "STGCN++ 4-Stream Fold ${FOLD} done!"
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "======================================================"
echo "Computing Summary"
echo "======================================================"

cd "${TAL_DIR}"
python -c "
import json
from pathlib import Path
import numpy as np

out_dir = Path('${OUT_DIR}')
summary = {}

# Collect metrics for each model
for model in ['posec3d', 'stgcnpp_4stream']:
    model_dir = out_dir / model
    fold_metrics = []
    
    for fold in [0, 1]:
        metrics_file = model_dir / f'fold{fold}' / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                m = json.load(f)
                m['fold'] = fold
                fold_metrics.append(m)
                print(f'{model} fold{fold}: mAP@0.5 = {m.get(\"mAP@0.5\", 0):.3f}')
    
    if fold_metrics:
        # Compute mean and std
        model_summary = {'per_fold': fold_metrics}
        for key in ['mAP@0.3', 'mAP@0.5', 'mAP@0.7', 'avg_mAP']:
            vals = [f.get(key, 0) for f in fold_metrics if key in f]
            if vals:
                model_summary[f'{key}_mean'] = float(np.mean(vals))
                model_summary[f'{key}_std'] = float(np.std(vals))
        summary[model] = model_summary

# Save summary
with open(out_dir / 'ce_models_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print()
print('=== CE Models TAL Summary ===')
for model, data in summary.items():
    map05 = data.get('mAP@0.5_mean', 0)
    map05_std = data.get('mAP@0.5_std', 0)
    avg_map = data.get('avg_mAP_mean', 0)
    print(f'{model}: mAP@0.5 = {map05:.3f} +/- {map05_std:.3f}, avg_mAP = {avg_map:.3f}')
"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "Results saved to: ${OUT_DIR}"
echo "=========================================="

