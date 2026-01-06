#!/bin/bash
#SBATCH --job-name=stgcn_focal_cont
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/tal/stgcnpp_tal_focal_continue_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/tal/stgcnpp_tal_focal_continue_%j.err

# ============================================================================
# Continue STGCN++ Focal Training - Remaining modalities/folds
#
# Trains: jm fold 2, bm folds 0, 1, 2
# Also runs evaluation on completed folds that didn't get evaluated
# ============================================================================

set -eo pipefail

echo "=========================================="
echo "STGCN++ Focal Training - Continue"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

mkdir -p /orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/tal

export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

# Configuration
CONFIG_DIR="configs/stgcn++/stgcnpp_sails_ntu60p"
BASE_ANN_DIR="data/sails/tal/cv_4class/5class_windows_conf04"
BASE_WORK_DIR="work_dirs/stgcnpp/tal/cv_4class_5class_focal"
EPOCHS=24
BG_SUBSAMPLE=0.2

echo ""
echo "Remaining training:"
echo "  jm: fold 2"
echo "  bm: folds 0, 1, 2"
echo ""

# Function to train a modality/fold
train_fold() {
    local MODALITY=$1
    local FOLD=$2
    
    local CONFIG="${CONFIG_DIR}/${MODALITY}_tal_5class_focal.py"
    local ANN_FILE="${BASE_ANN_DIR}/fold${FOLD}.pkl"
    local WORK_DIR="${BASE_WORK_DIR}/${MODALITY}/fold${FOLD}"
    
    echo ""
    echo "------------------------------------------------------"
    echo "TRAINING: $MODALITY fold $FOLD"
    echo "------------------------------------------------------"
    
    # Check if already done
    local PRED_CSV="${WORK_DIR}/eval_val/predictions_clip.csv"
    if [ -f "$PRED_CSV" ]; then
        echo "Already complete (predictions exist), skipping"
        return
    fi
    
    # Train
    python tools/train_weighted.py $CONFIG \
        --ann-file $ANN_FILE \
        --work-dir $WORK_DIR \
        --total-epochs $EPOCHS \
        --bg-subsample $BG_SUBSAMPLE \
        --validate \
        --launcher none
    
    # Evaluate
    BEST_CKPT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT="${WORK_DIR}/latest.pth"
    fi
    
    echo "Evaluating: $BEST_CKPT"
    python tools/evaluate_sails.py $CONFIG \
        -C $BEST_CKPT \
        --split val \
        --output-dir ${WORK_DIR}/eval_val \
        --task tal \
        --cfg-options \
            data.train.dataset.ann_file=$ANN_FILE \
            data.val.ann_file=$ANN_FILE \
            model.cls_head.num_classes=5
}

# Function to just evaluate (for folds that trained but didn't eval)
eval_fold() {
    local MODALITY=$1
    local FOLD=$2
    
    local CONFIG="${CONFIG_DIR}/${MODALITY}_tal_5class_focal.py"
    local ANN_FILE="${BASE_ANN_DIR}/fold${FOLD}.pkl"
    local WORK_DIR="${BASE_WORK_DIR}/${MODALITY}/fold${FOLD}"
    
    local PRED_CSV="${WORK_DIR}/eval_val/predictions_clip.csv"
    if [ -f "$PRED_CSV" ]; then
        echo "$MODALITY fold $FOLD: Already evaluated, skipping"
        return
    fi
    
    BEST_CKPT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        echo "$MODALITY fold $FOLD: No checkpoint, skipping"
        return
    fi
    
    echo ""
    echo "------------------------------------------------------"
    echo "EVALUATING: $MODALITY fold $FOLD"
    echo "------------------------------------------------------"
    
    python tools/evaluate_sails.py $CONFIG \
        -C $BEST_CKPT \
        --split val \
        --output-dir ${WORK_DIR}/eval_val \
        --task tal \
        --cfg-options \
            data.train.dataset.ann_file=$ANN_FILE \
            data.val.ann_file=$ANN_FILE \
            model.cls_head.num_classes=5
}

# First, evaluate any folds that trained but didn't get evaluated
echo ""
echo "======================================================"
echo "Phase 1: Evaluate completed folds missing evaluation"
echo "======================================================"

eval_fold jm 0
eval_fold jm 1

# Then train remaining folds
echo ""
echo "======================================================"
echo "Phase 2: Train remaining folds"
echo "======================================================"

train_fold jm 2
train_fold bm 0
train_fold bm 1
train_fold bm 2

# ============================================================================
# Phase 3: 4-Stream Fusion Evaluation
# ============================================================================
echo ""
echo "======================================================"
echo "Phase 3: 4-Stream Fusion Evaluation"
echo "======================================================"

for FOLD in 0 1 2; do
    echo ""
    echo "------------------------------------------------------"
    echo "4-Stream Fusion: Fold $FOLD"
    echo "------------------------------------------------------"
    
    # Check all modalities have predictions
    ALL_EXIST=true
    for MOD in j b jm bm; do
        PRED_CSV="${BASE_WORK_DIR}/${MOD}/fold${FOLD}/eval_val/predictions_clip.csv"
        if [ ! -f "$PRED_CSV" ]; then
            echo "Missing: $MOD fold $FOLD"
            ALL_EXIST=false
        fi
    done
    
    if [ "$ALL_EXIST" = false ]; then
        echo "Skipping fold $FOLD - missing modality predictions"
        continue
    fi
    
    python -c "
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score, roc_auc_score

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

FOLD = ${FOLD}
BASE_WORK_DIR = '${BASE_WORK_DIR}'
CLASS_NAMES = ['hands_flapping', 'jumping', 'rocking', 'spinning', 'background']
BACKGROUND_CLASS = 4

def top_k_accuracy(scores, labels, topk=(1, 2)):
    results = {}
    maxk = max(topk)
    top_k_preds = np.argsort(scores, axis=1)[:, ::-1][:, :maxk]
    for k in topk:
        correct = sum(1 for i, label in enumerate(labels) if label in top_k_preds[i, :k])
        results[f'top{k}_acc'] = correct / len(labels) * 100
    return results

def compute_all_metrics(y_true, y_pred, y_scores, prefix=''):
    \"\"\"Compute metrics for TAL task (5-class with background).\"\"\"
    metrics = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Top-k accuracy
    topk = top_k_accuracy(y_scores, y_true, topk=(1, 2))
    metrics[f'{prefix}top1_acc'] = topk['top1_acc']
    metrics[f'{prefix}top2_acc'] = topk['top2_acc']
    
    # All-class metrics
    metrics[f'{prefix}macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics[f'{prefix}weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    metrics[f'{prefix}macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics[f'{prefix}macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics[f'{prefix}cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # RMM vs Background binary metrics
    y_true_binary = (y_true != BACKGROUND_CLASS).astype(int)
    y_pred_binary = (y_pred != BACKGROUND_CLASS).astype(int)
    rmm_scores = y_scores[:, :BACKGROUND_CLASS].sum(axis=1)
    
    metrics[f'{prefix}rmm_vs_bg_accuracy'] = (y_true_binary == y_pred_binary).mean() * 100
    metrics[f'{prefix}rmm_vs_bg_f1'] = f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0) * 100
    metrics[f'{prefix}rmm_vs_bg_precision'] = precision_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0) * 100
    metrics[f'{prefix}rmm_vs_bg_recall'] = recall_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0) * 100
    
    try:
        if len(np.unique(y_true_binary)) > 1:
            metrics[f'{prefix}rmm_vs_bg_auc'] = roc_auc_score(y_true_binary, rmm_scores) * 100
        else:
            metrics[f'{prefix}rmm_vs_bg_auc'] = 0.0
    except:
        metrics[f'{prefix}rmm_vs_bg_auc'] = 0.0
    
    # Error rates
    bg_mask = y_true_binary == 0
    if bg_mask.sum() > 0:
        metrics[f'{prefix}bg_false_alarm_rate'] = (y_pred_binary[bg_mask] == 1).mean() * 100
    else:
        metrics[f'{prefix}bg_false_alarm_rate'] = 0.0
    
    rmm_mask = y_true_binary == 1
    if rmm_mask.sum() > 0:
        metrics[f'{prefix}rmm_miss_rate'] = (y_pred_binary[rmm_mask] == 0).mean() * 100
    else:
        metrics[f'{prefix}rmm_miss_rate'] = 0.0
    
    # RMM-only metrics (excluding background)
    rmm_indices = list(range(BACKGROUND_CLASS))
    rmm_mask_multiclass = np.isin(y_true, rmm_indices)
    if rmm_mask_multiclass.sum() > 0:
        y_true_rmm = y_true[rmm_mask_multiclass]
        y_pred_rmm = y_pred[rmm_mask_multiclass]
        metrics[f'{prefix}rmm_only_macro_f1'] = f1_score(y_true_rmm, y_pred_rmm, average='macro', zero_division=0) * 100
        metrics[f'{prefix}rmm_only_macro_precision'] = precision_score(y_true_rmm, y_pred_rmm, average='macro', zero_division=0) * 100
        metrics[f'{prefix}rmm_only_macro_recall'] = recall_score(y_true_rmm, y_pred_rmm, average='macro', zero_division=0) * 100
    else:
        metrics[f'{prefix}rmm_only_macro_f1'] = 0.0
        metrics[f'{prefix}rmm_only_macro_precision'] = 0.0
        metrics[f'{prefix}rmm_only_macro_recall'] = 0.0
    
    # Per-class metrics (RMM only, skip background)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, name in enumerate(CLASS_NAMES):
        if 'background' in name.lower():
            continue
        if i < len(per_class_f1):
            metrics[f'{prefix}f1_{name}'] = per_class_f1[i] * 100
            metrics[f'{prefix}prec_{name}'] = per_class_prec[i] * 100
            metrics[f'{prefix}rec_{name}'] = per_class_rec[i] * 100
    
    return metrics

def plot_confusion_matrix(cm, output_path, title='Confusion Matrix'):
    if not HAS_PLOT:
        return
    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, square=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# Load predictions from all modalities
modalities = ['j', 'b', 'jm', 'bm']
weights = [2, 2, 1, 1]
all_probs = []
labels = None
segment_ids = None

for mod in modalities:
    pred_csv = Path(BASE_WORK_DIR) / mod / f'fold{FOLD}' / 'eval_val' / 'predictions_clip.csv'
    df = pd.read_csv(pred_csv)
    
    score_cols = [c for c in df.columns if c.startswith('score_class')]
    scores = df[score_cols].values
    
    # Softmax to get probabilities
    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    all_probs.append(probs)
    
    if labels is None:
        labels = df['true_label'].values
        segment_ids = df['segment_id'].values

# Weighted fusion
fused_probs = sum(w * p for w, p in zip(weights, all_probs)) / sum(weights)
fused_preds = fused_probs.argmax(axis=1)

# Save fused predictions
output_dir = Path(BASE_WORK_DIR) / '4stream' / f'fold{FOLD}'
output_dir.mkdir(parents=True, exist_ok=True)

fused_df = pd.DataFrame({
    'segment_id': segment_ids,
    'true_label': labels,
    'pred_label': fused_preds,
})
for i in range(fused_probs.shape[1]):
    fused_df[f'score_class{i}'] = fused_probs[:, i]
fused_df.to_csv(output_dir / 'predictions_clip.csv', index=False)

# Compute metrics (clip-level only)
metrics = compute_all_metrics(labels, fused_preds, fused_probs, prefix='clip_')
metrics['num_clips'] = len(labels)
metrics['task'] = 'tal'

# Confusion matrix
cm_clip = confusion_matrix(labels, fused_preds)
plot_confusion_matrix(cm_clip, output_dir / 'confusion_matrix_clip.png', f'Fold {FOLD} - 4-Stream')
np.save(output_dir / 'confusion_matrix_clip.npy', cm_clip)

# Save metrics
with open(output_dir / 'metrics.json', 'w') as f:
    json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()}, f, indent=2)

print(f'Fold {FOLD}: top1={metrics[\"clip_top1_acc\"]:.2f}%, rmm_vs_bg_f1={metrics[\"clip_rmm_vs_bg_f1\"]:.2f}%, rmm_only_f1={metrics[\"clip_rmm_only_macro_f1\"]:.2f}%')
"
done

# ============================================================================
# Phase 4: CV Summary
# ============================================================================
echo ""
echo "======================================================"
echo "Phase 4: CV Summary"
echo "======================================================"

python -c "
import json
import numpy as np
from pathlib import Path

base_dir = Path('${BASE_WORK_DIR}/4stream')
folds = []

for fold in range(3):
    metrics_file = base_dir / f'fold{fold}' / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
            m['fold'] = fold
            folds.append(m)
            print(f'Fold {fold}: top1={m.get(\"clip_top1_acc\", 0):.2f}%, rmm_vs_bg_f1={m.get(\"clip_rmm_vs_bg_f1\", 0):.2f}%')

if folds:
    summary = {'per_fold': folds, 'model': 'STGCN++ 4-Stream', 'loss': 'focal', 'task': 'tal'}
    metric_keys = [k for k in folds[0].keys() if k not in ('fold', 'task')]
    for key in metric_keys:
        vals = [f[key] for f in folds if key in f]
        if vals and isinstance(vals[0], (int, float)):
            summary[f'{key}_mean'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))
    
    with open(base_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print('CV Summary (STGCN++ 4-Stream Focal):')
    print(f'  Top-1: {summary.get(\"clip_top1_acc_mean\", 0):.2f} +/- {summary.get(\"clip_top1_acc_std\", 0):.2f}%')
    print(f'  Macro-F1: {summary.get(\"clip_macro_f1_mean\", 0):.2f}%  Macro-Prec: {summary.get(\"clip_macro_precision_mean\", 0):.2f}%  Macro-Rec: {summary.get(\"clip_macro_recall_mean\", 0):.2f}%')
    print(f'  RMM vs BG F1: {summary.get(\"clip_rmm_vs_bg_f1_mean\", 0):.2f}%  AUC: {summary.get(\"clip_rmm_vs_bg_auc_mean\", 0):.2f}%')
    print(f'  RMM-Only Macro-F1: {summary.get(\"clip_rmm_only_macro_f1_mean\", 0):.2f}%')
else:
    print('No fold metrics found')
"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

