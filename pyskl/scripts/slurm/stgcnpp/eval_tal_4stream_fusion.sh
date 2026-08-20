#!/bin/bash
#SBATCH --job-name=stgcnpp_tal_eval
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-logs/stgcnpp_tal_eval_%j.out
#SBATCH --error=slurm-logs/stgcnpp_tal_eval_%j.err

# ============================================================================
# STGCN++ TAL Evaluation: Per-modality + 4-Stream Fusion
# 
# Runs evaluation on already-trained models:
# - Per-modality evaluation with evaluate_sails.py
# - 4-stream fusion with weighted combination
# - Per-class metrics and confusion matrices
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

echo "=========================================="
echo "STGCN++ TAL Evaluation: Per-modality + 4-Stream Fusion"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

conda activate pyskl

cd ${REPO_ROOT}/pyskl

# Configuration
CONFIG_DIR="configs/stgcn++/stgcnpp_sails_ntu60p"
BASE_ANN_DIR="data/sails/tal/cv_4class/5class_windows_conf04"
BASE_WORK_DIR="work_dirs/stgcnpp/tal/cv_4class_5class_bgsub"

MODALITIES=("j" "b" "jm" "bm")
MODALITY_NAMES=("Joint" "Bone" "Joint Motion" "Bone Motion")
CLASS_NAMES='["hands flapping", "jumping", "rocking", "spinning", "background"]'

echo ""
echo "Base work dir: $BASE_WORK_DIR"
echo ""

# Process each fold
for FOLD in 0 1; do
    ANN_FILE="${BASE_ANN_DIR}/fold${FOLD}.pkl"
    
    echo ""
    echo "######################################################"
    echo "# FOLD ${FOLD}"
    echo "######################################################"
    
    # Evaluate each modality
    for i in "${!MODALITIES[@]}"; do
        MOD="${MODALITIES[$i]}"
        MOD_NAME="${MODALITY_NAMES[$i]}"
        CONFIG="${CONFIG_DIR}/${MOD}_tal_5class.py"
        WORK_DIR="${BASE_WORK_DIR}/${MOD}/fold${FOLD}"
        
        echo ""
        echo "  Evaluating ${MOD_NAME} (${MOD})..."
        
        BEST_CKPT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)
        if [ -z "$BEST_CKPT" ]; then
            BEST_CKPT="${WORK_DIR}/latest.pth"
        fi
        
        if [ -f "$BEST_CKPT" ]; then
            python tools/evaluate_sails.py $CONFIG \
                -C $BEST_CKPT \
                --split val \
                --output-dir ${WORK_DIR}/eval_val \
                --cfg-options \
                    data.train.dataset.ann_file=$ANN_FILE \
                    data.val.ann_file=$ANN_FILE \
                    model.cls_head.num_classes=5
            
            # Print key metrics
            if [ -f "${WORK_DIR}/eval_val/metrics.json" ]; then
                python -c "
import json
with open('${WORK_DIR}/eval_val/metrics.json') as f:
    m = json.load(f)
print(f'    Top-1: {m.get(\"clip_top1_acc\", 0):.2f}%, Macro-F1: {m.get(\"clip_macro_f1\", 0):.2f}%')
"
            fi
        else
            echo "    WARNING: No checkpoint found at $BEST_CKPT"
        fi
    done
    
    # 4-Stream Fusion
    echo ""
    echo "  Running 4-Stream Fusion for Fold ${FOLD}..."
    
    python -c "
import torch
import numpy as np
from functools import wraps
import json
import pickle

_original_torch_load = torch.load
@wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from mmcv import Config
from mmcv.runner import load_checkpoint
from pyskl.datasets import build_dataloader, build_dataset
from pyskl.models import build_model
from pathlib import Path
import glob
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

FOLD = ${FOLD}
ANN_FILE = '${ANN_FILE}'
BASE_WORK_DIR = '${BASE_WORK_DIR}'
CLASS_NAMES = ${CLASS_NAMES}

def top_k_accuracy(scores, labels, topk=(1, 2)):
    results = {}
    maxk = max(topk)
    top_k_preds = np.argsort(scores, axis=1)[:, ::-1][:, :maxk]
    for k in topk:
        correct = sum(1 for i, label in enumerate(labels) if label in top_k_preds[i, :k])
        results[f'top{k}_acc'] = correct / len(labels) * 100
    return results

def compute_all_metrics(y_true, y_pred, y_scores, prefix=''):
    metrics = {}
    topk = top_k_accuracy(y_scores, y_true, topk=(1, 2))
    metrics[f'{prefix}top1_acc'] = topk['top1_acc']
    metrics[f'{prefix}top2_acc'] = topk['top2_acc']
    metrics[f'{prefix}macro_f1'] = f1_score(y_true, y_pred, average='macro') * 100
    metrics[f'{prefix}weighted_f1'] = f1_score(y_true, y_pred, average='weighted') * 100
    metrics[f'{prefix}macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics[f'{prefix}macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics[f'{prefix}cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    for i, class_name in enumerate(CLASS_NAMES):
        y_true_binary = (np.array(y_true) == i).astype(int)
        y_pred_binary = (np.array(y_pred) == i).astype(int)
        if y_true_binary.sum() > 0:
            metrics[f'{prefix}f1_{class_name}'] = f1_score(y_true_binary, y_pred_binary) * 100
            metrics[f'{prefix}prec_{class_name}'] = precision_score(y_true_binary, y_pred_binary, zero_division=0) * 100
            metrics[f'{prefix}rec_{class_name}'] = recall_score(y_true_binary, y_pred_binary, zero_division=0) * 100
        else:
            metrics[f'{prefix}f1_{class_name}'] = 0.0
            metrics[f'{prefix}prec_{class_name}'] = 0.0
            metrics[f'{prefix}rec_{class_name}'] = 0.0
    return metrics

def plot_confusion_matrix(cm, output_path, title='Confusion Matrix'):
    if not HAS_PLOT:
        return
    plt.figure(figsize=(12, 10))
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

def get_predictions(config_path, checkpoint_path, ann_file):
    cfg = Config.fromfile(config_path)
    cfg.data.test.ann_file = ann_file
    cfg.data.test.split = 'val'
    cfg.model.cls_head.num_classes = 5
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader = build_dataloader(dataset, videos_per_gpu=1, workers_per_gpu=2, shuffle=False, drop_last=False)
    
    model = build_model(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.cuda()
    model.eval()
    
    all_scores, all_labels = [], []
    with torch.no_grad():
        for data in dataloader:
            keypoint = data['keypoint'].cuda()
            label = data['label'].item()
            output = model(keypoint, label=None, return_loss=False)
            scores = output[0] if isinstance(output, (list, tuple)) else output
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            all_scores.append(scores.squeeze())
            all_labels.append(label)
    return np.array(all_scores), np.array(all_labels)

modalities = ['j', 'b', 'jm', 'bm']
weights = [2, 2, 1, 1]
all_probs = []
labels = None
available_mods = []

for mod in modalities:
    config = f'configs/stgcn++/stgcnpp_sails_ntu60p/{mod}_tal_5class.py'
    ckpt_pattern = f'{BASE_WORK_DIR}/{mod}/fold{FOLD}/best_*.pth'
    ckpts = glob.glob(ckpt_pattern)
    ckpt = ckpts[0] if ckpts else f'{BASE_WORK_DIR}/{mod}/fold{FOLD}/latest.pth'
    
    if not Path(ckpt).exists():
        print(f'    WARNING: Checkpoint not found for {mod}')
        continue
    
    print(f'    Loading {mod}...')
    scores, labels = get_predictions(config, ckpt, ANN_FILE)
    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    all_probs.append(probs)
    available_mods.append(mod)

if all_probs:
    used_weights = [weights[modalities.index(m)] for m in available_mods]
    fused_probs = sum(w * p for w, p in zip(used_weights, all_probs)) / sum(used_weights)
    fused_preds = fused_probs.argmax(axis=1)
    
    output_dir = Path(f'{BASE_WORK_DIR}/4stream/fold{FOLD}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = compute_all_metrics(labels, fused_preds, fused_probs, prefix='clip_')
    metrics['num_clips'] = len(labels)
    metrics['modalities_used'] = available_mods
    
    cm = confusion_matrix(labels, fused_preds)
    plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png', f'Fold {FOLD} - 4-Stream Fusion (TAL)')
    
    with open(output_dir / 'confusion_matrix.json', 'w') as f:
        json.dump({'matrix': cm.tolist(), 'class_names': CLASS_NAMES}, f, indent=2)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in metrics.items()}, f, indent=2)
    
    print(f'    4-Stream Fusion Results:')
    print(f'      Top-1: {metrics[\"clip_top1_acc\"]:.2f}%, Macro-F1: {metrics[\"clip_macro_f1\"]:.2f}%')
    print(f'      Per-class F1:')
    for cn in CLASS_NAMES:
        print(f'        {cn}: {metrics.get(f\"clip_f1_{cn}\", 0):.2f}%')
"
done

# CV Summary
echo ""
echo "=========================================="
echo "4-Stream Fusion CV Summary"
echo "=========================================="

python -c "
import json
import numpy as np
from pathlib import Path

base_dir = Path('${BASE_WORK_DIR}/4stream')
folds = []

for fold in range(2):
    metrics_file = base_dir / f'fold{fold}' / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
            m['fold'] = fold
            folds.append(m)

if folds:
    summary = {'per_fold': folds, 'note': 'Partial CV (fold 0-1 only)'}
    metric_keys = [k for k in folds[0].keys() if k not in ['fold', 'modalities_used']]
    for key in metric_keys:
        vals = [f[key] for f in folds if key in f]
        if vals and isinstance(vals[0], (int, float)):
            summary[f'{key}_mean'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))
    
    with open(base_dir / 'cv_summary_partial.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print('4-Stream Fusion CV Summary:')
    print(f'  Clip Top-1:    {summary.get(\"clip_top1_acc_mean\", 0):.2f} ± {summary.get(\"clip_top1_acc_std\", 0):.2f}%')
    print(f'  Clip Macro-F1: {summary.get(\"clip_macro_f1_mean\", 0):.2f} ± {summary.get(\"clip_macro_f1_std\", 0):.2f}%')
    print()
    print('Per-class F1:')
    for cn in ['hands flapping', 'jumping', 'rocking', 'spinning', 'background']:
        mean_key = f'clip_f1_{cn}_mean'
        std_key = f'clip_f1_{cn}_std'
        if mean_key in summary:
            print(f'  {cn}: {summary[mean_key]:.2f} ± {summary[std_key]:.2f}%')
"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="


