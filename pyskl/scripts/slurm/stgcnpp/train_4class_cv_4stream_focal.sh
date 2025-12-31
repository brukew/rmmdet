#!/bin/bash
#SBATCH --job-name=stgcnpp_4cls_cv_focal
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/4class_cv_4stream_focal_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/4class_cv_4stream_focal_%j.err

echo "=========================================="
echo "STGCN++ Training: 4-class CV (4-Stream) WITH FOCAL LOSS + EARLY STOPPING"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

BASE_ANN_DIR="data/sails/cv/4class_conf04"
BASE_WORK_DIR="work_dirs/stgcnpp/cv/4class_conf04_focal"
EPOCHS=24
LR=0.01
# Class names - use default (don't pass --class-names for 4-class)

MODALITIES=("j" "b" "jm" "bm")
MODALITY_NAMES=("Joint" "Bone" "Joint Motion" "Bone Motion")

for FOLD in 0 1 2; do
    ANN_FILE="${BASE_ANN_DIR}/fold${FOLD}.pkl"
    
    echo ""
    echo "######################################################"
    echo "# FOLD ${FOLD}"
    echo "######################################################"
    
    for i in "${!MODALITIES[@]}"; do
        MOD="${MODALITIES[$i]}"
        MOD_NAME="${MODALITY_NAMES[$i]}"
        CONFIG="configs/stgcn++/stgcnpp_sails_ntu60p/${MOD}_focal.py"
        WORK_DIR="${BASE_WORK_DIR}_${MOD}/fold${FOLD}"
        
        echo ""
        echo "=========================================="
        echo "Fold ${FOLD}: Training ${MOD_NAME} (${MOD}) WITH FOCAL LOSS"
        echo "=========================================="
        
        # Train with focal loss (early stopping configured in config)
        python tools/train.py $CONFIG \
            --work-dir $WORK_DIR \
            --validate \
            --launcher none \
            --cfg-options \
                data.train.dataset.ann_file=$ANN_FILE \
                data.val.ann_file=$ANN_FILE \
                total_epochs=$EPOCHS \
                optimizer.lr=$LR
        
        BEST_CKPT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)
        if [ -z "$BEST_CKPT" ]; then
            BEST_CKPT="${WORK_DIR}/latest.pth"
        fi
        
        echo "Evaluating ${MOD}: $BEST_CKPT"
        
        python tools/evaluate_sails.py $CONFIG \
            -C $BEST_CKPT \
            --split val \
            --output-dir ${WORK_DIR}/eval_val \
            --cfg-options \
                data.train.dataset.ann_file=$ANN_FILE \
                data.val.ann_file=$ANN_FILE
    done
    
    # 4-Stream Fusion for this fold
    echo ""
    echo "=========================================="
    echo "Fold ${FOLD}: Running 4-Stream Fusion"
    echo "=========================================="
    
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
CLASS_NAMES = ['hands flapping', 'jumping', 'rocking', 'spinning']

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
    return metrics

def plot_confusion_matrix(cm, output_path, title='Confusion Matrix'):
    if not HAS_PLOT:
        return
    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, square=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def get_predictions_with_ids(config_path, checkpoint_path, ann_file):
    cfg = Config.fromfile(config_path)
    cfg.data.test.ann_file = ann_file
    cfg.data.test.split = 'val'
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader = build_dataloader(dataset, videos_per_gpu=1, workers_per_gpu=2, shuffle=False, drop_last=False)
    
    with open(ann_file, 'rb') as f:
        ann_data = pickle.load(f)
    annotations = ann_data['annotations']
    val_ids = set(ann_data['split'].get('val', []))
    val_annotations = [a for a in annotations if a.get('frame_dir', '') in val_ids]
    
    model = build_model(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.cuda()
    model.eval()
    
    all_scores, all_labels, segment_ids = [], [], []
    idx = 0
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
            seg_id = val_annotations[idx].get('frame_dir', f'sample_{idx}') if idx < len(val_annotations) else f'sample_{idx}'
            segment_ids.append(seg_id)
            idx += 1
    return np.array(all_scores), np.array(all_labels), segment_ids

modalities = ['j', 'b', 'jm', 'bm']
weights = [2, 2, 1, 1]
all_probs = []
labels, segment_ids = None, None

for mod in modalities:
    config = f'configs/stgcn++/stgcnpp_sails_ntu60p/{mod}_focal.py'
    ckpt_pattern = f'work_dirs/stgcnpp/cv/4class_conf04_focal_{mod}/fold{FOLD}/best_*.pth'
    ckpts = glob.glob(ckpt_pattern)
    ckpt = ckpts[0] if ckpts else f'work_dirs/stgcnpp/cv/4class_conf04_focal_{mod}/fold{FOLD}/latest.pth'
    
    print(f'Loading {mod}: {ckpt}')
    scores, labels, segment_ids = get_predictions_with_ids(config, ckpt, ANN_FILE)
    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    all_probs.append(probs)

fused_probs = sum(w * p for w, p in zip(weights, all_probs)) / sum(weights)
fused_preds = fused_probs.argmax(axis=1)

output_dir = Path(f'work_dirs/stgcnpp/cv/4class_conf04_focal_4stream/fold{FOLD}')
output_dir.mkdir(parents=True, exist_ok=True)

metrics = compute_all_metrics(labels, fused_preds, fused_probs, prefix='clip_')

from collections import defaultdict
video_preds_dict = defaultdict(list)
video_labels_dict = {}
video_scores_dict = defaultdict(list)
for seg_id, pred, label, scores in zip(segment_ids, fused_preds, labels, fused_probs):
    video_id = '_'.join(str(seg_id).split('_')[:-1]) if '_' in str(seg_id) else str(seg_id)
    video_preds_dict[video_id].append(pred)
    video_labels_dict[video_id] = label
    video_scores_dict[video_id].append(scores)

video_ids = list(video_preds_dict.keys())
video_labels = np.array([video_labels_dict[vid] for vid in video_ids])
video_preds = np.array([np.bincount(video_preds_dict[vid], minlength=len(CLASS_NAMES)).argmax() for vid in video_ids])
video_scores = np.array([np.mean(video_scores_dict[vid], axis=0) for vid in video_ids])

video_metrics = compute_all_metrics(video_labels, video_preds, video_scores, prefix='video_')
metrics.update(video_metrics)

cm_clip = confusion_matrix(labels, fused_preds)
cm_video = confusion_matrix(video_labels, video_preds)
plot_confusion_matrix(cm_clip, output_dir / 'confusion_matrix_clip.png', f'Fold {FOLD} - Clip (Focal)')
plot_confusion_matrix(cm_video, output_dir / 'confusion_matrix_video.png', f'Fold {FOLD} - Video (Focal)')

with open(output_dir / 'metrics.json', 'w') as f:
    json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()}, f, indent=2)

print(f'Fold {FOLD}: clip_top1={metrics[\"clip_top1_acc\"]:.2f}%, video_top1={metrics[\"video_top1_acc\"]:.2f}%')
"
done

# CV Summary
echo ""
echo "=========================================="
echo "Computing CV Summary"
echo "=========================================="

python -c "
import json
import numpy as np
from pathlib import Path

base_dir = Path('work_dirs/stgcnpp/cv/4class_conf04_focal_4stream')
folds = []

for fold in range(3):
    metrics_file = base_dir / f'fold{fold}' / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
            m['fold'] = fold
            folds.append(m)

if folds:
    summary = {'per_fold': folds}
    metric_keys = [k for k in folds[0].keys() if k != 'fold']
    for key in metric_keys:
        vals = [f[key] for f in folds if key in f]
        if vals and isinstance(vals[0], (int, float)):
            summary[f'{key}_mean'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))
    
    with open(base_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print('CV Summary (Focal Loss + Early Stopping):')
    print(f'  Clip Top-1: {summary.get(\"clip_top1_acc_mean\", 0):.2f} ± {summary.get(\"clip_top1_acc_std\", 0):.2f}%')
    print(f'  Video Top-1: {summary.get(\"video_top1_acc_mean\", 0):.2f} ± {summary.get(\"video_top1_acc_std\", 0):.2f}%')
    print(f'  Clip Macro-F1: {summary.get(\"clip_macro_f1_mean\", 0):.2f} ± {summary.get(\"clip_macro_f1_std\", 0):.2f}%')
    print(f'  Video Macro-F1: {summary.get(\"video_macro_f1_mean\", 0):.2f} ± {summary.get(\"video_macro_f1_std\", 0):.2f}%')
"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

