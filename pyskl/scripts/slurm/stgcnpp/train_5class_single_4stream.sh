#!/bin/bash
#SBATCH --job-name=stgcnpp_5cls_4s
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/5class_single_4stream_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/5class_single_4stream_%j.err

echo "=========================================="
echo "STGCN++ Training: 5-class Single Split (4-Stream)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

# Configuration
ANN_FILE="data/sails/single/5class_conf04.pkl"
BASE_WORK_DIR="work_dirs/stgcnpp/single/5class_conf04"
EPOCHS=24
LR=0.01

MODALITIES=("j" "b" "jm" "bm")
MODALITY_NAMES=("Joint" "Bone" "Joint Motion" "Bone Motion")

for i in "${!MODALITIES[@]}"; do
    MOD="${MODALITIES[$i]}"
    MOD_NAME="${MODALITY_NAMES[$i]}"
    CONFIG="configs/stgcn++/stgcnpp_sails_ntu60p/${MOD}.py"
    WORK_DIR="${BASE_WORK_DIR}_${MOD}"
    
    echo ""
    echo "=========================================="
    echo "Training ${MOD_NAME} (${MOD}) modality"
    echo "=========================================="
    
    # Train with 5 classes
    python tools/train.py $CONFIG \
        --work-dir $WORK_DIR \
        --cfg-options \
            model.cls_head.num_classes=5 \
            data.train.dataset.ann_file=$ANN_FILE \
            data.val.ann_file=$ANN_FILE \
            data.test.ann_file=$ANN_FILE \
            total_epochs=$EPOCHS \
            optimizer.lr=$LR \
        --validate --test-best \
        --launcher none
    
    # Evaluate on test set
    BEST_CKPT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT="${WORK_DIR}/latest.pth"
    fi
    
    echo "Evaluating ${MOD}: $BEST_CKPT"
    
    python tools/evaluate_sails.py $CONFIG \
        -C $BEST_CKPT \
        --split test \
        --output-dir ${WORK_DIR}/eval_test \
        --cfg-options \
            model.cls_head.num_classes=5 \
            data.train.dataset.ann_file=$ANN_FILE \
            data.val.ann_file=$ANN_FILE \
            data.test.ann_file=$ANN_FILE
done

# 4-Stream Fusion with Comprehensive Evaluation
echo ""
echo "=========================================="
echo "Running 4-Stream Fusion with Full Metrics"
echo "=========================================="

python -c "
import torch
import numpy as np
from functools import wraps
import json
import csv
import pickle

# Monkey-patch torch.load for PyTorch 2.6+ compatibility
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

CLASS_NAMES = ['hands flapping', 'jumping', 'one hand flap', 'rocking', 'spinning']
NUM_CLASSES = 5

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
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        if i < len(per_class_f1):
            metrics[f'{prefix}f1_{name}'] = per_class_f1[i] * 100
            metrics[f'{prefix}prec_{name}'] = per_class_prec[i] * 100
            metrics[f'{prefix}rec_{name}'] = per_class_rec[i] * 100
    return metrics

def plot_confusion_matrix(cm, output_path, title='Confusion Matrix'):
    if not HAS_PLOT:
        return
    plt.figure(figsize=(12, 10))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, square=True)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j + 0.5, i + 0.7, f'({cm[i, j]})', ha='center', va='center', fontsize=8, color='gray')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')

def get_predictions_with_ids(config_path, checkpoint_path, ann_file, num_classes=5):
    cfg = Config.fromfile(config_path)
    cfg.model.cls_head.num_classes = num_classes
    cfg.data.test.ann_file = ann_file
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader = build_dataloader(dataset, videos_per_gpu=1, workers_per_gpu=2, shuffle=False, drop_last=False)
    
    with open(ann_file, 'rb') as f:
        ann_data = pickle.load(f)
    annotations = ann_data['annotations']
    test_ids = set(ann_data['split'].get('test', ann_data['split'].get('val', [])))
    test_annotations = [a for a in annotations if a.get('frame_dir', a.get('filename', '')) in test_ids]
    
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
            seg_id = test_annotations[idx].get('frame_dir', test_annotations[idx].get('filename', f'sample_{idx}')) if idx < len(test_annotations) else f'sample_{idx}'
            segment_ids.append(seg_id)
            idx += 1
    return np.array(all_scores), np.array(all_labels), segment_ids

# Get predictions from all 4 modalities
modalities = ['j', 'b', 'jm', 'bm']
weights = [2, 2, 1, 1]
all_probs, all_scores_raw = [], []
labels, segment_ids = None, None

for mod in modalities:
    config = f'configs/stgcn++/stgcnpp_sails_ntu60p/{mod}.py'
    ckpt_pattern = f'${BASE_WORK_DIR}_{mod}/best_*.pth'
    ckpts = glob.glob(ckpt_pattern)
    ckpt = ckpts[0] if ckpts else f'${BASE_WORK_DIR}_{mod}/latest.pth'
    
    print(f'Loading {mod}: {ckpt}')
    scores, labels, segment_ids = get_predictions_with_ids(config, ckpt, '${ANN_FILE}', num_classes=NUM_CLASSES)
    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    all_probs.append(probs)

# Weighted fusion
fused_probs = sum(w * p for w, p in zip(weights, all_probs)) / sum(weights)
fused_preds = fused_probs.argmax(axis=1)

# Output directory
output_dir = Path('${BASE_WORK_DIR}_4stream/eval_test')
output_dir.mkdir(parents=True, exist_ok=True)

# Compute all metrics (clip-level)
print('\\n=== Clip-Level Metrics ===')
metrics = compute_all_metrics(labels, fused_preds, fused_probs, prefix='clip_')
for k, v in sorted(metrics.items()):
    if 'top' in k or 'f1' in k.lower() or 'kappa' in k:
        print(f'  {k}: {v:.2f}' if isinstance(v, float) and v > 1 else f'  {k}: {v:.4f}')

# Individual modality accuracies
print('\\n=== Per-Modality Accuracies ===')
modality_results = {}
for mod, probs in zip(modalities, all_probs):
    preds = probs.argmax(axis=1)
    acc = (preds == labels).mean() * 100
    modality_results[mod] = acc
    print(f'  {mod}: {acc:.2f}%')
modality_results['4stream_fusion'] = (fused_preds == labels).mean() * 100
print(f'  4-stream: {modality_results[\"4stream_fusion\"]:.2f}%')
metrics['modality_accuracies'] = modality_results

# Video-level aggregation
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
video_preds = np.array([np.bincount(video_preds_dict[vid], minlength=NUM_CLASSES).argmax() for vid in video_ids])
video_scores = np.array([np.mean(video_scores_dict[vid], axis=0) for vid in video_ids])

print('\\n=== Video-Level Metrics ===')
video_metrics = compute_all_metrics(video_labels, video_preds, video_scores, prefix='video_')
metrics.update(video_metrics)
for k, v in sorted(video_metrics.items()):
    if 'top' in k or 'f1' in k.lower() or 'kappa' in k:
        print(f'  {k}: {v:.2f}' if isinstance(v, float) and v > 1 else f'  {k}: {v:.4f}')

# Confusion matrices
cm_clip = confusion_matrix(labels, fused_preds)
cm_video = confusion_matrix(video_labels, video_preds)
plot_confusion_matrix(cm_clip, output_dir / 'confusion_matrix_clip.png', '4-Stream Fusion - Clip Level (5-class)')
plot_confusion_matrix(cm_video, output_dir / 'confusion_matrix_video.png', '4-Stream Fusion - Video Level (5-class)')

# Save predictions CSVs
with open(output_dir / 'predictions_clip.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['segment_id', 'true_label', 'pred_label', 'true_class', 'pred_class'] + [f'score_class{i}' for i in range(NUM_CLASSES)]
    writer.writerow(header)
    for seg_id, true_label, pred_label, scores in zip(segment_ids, labels, fused_preds, fused_probs):
        row = [seg_id, true_label, pred_label, CLASS_NAMES[true_label], CLASS_NAMES[pred_label]] + [f'{s:.4f}' for s in scores]
        writer.writerow(row)
print(f'  Saved: {output_dir}/predictions_clip.csv')

with open(output_dir / 'predictions_video.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['video_id', 'true_label', 'pred_label', 'true_class', 'pred_class'] + [f'score_class{i}' for i in range(NUM_CLASSES)]
    writer.writerow(header)
    for vid, true_label, pred_label, scores in zip(video_ids, video_labels, video_preds, video_scores):
        row = [vid, true_label, pred_label, CLASS_NAMES[true_label], CLASS_NAMES[pred_label]] + [f'{s:.4f}' for s in scores]
        writer.writerow(row)
print(f'  Saved: {output_dir}/predictions_video.csv')

# Save all metrics
with open(output_dir / 'metrics.json', 'w') as f:
    json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()}, f, indent=2)
print(f'  Saved: {output_dir}/metrics.json')

print('\\n4-Stream fusion evaluation complete!')
"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="


