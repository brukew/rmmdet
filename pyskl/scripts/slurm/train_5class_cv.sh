#!/bin/bash
#SBATCH --job-name=posec3d_5class_cv
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/posec3d/5class_cv_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/posec3d/5class_cv_%j.err

echo "=========================================="
echo "PoseC3D Training: 5-class Cross-Validation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

# Configuration
CONFIG="configs/posec3d/slowonly_r50_sails_k400p/joint.py"
DATA_DIR="data/sails/cv/5class_conf04"
WORK_DIR="work_dirs/posec3d/cv/5class_conf04"
EPOCHS=12
LR=0.00125  # Scaled for 1 GPU
NUM_CLASSES=5

# Class names for 5-class
CLASS_NAMES="hands_flapping jumping one_hand_flap rocking spinning"

echo ""
echo "Config: $CONFIG"
echo "Data dir: $DATA_DIR"
echo "Work dir: $WORK_DIR"
echo "Epochs: $EPOCHS"
echo "LR: $LR"
echo "Num classes: $NUM_CLASSES"
echo ""

mkdir -p $WORK_DIR

# Train each fold sequentially
for FOLD in 0 1 2; do
    echo ""
    echo "=========================================="
    echo "FOLD $FOLD / 2"
    echo "=========================================="
    
    FOLD_ANN="${DATA_DIR}/fold${FOLD}.pkl"
    FOLD_WORK="${WORK_DIR}/fold${FOLD}"
    
    if [ ! -f "$FOLD_ANN" ]; then
        echo "ERROR: Annotation file not found: $FOLD_ANN"
        continue
    fi
    
    echo "Training fold $FOLD..."
    python tools/train.py $CONFIG \
        --work-dir $FOLD_WORK \
        --cfg-options \
            ann_file=$FOLD_ANN \
            data.train.dataset.ann_file=$FOLD_ANN \
            data.val.ann_file=$FOLD_ANN \
            data.test.ann_file=$FOLD_ANN \
            total_epochs=$EPOCHS \
            optimizer.lr=$LR \
            model.cls_head.num_classes=$NUM_CLASSES \
        --validate \
        --launcher none
    
    # Evaluate
    BEST_CKPT=$(ls -t ${FOLD_WORK}/best_*.pth 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT="${FOLD_WORK}/latest.pth"
    fi
    
    echo "Evaluating fold $FOLD..."
    python tools/evaluate_sails.py $CONFIG \
        -C $BEST_CKPT \
        --split val \
        --output-dir ${FOLD_WORK}/eval_val \
        --cfg-options ann_file=$FOLD_ANN model.cls_head.num_classes=$NUM_CLASSES \
        --class-names $CLASS_NAMES
done

# Aggregate CV results
echo ""
echo "=========================================="
echo "Aggregating CV Results"
echo "=========================================="

python3 << EOF
import json
import numpy as np
from pathlib import Path

work_dir = "$WORK_DIR"
folds = [0, 1, 2]

all_metrics = []
for fold in folds:
    metrics_path = Path(work_dir) / f"fold{fold}" / "eval_val" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_metrics.append(json.load(f))
        print(f"Loaded fold {fold} metrics")

if not all_metrics:
    print("No metrics found")
    exit(0)

# Compute averages
avg_metrics = {}
std_metrics = {}
metric_keys = [k for k in all_metrics[0].keys() if isinstance(all_metrics[0][k], (int, float))]

for key in metric_keys:
    values = [m[key] for m in all_metrics if key in m]
    if values:
        avg_metrics[key] = float(np.mean(values))
        std_metrics[key] = float(np.std(values))

print("\n" + "=" * 50)
print("5-CLASS CV RESULTS (mean ± std)")
print("=" * 50)

for level in ['clip', 'video']:
    print(f"\n{level.upper()}-Level:")
    for metric in ['top1_acc', 'top2_acc', 'macro_f1', 'weighted_f1', 'cohens_kappa']:
        key = f'{level}_{metric}'
        if key in avg_metrics:
            if 'kappa' in key:
                print(f"  {metric}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
            else:
                print(f"  {metric}: {avg_metrics[key]:.2f}% ± {std_metrics[key]:.2f}%")

# Save summary
summary = {"num_folds": len(all_metrics), "average": avg_metrics, "std": std_metrics}
with open(Path(work_dir) / "cv_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to: {work_dir}/cv_summary.json")
EOF

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

