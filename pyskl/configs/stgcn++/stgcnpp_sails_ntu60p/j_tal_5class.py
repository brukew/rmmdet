"""
STGCN++ config for SAILS TAL (Temporal Action Localization) windows.

Classes (5): 4 RMM types + background
    0: hands flapping
    1: jumping
    2: rocking
    3: spinning
    4: background

Note: Background class dominates (~89%). Use train_weighted.py with --bg-subsample
to reduce background via class_prob sampling.

Usage:
    # Training with background subsampling and class weighting
    python tools/train_weighted.py configs/stgcn++/stgcnpp_sails_ntu60p/j_tal_5class.py \
        --ann-file data/sails/tal/cv_4class/5class_windows_conf04/fold0.pkl \
        --bg-subsample 0.2 \
        --work-dir work_dirs/stgcnpp/tal/cv_4class_5class_bgsub/fold0 \
        --validate --launcher none
"""

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',      # Learnable adjacency matrix
        gcn_with_res=True,        # Enhanced residual connections
        tcn_type='mstcn',         # Multi-scale temporal convolution
        graph_cfg=dict(layout='coco', mode='spatial')),  # COCO-17 keypoints
    cls_head=dict(type='GCNHead', num_classes=5, in_channels=256))  # 4 RMM + background

# Dataset configuration
dataset_type = 'PoseDataset'
# Default annotation file (override via --ann-file in train_weighted.py)
ann_file = 'data/sails/tal/cv_4class/5class_windows_conf04/fold0.pkl'

# Data pipelines - adapted for TAL 2s windows (~60 frames at 30fps)
train_pipeline = [
    dict(type='PreNormalize2D'),  # Normalize 2D keypoints to [-1, 1]
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),  # Joint features
    dict(type='UniformSample', clip_len=48),  # Sample 48 frames from 2s window
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),  # SAILS: single person per clip
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=48, num_clips=10),  # 10 clips for TTA
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

# Data loaders
# Note: class_prob will be injected by train_weighted.py when --bg-subsample is used
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=test_pipeline))  # Use val as test for CV

# Optimizer - lower learning rate for finetuning
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)

# Learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)

# Training settings
total_epochs = 24  # More epochs since early stopping will cut short if needed
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 2))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'

# Early stopping configuration
early_stopping = dict(
    patience=5,  # Stop if no improvement for 5 epochs
    monitor='top1_acc',
    min_delta=0.0)

# Output directory (override via --work-dir)
work_dir = './work_dirs/stgcnpp/tal/cv_4class_5class_bgsub'

# ============================================================================
# CRITICAL: Pretrained weights from NTU60 HRNet
# ============================================================================
load_from = 'checkpoints/stgcnpp/ntu60_hrnet_j.pth'
find_unused_parameters = True  # Required when num_classes differs from pretrained



