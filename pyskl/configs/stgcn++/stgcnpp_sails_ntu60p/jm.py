"""
STGCN++ config for SAILS RMM dataset finetuning from NTU60 HRNet pretrained weights.
Joint Motion modality (jm) - temporal velocity of keypoints.

Classes (4):
    0: hands flapping
    1: jumping
    2: rocking
    3: spinning

Usage:
    # Training with validation (single GPU)
    python tools/train.py configs/stgcn++/stgcnpp_sails_ntu60p/jm.py --launcher none --validate

    # Testing
    python tools/test.py configs/stgcn++/stgcnpp_sails_ntu60p/jm.py CHECKPOINT --launcher none --eval top_k_accuracy mean_class_accuracy
"""

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',      # Learnable adjacency matrix
        gcn_with_res=True,        # Enhanced residual connections
        tcn_type='mstcn',         # Multi-scale temporal convolution
        graph_cfg=dict(layout='coco', mode='spatial')),  # COCO-17 keypoints
    cls_head=dict(type='GCNHead', num_classes=4, in_channels=256))

# Dataset configuration
dataset_type = 'PoseDataset'
ann_file = 'data/sails/single/4class_conf04.pkl'  # Default: 4-class single split

# Data pipelines - JOINT MOTION modality
train_pipeline = [
    dict(type='PreNormalize2D'),  # Normalize 2D keypoints to [-1, 1]
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),  # Joint motion features
    dict(type='UniformSample', clip_len=100),  # Sample 100 frames
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),  # SAILS: single person per clip
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='UniformSample', clip_len=100, num_clips=10),  # 10 clips for TTA
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

# Data loaders
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,  # Repeat small dataset for more iterations per epoch
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='test', pipeline=test_pipeline))

# Optimizer - lower learning rate for finetuning
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)

# Learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)

# Training settings
total_epochs = 24  # More epochs for small dataset finetuning
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 2))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'

# Output directory
work_dir = './work_dirs/stgcn++/stgcnpp_sails_ntu60p/jm'

# ============================================================================
# CRITICAL: Pretrained weights from NTU60 HRNet (joint motion modality)
# ============================================================================
load_from = 'checkpoints/stgcnpp/ntu60_hrnet_jm.pth'
find_unused_parameters = True  # Required when num_classes differs from pretrained



