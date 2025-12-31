"""
PoseC3D config for SAILS RMM dataset with Focal Loss.

Focal Loss addresses class imbalance by down-weighting easy examples and
focusing on hard, misclassified samples. This is particularly useful for
the SAILS dataset where 'hands flapping' dominates (55% of samples).

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Parameters:
    - gamma: Focusing parameter. Higher values = more focus on hard examples.
             gamma=0 is equivalent to standard cross-entropy.
             gamma=2 is the recommended default from the original paper.
    - alpha: Per-class weights to balance class frequencies.
             Can use inverse-frequency weights or manual tuning.

Classes (4):
    0: hands flapping (264 samples, ~55%)
    1: jumping (116 samples, ~24%)
    2: rocking (57 samples, ~12%)
    3: spinning (36 samples, ~8%)

Usage:
    # Training with validation
    python tools/train.py configs/posec3d/slowonly_r50_sails_k400p/joint_focal.py \
        --work-dir work_dirs/posec3d/single/4class_focal \
        --validate --launcher none

    # With class weights (alpha)
    python tools/train.py configs/posec3d/slowonly_r50_sails_k400p/joint_focal.py \
        --work-dir work_dirs/posec3d/single/4class_focal_weighted \
        --validate --launcher none \
        --cfg-options model.cls_head.loss_cls.alpha='[0.45,1.02,2.07,3.28]'
"""

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(3, 4, 6),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=4,
        dropout=0.5,
        # Focal Loss configuration
        loss_cls=dict(
            type='FocalLoss',
            gamma=2.0,           # Focusing parameter (2.0 is standard)
            alpha=None,          # Set to list for class weights, e.g. [0.45, 1.02, 2.07, 3.28]
            loss_weight=1.0)),
    test_cfg=dict(average_clips='prob'))

# Dataset configuration
dataset_type = 'PoseDataset'
ann_file = 'data/sails/single/4class_conf04.pkl'

# COCO keypoint indices for flipping augmentation
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# Data pipelines
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(48, 48), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=True, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

# Data loaders
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='test', pipeline=test_pipeline))

# Optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0003)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# Learning rate schedule
lr_config = dict(policy='step', step=[9, 11])

# Training settings
total_epochs = 12
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 2))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'

# Early stopping configuration
early_stopping = dict(
    patience=2,
    monitor='top1_acc',
    min_delta=0.0)

# Output directory
work_dir = './work_dirs/posec3d/slowonly_r50_sails_k400p/joint_focal'

# Pretrained weights
load_from = 'https://download.openmmlab.com/mmaction/skeleton/posec3d/k400_posec3d-041f49c6.pth'
find_unused_parameters = True

