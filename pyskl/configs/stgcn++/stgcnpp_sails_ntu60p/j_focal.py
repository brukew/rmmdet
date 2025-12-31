"""
STGCN++ config for SAILS RMM dataset with Focal Loss.
Joint modality (j) - uses raw keypoint coordinates.

Focal Loss addresses class imbalance by down-weighting easy examples and
focusing on hard, misclassified samples.

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Classes (4):
    0: hands flapping
    1: jumping
    2: rocking
    3: spinning

Usage:
    # Training with validation (single GPU)
    python tools/train.py configs/stgcn++/stgcnpp_sails_ntu60p/j_focal.py --launcher none --validate

    # With early stopping
    python tools/train.py configs/stgcn++/stgcnpp_sails_ntu60p/j_focal.py --launcher none --validate \
        --cfg-options early_stopping.patience=2
"""

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(
        type='GCNHead',
        num_classes=4,
        in_channels=256,
        loss_cls=dict(
            type='FocalLoss',
            gamma=2.0,
            alpha=None,
            loss_weight=1.0)))

# Dataset configuration
dataset_type = 'PoseDataset'
ann_file = 'data/sails/single/4class_conf04.pkl'

# Data pipelines - JOINT modality
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10),
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
        times=10,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='test', pipeline=test_pipeline))

# Optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)

# Learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)

# Training settings
total_epochs = 24
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
work_dir = './work_dirs/stgcn++/stgcnpp_sails_ntu60p/j_focal'

# Pretrained weights
load_from = 'checkpoints/stgcnpp/ntu60_hrnet_j.pth'
find_unused_parameters = True

