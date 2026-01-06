"""
STGCN++ config for SAILS TAL (Temporal Action Localization) windows.
Joint Motion modality (jm) - temporal velocity of keypoints.

Classes (5): 4 RMM types + background
    0: hands flapping
    1: jumping
    2: rocking
    3: spinning
    4: background

Usage:
    python tools/train_weighted.py configs/stgcn++/stgcnpp_sails_ntu60p/jm_tal_5class.py \
        --ann-file data/sails/tal/cv_4class/5class_windows_conf04/fold0.pkl \
        --bg-subsample 0.2 \
        --work-dir work_dirs/stgcnpp/tal/cv_4class_5class_bgsub/jm/fold0 \
        --validate --launcher none
"""

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=5, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/sails/tal/cv_4class/5class_windows_conf04/fold0.pkl'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='UniformSample', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='UniformSample', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='UniformSample', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)

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

work_dir = './work_dirs/stgcnpp/tal/cv_4class_5class_bgsub/jm'

load_from = 'checkpoints/stgcnpp/ntu60_hrnet_jm.pth'
find_unused_parameters = True



