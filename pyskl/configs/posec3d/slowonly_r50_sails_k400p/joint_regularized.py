# PoseC3D config with STRONG regularization for small dataset
# Addresses severe overfitting observed in training runs

_base_ = './joint.py'

# ========================
# KEY ANTI-OVERFITTING CHANGES
# ========================

# 1. FREEZE BACKBONE - Only train classifier head
# frozen_stages=3 freezes all 3 ResNet stages (the backbone has 3 stages for PoseC3D)
# This dramatically reduces trainable parameters
model = dict(
    backbone=dict(
        frozen_stages=3,  # Freeze entire backbone!
    ),
)

# 2. STRONGER WEIGHT DECAY + Lower LR
optimizer = dict(
    type='SGD',
    lr=0.00125,  # Lower LR for small dataset
    momentum=0.9,
    weight_decay=0.001,  # Increased from 0.0003
)

# 3. FEWER EPOCHS with early stopping behavior
total_epochs = 10  # Reduced from 12

# 4. REDUCED DATA REPETITION (at RepeatDataset level)
data = dict(
    train=dict(
        times=5,  # Reduced from 10
    )
)
