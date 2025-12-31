# STGCN++ config with STRONG regularization for small dataset
# Addresses severe overfitting observed in training runs

_base_ = './j.py'

# ========================
# KEY ANTI-OVERFITTING CHANGES
# ========================

# 1. ADD DROPOUT TO CLASSIFIER HEAD (was 0!)
model = dict(
    cls_head=dict(
        dropout=0.5,  # Critical: add regularization
    )
)

# 2. STRONGER WEIGHT DECAY + Lower LR
optimizer = dict(
    type='SGD',
    lr=0.005,  # Reduced from 0.01
    momentum=0.9,
    weight_decay=0.001,  # Increased from 0.0005
    nesterov=True,
)

# 3. FEWER EPOCHS - best performance is often at epoch 4-8
total_epochs = 16  # Reduced from 24

# 4. REDUCED DATA REPETITION (at RepeatDataset level)
data = dict(
    train=dict(
        times=5,  # Reduced from 10
    )
)
