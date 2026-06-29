# Methods

We use a multi-modal approach to RMM recognition and localization, combining **RGB appearance**, **pose/skeleton dynamics**, and **late fusion**. Methods fall into three buckets aligned with the tasks:

- **Action recognition (clip-level classification)**: predict RMM type from short labeled clips.
- **Window-level detection**: predict RMM/background (and RMM type) from short fixed windows sampled from full videos.
- **Temporal Action Localization (TAL)**: predict temporal segments \([start, end]\) and associated labels over full videos, using either (1) window-score postprocessing or (2) end-to-end transformer localization.

Across all tasks, we account for **severe class imbalance** (especially background-heavy window data and rare RMM types) using a combination of **class weighting** and **balanced sampling** strategies.

## Models

This project evaluates complementary model families that trade off robustness, speed, and sensitivity to motion cues.

### PoseC3D (skeleton CNN)

- **Input**: pose keypoints converted to **pseudo-heatmaps** over time.
- **Core idea**: use a 3D CNN backbone to learn spatiotemporal motion patterns from pose.
- **Strengths**: motion-focused (less sensitive to background appearance), strong performance for common classes.
- **Limitations**: depends on pose quality and coverage; rare/subtle classes can remain difficult under heavy imbalance.

### STGCN++ (skeleton graph model)

- **Input**: skeleton **graphs** (keypoints as nodes; edges defined by the body graph).
- **Core idea**: graph convolutions + temporal convolutions model joint dynamics directly.
- **Strengths**: efficient and fast; multiple motion-centric modalities (joint/bone and motion variants) capture complementary dynamics.
- **Limitations**: sensitive to pose errors and missing frames; may lose appearance context helpful for certain behaviors.

### V-JEPA2 (RGB video encoder)

- **Input**: RGB frames (optionally cropped to a person-of-interest).
- **Core idea**: fine-tune a strong pretrained video encoder for RMM classification/detection, and reuse its features for TAL.
- **Strengths**: captures appearance + motion cues; supports both window-level models and feature extraction for transformer TAL.
- **Limitations**: can be more sensitive to background/scene variation without reliable person isolation.

### Qwen2.5-VL (zero-shot vision-language baseline)

- **Input**: RGB frame samples with prompt-based inference and multi-window voting.
- **Role**: provides a no-training baseline to quantify how far domain-specific learning is needed for subtle RMM motion patterns.

**How frames are passed in (Qwen2.5-VL):**
- **Multi-window inference**: each clip is evaluated using **4 temporal windows**, each represented by **16 consecutive frames**.
- **Window placement**: the 4 windows are sampled to cover the clip over time (evenly spaced), so the VLM sees multiple parts of the motion sequence.
- **Aggregation**: the final clip prediction is computed via **majority vote across the 4 windows**, and the vote distribution is treated as a pseudo-probability score.

### ActionFormer (transformer TAL, via OpenTAD)

- **Input**: precomputed V-JEPA2 features extracted densely over the video timeline.
- **Core idea**: predict temporal segments and labels directly using a transformer-based detector.
- **Strengths**: learns temporal context and multi-scale localization better than window postprocessing.
- **Limitations**: requires feature extraction and detector training; higher overall complexity than window-based baselines.

## Fusion

Fusion is used to combine complementary information from RGB and skeleton-based models without training a heavy multimodal backbone.

### Classification fusion (clip-level)

We evaluate multiple late-fusion strategies operating on model logits/probabilities:

- **Scalar weighting (α)**: a single learned weight interpolates between RGB and skeleton logits.
- **Per-class weighting**: a separate α per class allows class-dependent reliance on RGB vs pose.
- **MLP fusion**: a small MLP combines concatenated logits nonlinearly.

To reduce overfitting to a single fold and to fairly evaluate the fusion layer, fusion is evaluated using **nested cross-validation** within each outer CV fold (inner folds used to fit fusion; held-out inner splits used to score it).

### TAL fusion (window-based)

For TAL window-based baselines, fusion is performed on **window-level class scores**:

- Uses an **MLP on log-probabilities** (product-of-experts style features).
- Uses grouped splits (by video) to prevent leakage of near-duplicate windows across train/val in fusion fitting.
- Handles missing modalities by falling back to a neutral distribution and including missing-indicator features.

## Action Recognition

### Clip-level classification (RMM type)

**Inputs:**
- **RGB clips** for V-JEPA2 and Qwen2.5-VL.
- **Pose-derived inputs** for PoseC3D/STGCN++ (heatmaps or graphs from keypoints).

**Protocol:**
- Evaluate using **3-fold CV** with **Leave-Child-Timepoint-Out (LCTO)** grouping.
- Report **clip-level** metrics

### Class imbalance strategies

Class imbalance is a first-order issue for clip classification (rare RMM types) and window-level detection (dominant background).

- **Class-weighted losses**:
  - Skeleton-based classifiers (PoseC3D / STGCN++) support inverse-frequency class weighting to reduce majority-class collapse.
- **Focal loss**:
  - For some skeleton-based runs, we also evaluated **focal loss** as an alternative to cross-entropy to emphasize harder / minority-class examples (requires tuning and is compared directly against cross-entropy baselines in the results).
- **Balanced sampling**:
  - For window-level detection, background can be downsampled and rare classes can be upsampled to increase minority-class learning signal.

## TAL

### Window-based

Window-based TAL has two stages:

1. **Window scoring**: train a window-level model that outputs class scores for each 2s window (with 1s stride).
2. **Post-processing to segments**: convert window scores into predicted segments using simple temporal rules (thresholding, optional smoothing, and merging nearby detections).

Key components:

- **Window-level models** can be RGB-based (V-JEPA2) or pose-based (PoseC3D/STGCN++).
- **Post-processing hyperparameters** (threshold/smoothing/merge-gap) are tuned via a small grid search; strong affect on segment boundaries and recall.
- **Fusion** is applied at the window-score level to combine RGB and pose evidence before post-processing

### Segment-based

Segment-based TAL is implemented with ActionFormer, a single-stage transformer detector that directly regresses action boundaries from dense video features.

#### Feature Extraction

Each video is converted into a fixed-rate feature sequence using a fine-tuned V-JEPA2 checkpoint:

- **Backbone checkpoint**: fine-tuned V-JEPA2 model from the window-level detection task.
- **Snippet extraction**: each video is broken into **non-overlapping 16-frame snippets** (stride = 16 frames).
- **Embedding**: for each snippet, V-JEPA2 produces a **1024-dimensional feature vector** (mean-pooled over patch embeddings).
- **Output format**: one feature file per video with shape `(T, 1024)`, where `T` is the number of snippets.

#### Model Architecture

ActionFormer processes the entire feature sequence at once through:

1. **Projection**: A 1D convolutional transformer that reduces features from 1024→512 dimensions and builds temporal context.
2. **Feature Pyramid Network (FPN)**: Produces multi-scale feature maps at 6 temporal resolutions (strides 1, 2, 4, 8, 16, 32), enabling detection of actions ranging from ~1 second to several minutes.
3. **Detection Head**: At each FPN level and temporal position, the model predicts:
   - **Classification scores** for each action class (4 RMM types or binary)
   - **Regression offsets** to the action's start and end times relative to that position

Each temporal position at each FPN scale produces independent proposals, resulting in thousands of raw candidate detections per video.

#### Post-Processing

Raw proposals are refined through multi-step filtering:

1. **Score thresholding**: Discard proposals with confidence below 0.001.
2. **Soft-NMS**: Suppress overlapping proposals with σ=0.5, using tIoU threshold of 0.1. Unlike hard NMS, soft-NMS gradually decreases scores of overlapping detections rather than removing them entirely.
3. **Top-k selection**: Keep at most 200 proposals per video after NMS.
4. **Segment voting** (threshold=0.7): Refine boundary predictions by aggregating nearby proposals.

The final output is a ranked list of `(start_time, end_time, class_label, confidence_score)` tuples for each video, evaluated against ground-truth annotations using temporal IoU-based metrics.
