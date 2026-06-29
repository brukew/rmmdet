# Tasks

We cover three closely-related tasks moving from clip-level recognition to full-video localization. Below is a summary of **what each task asks the model to do** and **which metrics are used** (metric definitions are detailed in `writeup/METRICS.md`).

## Action Recognition

### Clip-level classification (RMM type)

**Goal:** Given a short clip containing an annotated RMM segment, predict the RMM **type**.

**Label variants:**
- **4-class:** hands flapping (incl. one-hand flap merged), jumping, rocking, spinning
- **5-class:** adds one hand flap as a distinct class

**Evaluation setup:** 3-fold cross-validation with **Leave-Child-Timepoint-Out (LCTO)** grouping to avoid leakage across subject/timepoint.

**Metrics reported (clip-level and video-level):**
- **Top-1 / Top-2 Accuracy**
- **Macro F1**
- **Macro Precision**
- **Macro Recall**
- **Cohen’s κ** (agreement beyond chance)

### Window-level detection (includes background)

**Goal:** Given a short **fixed-length window** sampled from a full video, predict whether it contains an RMM and, if so, which **RMM type**. This is the “detection” formulation used to bridge full-video annotations to window-based baselines.

**Labels:** 5-class window classification including **background**:
- background, hands flapping, jumping, rocking, spinning

**Notes:**
- The window-level detection setting is **highly imbalanced** (background windows dominate).
- This task produces per-window scores that can be postprocessed into segments for TAL.

**Metrics reported:**
- **Top-1 / Top-2 Accuracy**
- **Macro F1** and **per-class F1** (to quantify rare-class behavior)
- **Macro Precision**
- **Macro Recall**
- **Cohen’s κ**

## Temporal Action Localization

### Segment-level TAL (multi-class localization)

**Goal:** Given a full video, output a set of predicted **temporal segments** \([start, end]\) with an associated **RMM class label** (hands flapping, jumping, rocking, spinning). A correct prediction requires both:

- **Correct class**, and
- **Accurate temporal boundaries** (sufficient overlap with ground truth).

**Primary metric:** **mAP at temporal IoU (tIoU)** thresholds.
- **Recall@tIoU**

**Notes:**
- Background is not treated as an evaluated class in multi-class TAL mAP (mAP is over RMM classes).

### Binary TAL (broad localization: RMM vs background)

**Goal:** Detect **where any RMM occurs** in a video, without requiring correct RMM subtype classification. This collapses all RMM types into a single positive class.

**Metrics reported:**
- **mAP@tIoU** (same style as multi-class TAL, but for a single positive class)
- **Recall@tIoU** for window-based binary baselines (useful for broad event coverage)
- (For proposal-based transformer evaluation) **Top-k recall (R@k) at tIoU thresholds**