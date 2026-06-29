# Data

The same underlying annotations support:

- **Action recognition / classification** on short, labeled clips.
- **Temporal Action Localization (TAL)** on full videos (predicting both *what* happened and *when* it happened).

## Class Definitions

### RMM classes (type labels)

Across the project, RMM segments/clips are labeled using the following behavior categories:

- **Hands Flapping**: Repetitive movement of the hands at the wrists either vertically or horizontally
- **One Hand Flap**: Repetitive movement of only one hand at the wrists either vertically or horizontally
- **Rocking**: Repetitive front-to-back or side-to-side movement of the body 
- **Spinning**: Repetitive turning of the body in a circular motion 
- **Jumping**: Repetitive bouncing of the entire body involving bending of the knees and feet leaving the floor or nearly leaving the floor

### Task variants

- **4-class (RMM type)**: `hands flapping`, `jumping`, `rocking`, `spinning`. The `one hand flap` label is is merged into hands flapping.
- **5-class (RMM type)**: adds `one hand flap` as its own class.
- **Binary**: `RMM` vs `background` (all RMM types collapsed into a single positive class).

## Annotation Results

### Clip-level labeled dataset (action recognition)

- **Total labeled clips**: **654** short clips derived from annotated RMM segments.
- **Grouping for leakage-safe splits**: **151 LCTO groups** (Leave-Child-Timepoint-Out grouping), where each group corresponds to a unique subject-timepoint combination.

### Full-video temporal annotations (TAL)

- TAL uses **ground-truth temporal segments** (start/end times for each RMM class) over full videos.
- For training/evaluation, the videos are converted into **sliding windows** (see split details below), with windows labeled based on overlap with ground-truth segments.

## Data Splits

### 4-class

#### Action recognition (clip-based)

Clip-level classification is evaluated using **3-fold cross-validation**:

| Split type | Train | Val |
| --- | --- | --- |
| **3-fold CV (per fold)** | 436 clips (~101 groups) | 218 clips (~50 groups) |

The **3-fold CV** is performed at the **LCTO-group** level to prevent leakage across subject/timepoint.

Class imbalance is substantial in the 4-class setting (example training distribution):

| Class | Train samples | Percentage |
| --- | --- | --- |
| hands flapping | 357 | 55.9% |
| jumping | 157 | 24.6% |
| rocking | 77 | 12.1% |
| spinning | 49 | 7.7% |

#### TAL (window-based)

For TAL, each full video is converted into overlapping windows:

- **Window length**: 2 seconds
- **Stride**: 1 second (50% overlap)
- **Labeling**: a class is marked positive for a window if the window overlaps a ground-truth segment with **tIoU ≥ 0.3**
- **Multi-label**: windows can contain multiple RMM labels (overlapping behaviors)
- **Background**: windows with no positive labels are background (≈90% of all windows)

Example 4-class CV window statistics:

| Split | Total windows | Background % | hands flapping | jumping | rocking | spinning |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| fold_0_train | 9,207 | 90.2% | 504 | 268 | 131 | 29 |
| fold_0_val | 4,938 | 90.8% | 295 | 94 | 47 | 25 |
| fold_1_train | 10,229 | 91.2% | 538 | 231 | 122 | 30 |
| fold_1_val | 3,916 | 88.4% | 261 | 131 | 56 | 24 |
| fold_2_train | 8,854 | 89.8% | 556 | 225 | 103 | 49 |
| fold_2_val | 5,291 | 91.5% | 243 | 137 | 75 | 5 |

### 5-class

#### Action recognition (clip-based)

The 5-class action recognition setting introduces **`one_hand_flap`** as a distinct label (in addition to hands flapping, jumping, rocking, and spinning). Splits follow the same LCTO-safe **3-fold CV** strategy, but the classification label space is expanded to 5 RMM types.

### Binary

Binary labels are derived by collapsing all RMM types into a single **RMM-positive** class:

- **Positive (RMM)**: any window/segment containing an RMM event (any type)
- **Negative (background)**: windows with no RMM labels

This setting reuses the same underlying split strategy as the TAL windows, but simplifies the label space to support **broad localization** (detect *where any RMM occurs*, regardless of type).

## Tracking + Pose Estimation

The dataset work relies on two supporting signals used for model inputs and preprocessing:

- **Person-of-interest tracking / cropping**:
  - Uses **SAM3** to track the child/person-of-interest and produce masks/crops.
  - The **COI (child/person of interest)** is manually specified per video for tracking.
- **Pose estimation**:
  - Uses **HRNet** pose estimation (fine-tuned on COCO-WholeBody).
  - Pose-derived keypoints support skeleton-based models (PoseC3D / STGCN++) and enable window-level pose coverage tracking.