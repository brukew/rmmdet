# TAL (Temporal Action Localization) Data Splits

This directory contains window-level splits for temporal action localization of RMM behaviors.

## Overview

- **Window length**: 2 seconds
- **Window stride**: 1 second (50% overlap)
- **Label format**: Multi-label (windows can have multiple RMM classes)
- **tIoU threshold**: 0.3 (for positive label assignment)
- **Time semantics**: Half-open intervals `[start, end)` with GT segments converted via `end_exclusive = end_sec + 1.0`

## Directory Structure

```
tal/
├── README.md                           # This file
├── label_maps/
│   ├── label_map_4class.json           # 4-class label mapping
│   └── label_map_5class.json           # 5-class label mapping
├── metadata_4class_cv.json             # Generation metadata + statistics
├── metadata_4class_single.json
├── metadata_5class_cv.json
├── metadata_5class_single.json
├── splits_cv_4class/                   # 4-class 3-fold CV windows
│   ├── fold_0_train_windows.csv
│   ├── fold_0_val_windows.csv
│   ├── fold_1_train_windows.csv
│   ├── fold_1_val_windows.csv
│   ├── fold_2_train_windows.csv
│   └── fold_2_val_windows.csv
├── splits_cv_5class/                   # 5-class 3-fold CV windows
│   └── ...
├── splits_single_4class/               # 4-class single split windows
│   ├── train_windows.csv
│   ├── val_windows.csv
│   └── test_windows.csv
├── splits_single_5class/               # 5-class single split windows
│   └── ...
├── video_assignment/                   # Video-level split assignments
│   ├── video_to_split_4class_cv.csv
│   ├── video_to_split_4class_single.csv
│   ├── video_to_split_5class_cv.csv
│   └── video_to_split_5class_single.csv
├── make_tal_video_assignments.py       # Script: derive video-to-split mapping
├── make_tal_window_splits.py           # Script: generate window-level splits
├── validate_tal_splits.py              # Script: validation and sanity checks
└── cut_tal_windows.py                  # Script: (optional) cut window clips
```

## Window CSV Schema

Each window CSV contains these columns:

| Column | Description |
|--------|-------------|
| `window_id` | Unique identifier: `{child_id}_{video_stem}__t{start_ms}_{end_ms}` |
| `video_key` | Normalized video file path (for matching) |
| `video_file` | Original video file path from splits |
| `filename` | Video filename only |
| `child_id` | Child/subject identifier |
| `video_path` | Full standardized mp4 path |
| `fps` | Video frame rate |
| `duration_sec` | Total video duration |
| `start_sec` | Window start time (half-open) |
| `end_sec` | Window end time (half-open) |
| `labels` | JSON list of class indices, e.g., `[0, 2]` |
| `primary_label` | Single-label for compatibility (-1 = background) |
| `is_background` | 1 if no positive labels, 0 otherwise |
| `tiou_*` | Per-class max tIoU with GT segments |
| `mask_severity` | Coverage quality: `full`, `mild` (1-5 frames missing), `moderate` (6-20 frames), `severe` (>20) |
| `mask_cause` | Why frames missing: `full`, `sparse`, `exceeds_cache`, `no_data` |
| `mask_frames_missing` | Number of frames (out of ~60) without SAM3 mask data |
| `pose_severity` | Coverage quality: `full`, `mild` (1-5 frames missing), `moderate` (6-20 frames), `severe` (>20 frames) |
| `pose_cause` | Why frames missing: `full`, `sparse`, `exceeds_cache`, `no_data` |
| `pose_frames_missing` | Number of frames (out of ~60) without HRNet pose data |

### Mask/Pose Availability

- **Cause categories**:
  - `full`: All frames have data
  - `sparse`: Gaps in coverage (missing mask/pose for some frames)
  - `exceeds_cache`: Window falls beyond the cache limit
  - `no_data`: No cache file found for this video
- **Pyskl requirement**: Windows need ≥33% of frames with pose data for skeleton-based models
- **Impact**: Windows with `pose_severity='severe'` are skipped during pickle generation (~10% train, ~27% val for fold 0)

## Class Labels

### 4-Class
| Index | Label |
|-------|-------|
| 0 | hands flapping |
| 1 | jumping |
| 2 | rocking |
| 3 | spinning |

### 5-Class
| Index | Label |
|-------|-------|
| 0 | hands flapping |
| 1 | jumping |
| 2 | one hand flap |
| 3 | rocking |
| 4 | spinning |

## Labeling Rule

For each window and each class:
1. Compute max tIoU between the window and all GT segments of that class
2. If `max_tIoU >= 0.3`, mark class as positive
3. `is_background = 1` only if no class passes threshold
4. `primary_label` = class with highest max tIoU (ties broken by priority order)

## Statistics (4-class CV)

| Split | Total Windows | Background % | hands flapping | jumping | rocking | spinning |
|-------|---------------|--------------|----------------|---------|---------|----------|
| fold_0_train | 9,207 | 90.2% | 504 | 268 | 131 | 29 |
| fold_0_val | 4,938 | 90.8% | 295 | 94 | 47 | 25 |
| fold_1_train | 10,229 | 91.2% | 538 | 231 | 122 | 30 |
| fold_1_val | 3,916 | 88.4% | 261 | 131 | 56 | 24 |
| fold_2_train | 8,854 | 89.8% | 556 | 225 | 103 | 49 |
| fold_2_val | 5,291 | 91.5% | 243 | 137 | 75 | 5 |

## Usage

### Generate Splits (already done)

```bash
# Generate video assignments
python make_tal_video_assignments.py --task 4class --mode cv

# Generate window splits
python make_tal_window_splits.py --task 4class --mode cv

# Validate
python validate_tal_splits.py --task 4class --mode cv
```

### Cut Window Clips (optional)

If you need physical clip files on disk:

```bash
# Cut all training windows
python cut_tal_windows.py \
    --csv splits_single_4class/train_windows.csv \
    --output-dir /path/to/clips \
    --jobs 8

# Cut only positive windows (for quick debugging)
python cut_tal_windows.py \
    --csv splits_single_4class/train_windows.csv \
    --output-dir /path/to/positive_clips \
    --filter-positive \
    --max-clips 100
```

### Loading in Python

```python
import csv
import json
from pathlib import Path

def load_tal_windows(csv_path):
    """Load TAL window CSV."""
    windows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            window = dict(row)
            window['start_sec'] = float(row['start_sec'])
            window['end_sec'] = float(row['end_sec'])
            window['labels'] = json.loads(row['labels'])
            window['is_background'] = int(row['is_background']) == 1
            windows.append(window)
    return windows

# Example usage
train_windows = load_tal_windows('splits_single_4class/train_windows.csv')
print(f"Total windows: {len(train_windows)}")
print(f"Background: {sum(w['is_background'] for w in train_windows)}")
```

## Relationship to Classification Splits

These TAL splits are derived from the existing classification splits in `actreg/dataprep/splits/`:

1. **Video assignments** are derived from existing segment splits (LCTO-safe groupings preserved)
2. **GT segments** are loaded from the same source CSVs
3. **Windows** cover the full video timeline with consistent 2s/1s windowing

The classification clips (654 segments) correspond to the **positive windows** with high tIoU, but TAL splits include all windows including background.

## Pose pickle generation (pyskl) notes

- **Why windows get skipped**: PoseC3D/ST-GCN pickle generation skips windows with **too few pose frames** in the window (effectively “no usable pose”).
- **How to predict skips from the CSV**:
  - Skipped windows fall almost entirely into `pose_severity=severe` with `pose_cause` in `{exceeds_cache, sparse, no_data}`.
  - In practice, `pose_cause=exceeds_cache` dominates skips (windows beyond the SAM3/pose cache coverage).
- **Observed skip rates (CV 4-class, fold 0)**:
  - `fold_0_train`: **948/9207 (10.3%)** skipped; **RMM lost**: **15/901 (1.7%)**
  - `fold_0_val`: **1346/4938 (27.3%)** skipped; **RMM lost**: **23/454 (5.1%)**
  - Skips are **mostly background windows**; RMM coverage remains high.

## Notes

- Background windows (~90%) dominate; consider class-balanced sampling for training
- Multi-label windows exist (2-4% of positive windows) where classes overlap
- The `primary_label` column provides a single-label fallback for simpler models
- Window IDs are deterministic and reproducible based on timestamps

## See Also

- **TAL Evaluation Pipeline**: See `actreg/tal/README.md` for mAP evaluation scripts
- **V-JEPA Training**: `actreg/v-jepa/finetune_sails_vjepa2_tal.py`
- **pyskl Training**: `actreg/pyskl/tools/train_weighted.py`
