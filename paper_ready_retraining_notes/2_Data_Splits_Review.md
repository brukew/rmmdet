# Data Splits Review

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total RMM segments | 654 |
| Unique children | 99 |
| Unique videos | 298 |
| Unique LCTO groups (child × timepoint) | 151 |
| CV folds | 3 |
| Grouping strategy | `GroupKFold` on LCTO group |
| Timepoints present | 14_month, 36_month, unknown |

Children per timepoint:
- 14_month: 72 children, 286 segments
- 36_month: 68 children, 307 segments
- unknown: 11 children, 61 segments

---

## Class Distribution (Overall)

| Class | Count | % | Ratio to smallest |
|-------|-------|---|-------------------|
| hands flapping | 343 | 52.4% | 7.6× |
| jumping | 170 | 26.0% | 3.8× |
| rocking | 96 | 14.7% | 2.1× |
| spinning | 45 | 6.9% | 1.0× |

**Key concern:** Heavy imbalance. Hands flapping alone is >50% of the dataset; spinning has only 45 segments total.

---

## Per-Fold Split Sizes

### Segments

| Fold | Train segs | Val segs | Train vids | Val vids | Train LCTO | Val LCTO |
|------|-----------|---------|-----------|---------|-----------|---------|
| 0 | 436 | 218 | 197 | 101 | 101 | 50 |
| 1 | 436 | 218 | 204 | 94 | 101 | 50 |
| 2 | 436 | 218 | 195 | 103 | 100 | 51 |

### Windows (2s window, 1s stride)

| Fold | Train wins | Val wins | Train BG% | Val BG% | Train FG | Val FG |
|------|-----------|---------|----------|--------|---------|--------|
| 0 | 9,207 | 4,938 | 90.2% | 90.8% | 901 | 454 |
| 1 | 10,229 | 3,916 | 91.2% | 88.4% | 902 | 453 |
| 2 | 8,854 | 5,291 | 89.8% | 91.5% | 907 | 448 |

**Note:** ~90% of windows are background across all folds. Foreground window counts are roughly balanced across folds (~450 val FG each).

### Val Class Distribution Per Fold

| Class | Fold 0 | Fold 1 | Fold 2 |
|-------|--------|--------|--------|
| hands flapping | 127 (58.3%) | 113 (51.8%) | 103 (47.2%) |
| jumping | 45 (20.6%) | 65 (29.8%) | 60 (27.5%) |
| rocking | 31 (14.2%) | 24 (11.0%) | 41 (18.8%) |
| spinning | 15 (6.9%) | 16 (7.3%) | 14 (6.4%) |

**Note:** No class stratification was used in GroupKFold. Class proportions shift noticeably across folds (e.g., rocking ranges from 11.0%–18.8% of val).

---

## Segment Duration Statistics

| Class | N | Mean | Median | Min | Max | Std |
|-------|---|------|--------|-----|-----|-----|
| hands flapping | 343 | 2.6s | 2.0s | **-13.0s** | 69.0s | 4.6s |
| jumping | 170 | 3.0s | 2.0s | **-6.0s** | 17.0s | 3.0s |
| rocking | 96 | 6.1s | 3.0s | 0.0s | 49.0s | 7.4s |
| spinning | 45 | 10.0s | 7.0s | 0.0s | 37.0s | 9.4s |
| **ALL** | **654** | **3.7s** | **2.0s** | **-13.0s** | **69.0s** | **5.7s** |

Duration percentiles (all classes): p5=0.0s, p10=0.0s, p25=1.0s, p50=2.0s, p75=4.0s, p90=8.0s, p95=12.4s

### Duration buckets

| Bucket | Count | % |
|--------|-------|---|
| < 0s (INVALID) | 2 | 0.3% |
| = 0s (point annotations) | 65 | 9.9% |
| 1s | 144 | 22.0% |
| 2s | 173 | 26.5% |
| 3–5s | 135 | 20.6% |
| 5–10s | 82 | 12.5% |
| 10–20s | 39 | 6.0% |
| > 20s | 14 | 2.1% |

---

## Data Quality Issues

### 1. Negative-duration segments (2 segments)

| Segment ID | Class | Start | End | Duration |
|------------|-------|-------|-----|----------|
| N3L7A1I2B9_14_month_154_2_1 | jumping | 158.0 | 152.0 | -6.0s |
| M5F7G8S4F0_14_month_128_0_1 | hands flapping | 74.0 | 61.0 | -13.0s |

**Impact:** These are annotation errors (start > end). They produce invalid GT in OpenTAD JSON and confuse tIoU computation.  
**Action needed:** Swap start/end, or drop them.

### 2. Zero-duration segments (65 segments = 9.9% of data)

Breakdown by class:
- hands flapping: 46/343 (13.4%)
- jumping: 17/170 (10.0%)
- spinning: 1/45 (2.2%)
- rocking: 1/96 (1.0%)

**Impact:** With the `end_sec + 1.0` half-open convention used throughout the codebase, these become 1-second segments `[t, t+1)`. This is handled consistently, so they are functional. However, they represent instantaneous annotations — these are inherently noisier ground truth because the annotator marked a single second.

### 3. Time annotation granularity

All `start_sec` and `end_sec` are integer seconds. The `end_sec + 1.0` convention converts from closed `[start, end]` to half-open `[start, end+1)`. This means:
- `start=2, end=4` → segment covers `[2, 5)` = 3 seconds
- `start=8, end=8` → segment covers `[8, 9)` = 1 second

This is applied consistently across: `tal_map_eval.py`, `eval_two_stage_tal.py`, `convert_cv_splits_to_opentad_json.py`, `make_tal_window_splits.py`, `validate_tal_splits.py`.

---

## Child-Level Leakage Across Folds

The grouping key is LCTO = `(child_id, timepoint)`, **not** `child_id` alone. This means the same child at different timepoints can appear in different folds:

| Fold | Train children | Val children | Overlapping children |
|------|---------------|-------------|---------------------|
| 0 | 80 | 44 | 25 |
| 1 | 76 | 46 | 23 |
| 2 | 77 | 45 | 23 |

Example: Child `C1G1X2G4C6` — 36_month in train, 14_month in val (Fold 0).

**Implication:** A model could potentially learn child-specific appearance cues (e.g., clothing, home environment, body shape) from training data and exploit them at validation time. Whether this matters depends on how visually similar videos of the same child are across timepoints (14mo vs 36mo sessions are months apart, so appearance changes substantially).

**This is a known design decision, not a bug.** Grouping on `child_id` alone would reduce the effective number of groups from 151 to 99, making 3-fold CV less stable.

---

## File Locations

| Asset | Path |
|-------|------|
| Segment splits (4-class) | `dataprep/splits/cv_splits_4class/fold_{0,1,2}_{train,val}.csv` |
| Segment splits (5-class) | `dataprep/splits/cv_splits/fold_{0,1,2}_{train,val}.csv` |
| Split metadata | `dataprep/splits/cv_splits_4class/split_metadata.json` |
| Video assignments | `dataprep/tal/video_assignment/video_to_split_4class_cv.csv` |
| Window splits (4-class) | `dataprep/tal/splits_cv_4class/fold_{0,1,2}_{train,val}_windows.csv` |
| Window splits (5-class) | `dataprep/tal/splits_cv_5class/fold_{0,1,2}_{train,val}_windows.csv` |
| OpenTAD annotations | `OpenTAD/data/sails_rmm/annotations/fold{0,1,2}_anno.json` (generated) |
| Split generation notebook | `dataprep/clip_gen/rmm_annot.ipynb` |
| Video assignment script | `dataprep/tal/make_tal_video_assignments.py` |
| Window split script | `dataprep/tal/make_tal_window_splits.py` |
| OpenTAD conversion script | `OpenTAD/tools/prepare_data/sails_rmm/convert_cv_splits_to_opentad_json.py` |
| Split validation script | `dataprep/tal/validate_tal_splits.py` |

---

## Data Flow

```
rmm_annot.ipynb (Excel → segment CSVs)
    ↓
dataprep/splits/cv_splits_4class/fold_X_{train,val}.csv  (654 segments, 151 LCTO groups)
    ↓
make_tal_video_assignments.py
    ↓
dataprep/tal/video_assignment/video_to_split_4class_cv.csv  (298 videos)
    ↓
make_tal_window_splits.py  (2s window, 1s stride)
    ↓
dataprep/tal/splits_cv_4class/fold_X_{train,val}_windows.csv  (~14k windows/fold)
    ↓
convert_cv_splits_to_opentad_json.py
    ↓
OpenTAD/data/sails_rmm/annotations/fold{N}_anno.json
```

---

## Summary of Things to Deal With

1. **2 negative-duration segments** — must fix or drop before retraining
2. **65 zero-duration segments (→ 1s with half-open)** — functional but noisy; consider whether to keep, flag, or weight differently
3. **7.6× class imbalance** (hands flapping vs spinning) — need balanced sampling or class weighting for all training stages
4. **~23-25 children shared across train/val per fold** (at different timepoints) — known design choice; document as limitation
5. **No class stratification in GroupKFold** — val class proportions shift across folds; per-class metrics will have different statistical power per fold
6. **~90% background windows** — binary detector training needs negative sampling strategy
