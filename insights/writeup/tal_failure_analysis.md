# ActionFormer TAL Failure Analysis (Multiclass vs Binary)

This document analyzes failure modes for **ActionFormer** temporal action localization (TAL) models trained on V-JEPA2 features:

- **Multiclass**: 4 RMM types (hands flapping, jumping, rocking, spinning)
- **Binary**: RMM vs Background

It goes beyond aggregate mAP by decomposing *why* segments are missed (localization vs label confusion), and by correlating failures with preprocessing availability, video quality, and multi-person scenes.

## Executive Summary

**Reportable findings (fold-aware; score≥0.1; tIoU evaluated at 0.3/0.5/0.7):**

- **Binary improves detection at loose tIoU, but the advantage collapses at strict tIoU** (excluding invalid GT segments with duration ≤ 0):
  - **tIoU ≥ 0.3**: multiclass **66.9%**, binary **78.2%** (Δ **+11.3pp**)
  - **tIoU ≥ 0.5**: multiclass **42.5%**, binary **48.3%** (Δ **+5.8pp**)
  - **tIoU ≥ 0.7**: multiclass **22.4%**, binary **23.7%** (Δ **+1.3pp**)
  - **Interpretation**: binary mainly helps with *labeling/decision robustness*, not boundary tightness.

- **Multiclass misses split into three main mechanisms** (GT segments; tIoU≥0.3; score≥0.1):
  - **TP (correct label + localized)**: 313 (61.9%)
  - **Class confusion** (localized but wrong label): 58 (11.5%)
  - **Localization near-miss** (0 < tIoU < 0.3): 64 (12.6%)
  - **No overlap** (tIoU = 0): 71 (14.0%)

- **Binary’s extra hits are mostly “multiclass localized-but-wrong-label” cases**:
  - GT segments: **both**=296, **only binary**=70, **only multiclass**=17, **neither**=123
  - Among “only binary” segments, **68.6%** are cases where the multiclass model *localized* the segment (tIoU≥0.3) but assigned the *wrong label*.
  - **Actionable implication**: a **two-stage pipeline** (binary detection → multiclass type assignment) is well-motivated.

- **Rocking is the hardest multiclass class because it fails in both label and boundaries**:
  - Rocking GT breakdown: **36.5% TP**, **25.7% class confusion**, **25.7% near-miss**, **12.2% no-overlap**.

- **Multi-person scenes are a major external failure mode** (segment-level; tIoU≥0.3; score≥0.1; video ratings):
  - **Adult present**: multiclass recall **0.517** vs **0.649** (Δ **-13.3pp**); binary **0.610** vs **0.758** (Δ **-14.8pp**)
  - **Multiple children present**: multiclass Δ **-6.5pp**; binary Δ **-9.0pp**

- **mAP stays low because proposal explosion + weak score separation**:
  - At score≥0.1: **~79–80%** of predicted segments do **not** overlap any GT segment at tIoU≥0.3.
  - Score distributions for overlapping vs non-overlapping predictions are close (e.g., multiclass medians **0.035 vs 0.032**), implying poor ranking.

---

## 1. Evaluation Setup (Fold-Aware)

| Metric | Value |
|--------|-------|
| ActionFormer folds available | 3 (fold0, fold1, fold2) |
| Validation videos (union across folds) | 298 |
| GT videos within validation set | 224 |
| GT segments within validation set | 506 |
| Default score threshold used here | 0.1 |
| Default tIoU threshold used here | 0.3 (also analyzed at 0.5 and 0.7) |

**Important:** ActionFormer outputs are fold-specific (each JSON contains predictions for that fold’s validation videos). All analyses below are **fold-aware**: fold \(k\) predictions are evaluated only on fold \(k\) validation videos.

### 1.1 Ground Truth Class Distribution (Validation Set)

| Class | Segments | Videos |
|-------|----------|--------|
| Hands Flapping | 265 | 136 |
| Jumping | 134 | 70 |
| Rocking | 74 | 45 |
| Spinning | 33 | 21 |
| **Total** | **506** | **224** |

### 1.2 GT Segment Duration Artifacts (Annotation Reliability)

From `dataprep/rmm_segments.csv` restricted to validation-set videos:

- **Zero-duration segments**: 38 / 506 (**7.5%**)
  - Hands flapping: 29
  - Jumping: 9
- **Negative-duration segments**: 1 (jumping; start > end by 6s)
- Duration distribution:
  - **≤ 0s**: 7.5%
  - **≤ 1s**: 29.1%
  - **≤ 2s**: 56.5%
  - Median: 2.0s, Mean: 3.86s

**Why this matters:** duration≤0 segments are effectively unmatchable under tIoU-based scoring, creating unavoidable FNs and making strict-tIoU evaluation harsher.

---

## 2. Multiclass ActionFormer (4-class)

### 2.1 Fold-Level Segment Detection (TP/FP/FN)

Segment-level one-to-one matching, score≥0.1, tIoU≥0.3:

| Fold | Val videos | GT segments | TP | FP | FN | Precision | Recall |
|------|------------|------------|----|----|----|-----------|--------|
| 0 | 101 | 155 | 94 | 1030 | 61 | 8.36% | 60.65% |
| 1 | 94 | 179 | 122 | 1129 | 57 | 9.75% | 68.16% |
| 2 | 103 | 172 | 96 | 1418 | 76 | 6.34% | 55.81% |
| **All** | **298** | **506** | **312** | **3577** | **194** | **8.02%** | **61.66%** |

### 2.2 Miss Decomposition (Why Multiclass Fails)

For each GT segment, we find the best-overlapping prediction (score≥0.1) and categorize the outcome:

| Outcome | Count | Rate |
|---------|-------|------|
| **TP** (correct label + tIoU≥0.3) | 313 | 61.9% |
| **Class confusion** (wrong label but tIoU≥0.3) | 58 | 11.5% |
| **Localization near-miss** (0 < tIoU < 0.3) | 64 | 12.6% |
| **No overlap** (tIoU = 0) | 71 | 14.0% |

### 2.3 Confusion Patterns (Localized but Wrong Label)

Top localized confusion pairs (tIoU≥0.3, score≥0.1):

- **rocking → hands flapping** (most frequent)
- **hands flapping ↔ jumping** (bidirectional confusion)

Row-normalized view (best-overlap label; “none” = not localized at tIoU≥0.3):

| GT \\ Pred | hands flapping | jumping | rocking | spinning | none |
|-----------|----------------|---------|---------|----------|------|
| hands flapping | 50.6% | 16.6% | 5.3% | 0.8% | 26.8% |
| jumping | 35.1% | 38.1% | 5.2% | 0.7% | 20.9% |
| rocking | 36.5% | 0.0% | 23.0% | 2.7% | 37.8% |
| spinning | 9.1% | 3.0% | 12.1% | 51.5% | 24.2% |

### 2.4 Rocking Is a Compound Failure (Label + Boundary)

Reason breakdown by class (percent of GT segments in each category):

| Class | TP | Class confusion | Near-miss | No overlap |
|-------|----|-----------------|----------|------------|
| hands flapping | 67.2% | 6.0% | 9.8% | 17.0% |
| jumping | 66.4% | 12.7% | 9.0% | 11.9% |
| **rocking** | **36.5%** | **25.7%** | **25.7%** | 12.2% |
| spinning | 57.6% | 18.2% | 21.2% | 3.0% |

---

## 3. Binary ActionFormer (RMM vs BG)

Binary removes inter-class confusion, so it is a better “detector” but cannot provide RMM type.

### 3.1 Fold-Level Segment Detection (TP/FP/FN)

Segment-level one-to-one matching, score≥0.1, tIoU≥0.3:

| Fold | Val videos | GT segments | TP | FP | FN | Precision | Recall |
|------|------------|------------|----|----|----|-----------|--------|
| 0 | 101 | 155 | 102 | 664 | 53 | 13.32% | 65.81% |
| 1 | 94 | 179 | 137 | 764 | 42 | 15.21% | 76.54% |
| 2 | 103 | 172 | 119 | 970 | 53 | 10.93% | 69.19% |
| **All** | **298** | **506** | **358** | **2398** | **148** | **12.99%** | **70.75%** |

---

## 4. Multiclass vs Binary: What Actually Changes?

### 4.1 Cross-Model Segment Outcomes (GT Segments)

| Outcome | Count |
|---------|-------|
| Detected by both | 296 |
| **Only binary detects** | **70** |
| Only multiclass detects | 17 |
| Detected by neither | 123 |

Among the “only binary” segments, **68.6%** are cases where the multiclass model **localized** the segment (tIoU≥0.3) but assigned the **wrong label**.

---

## 5. Localization Sensitivity (tIoU Threshold)

Using score≥0.1, and excluding duration≤0 GT segments:

| tIoU | Multiclass recall | Binary recall | Δ |
|------|-------------------|--------------|---|
| ≥ 0.3 | 66.9% | 78.2% | +11.3pp |
| ≥ 0.5 | 42.5% | 48.3% | +5.8pp |
| ≥ 0.7 | 22.4% | 23.7% | +1.3pp |

---

## 6. Duration as a Hidden Driver (Short & Long Segments)

Detection rate by GT duration bin (tIoU≥0.3, score≥0.1):

| Duration bin | n | Multiclass | Binary |
|--------------|---|------------|--------|
| ≤ 0 | 38 | 0.0% | 0.0% |
| (0, 1] | 109 | 56.0% | 70.6% |
| (1, 2] | 139 | 67.6% | 81.3% |
| (2, 5] | 128 | 74.2% | 81.3% |
| (5, 10] | 55 | 72.7% | 81.8% |
| > 10 | 37 | 62.2% | 73.0% |

---

## 7. Proposal Explosion & Weak Calibration (Why mAP Is Low)

At score≥0.1 and tIoU≥0.3 (over all validation videos):

- **Multiclass** predicted 3,889 segments; **3,064 (78.8%)** do not overlap any GT segment at tIoU≥0.3.
- **Binary** predicted 2,756 segments; **2,206 (80.0%)** do not overlap any GT segment at tIoU≥0.3.

Background-only videos (no GT segments) still receive many predictions:

- Multiclass: 937 predicted segments in no-GT videos
- Binary: 703 predicted segments in no-GT videos

---

## 8. External Failure Modes (Video Ratings)

Using `SAILS_RATINGS_ALL_8.8.25.xlsx` (fold-aware; segment-level; tIoU≥0.3; score≥0.1):

| Factor | Multiclass Δ | Binary Δ |
|--------|--------------|----------|
| **Adult present** | **-13.3pp** | **-14.8pp** |
| Multiple children present | -6.5pp | -9.0pp |
| Child of interest unclear | -1.0pp | -4.4pp |

---

## 9. Concrete Recommendations (ActionFormer-Specific)

- **Two-stage design**: binary for detection + multiclass for type assignment (since most binary-only gains are multiclass label confusions).
- **Post-processing for proposal explosion**: tune score threshold / top-K / NMS to reduce the ~80% non-overlap proposal rate.
- **Boundary-aware improvement**: localization degrades sharply from tIoU 0.3→0.7; boundary tightness is a core limitation.
- **Handle annotation artifacts**: fix or filter duration≤0 GT segments; consider minimum-duration constraints.
- **Multi-person robustness**: improve crop/tracking or incorporate “other people” suppression when adults/other children are present.

---

## Appendix: Data Sources & Outputs

- **ActionFormer Multiclass predictions**: `OpenTAD/exps/sails_rmm/actionformer_vjepa_balanced_fold{0,1,2}/gpu1_id99/result_detection.json`
- **ActionFormer Binary predictions**: `OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold{0,1,2}/gpu1_id99/result_detection.json`
- **Fold definitions (validation videos)**: `dataprep/tal/splits_cv_4class/fold_{0,1,2}_val_windows.csv`
- **Ground truth segments**: `dataprep/rmm_segments.csv`
- **Video ratings**: `/orcd/data/satra/002/datasets/SAILS/data4analysis/Video Rating Data/SAILS_RATINGS_ALL_8.8.25.xlsx`
- **Fold-aware ActionFormer summary JSON**: `insights/tables/actionformer_failure_analysis.json`
