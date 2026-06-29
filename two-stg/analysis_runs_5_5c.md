# Deep Analysis: TriDet Two-Stage TAL (Runs 5 & 5c)

Comprehensive analysis of TriDet as a Stage 1 detector in the two-stage temporal action localization pipeline, compared to ActionFormer. All metrics are 3-fold cross-validation unless otherwise noted.

**Runs compared:**

| Run | Stage 1 | Stage 2 | avg_mAP |
|-----|---------|---------|---------|
| **1** | ActionFormer Binary | V-JEPA2 4-class | 17.83% |
| **5** | TriDet Binary | V-JEPA2 4-class | 18.65% |
| **5c** | TriDet Binary | 3-way MLP (V-JEPA2 + live PoseC3D/STGCN++) | 16.67% |
| — | ActionFormer Balanced | end-to-end 4-class (baseline) | 16.70% |

---

## 1. Per-Class Average Precision

### 1a. Per-class AP at key tIoU thresholds (3-fold CV mean ± std)

**tIoU = 0.3:**

| Class | Run 1 (AF+VJEPA) | Run 5 (TD+VJEPA) | Run 5c (TD+3way) |
|-------|-------------------|-------------------|-------------------|
| hands flapping | 40.1% ± 4.8% | 38.2% ± 5.5% | 28.5% ± 1.5% |
| jumping | 25.3% ± 15.5% | 24.8% ± 15.8% | 33.5% ± 6.9% |
| spinning | 35.9% ± 2.2% | 31.9% ± 7.5% | 21.9% ± 2.1% |
| rocking | 5.2% ± 2.0% | 10.3% ± 6.5% | 12.4% ± 6.6% |

**tIoU = 0.5:**

| Class | Run 1 (AF+VJEPA) | Run 5 (TD+VJEPA) | Run 5c (TD+3way) |
|-------|-------------------|-------------------|-------------------|
| hands flapping | 26.2% ± 3.4% | 26.1% ± 3.3% | 18.9% ± 1.6% |
| jumping | 23.3% ± 15.7% | 20.7% ± 15.9% | 25.5% ± 10.6% |
| spinning | 25.8% ± 2.1% | 24.6% ± 8.4% | 16.2% ± 3.9% |
| rocking | 3.2% ± 0.7% | 5.6% ± 3.3% | 8.3% ± 5.7% |

**tIoU = 0.7:**

| Class | Run 1 (AF+VJEPA) | Run 5 (TD+VJEPA) | Run 5c (TD+3way) |
|-------|-------------------|-------------------|-------------------|
| hands flapping | 7.8% ± 0.8% | 12.4% ± 0.3% | 8.7% ± 2.0% |
| jumping | 12.5% ± 6.9% | 15.2% ± 12.7% | 17.3% ± 8.9% |
| spinning | 19.2% ± 8.3% | 10.6% ± 3.1% | 4.4% ± 0.9% |
| rocking | 1.1% ± 1.0% | 3.1% ± 2.7% | 3.2% ± 2.1% |

### 1b. TriDet vs ActionFormer class-level delta (Run 5 − Run 1, CV mean)

| Class | Δ mAP@0.3 | Δ mAP@0.5 | Δ mAP@0.7 | Summary |
|-------|-----------|-----------|-----------|---------|
| hands flapping | −1.9 pp | −0.1 pp | **+4.6 pp** | Boundaries improve; coarse detection slightly worse |
| jumping | −0.5 pp | −2.6 pp | **+2.7 pp** | Same pattern: tight tIoU gains, loose tIoU loss |
| spinning | −4.0 pp | −1.2 pp | **−8.7 pp** | Worse at all thresholds (small sample size) |
| rocking | **+5.1 pp** | **+2.4 pp** | **+2.0 pp** | Consistent improvement — hardest class benefits most |

### 1c. Run 5 vs Run 5c: effect of 3-way fusion vs V-JEPA2-only (CV mean)

| Class | Run 5 (VJEPA only) | Run 5c (3-way) | Δ | Interpretation |
|-------|---------------------|-----------------|---|----------------|
| hands flapping @0.3 | 38.2% | 28.5% | −9.7 pp | Skeleton hurts dominant class |
| jumping @0.3 | 24.8% | 33.5% | +8.7 pp | Kinematic signal helps |
| rocking @0.3 | 10.3% | 12.4% | +2.1 pp | Slight skeleton benefit |
| spinning @0.3 | 31.9% | 21.9% | −10.0 pp | Skeleton hurts (sparse data) |

The 3-way fusion redistributes probability mass. Jumping and rocking (distinctive kinematics) benefit; hands flapping and spinning (visual-dominant) are hurt. The net effect is negative due to MLP distribution mismatch (trained on GT-clip skeleton scores, applied to noisier proposal-level scores).

---

## 2. Per-Video Analysis

### 2a. Aggregate video-level statistics (Run 5, 229 val videos across 3 folds)

Run 5 and 5c share the same TriDet proposals (Stage 1), so proposal-level stats are identical.

| Metric | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| GT segments per video | 2.2 | 2.0 | 1 | 14 |
| Predictions per video | 78.5 | 62.0 | 8 | 200 |
| Prediction:GT ratio | 45:1 | 34:1 | 6:1 | 200:1 |
| Proposal recall @tIoU=0.3 | 98.3% | 100% | 0% | 100% |
| Proposal recall @tIoU=0.5 | 91.2% | 100% | 0% | 100% |
| Video duration (sec) | 48 | 32 | 2 | 1037 |

Per-class video-level recall@0.3 (mean over videos containing that GT class):

| Class | Recall@0.3 | # Videos |
|-------|------------|----------|
| hands flapping | 98.2% | 138 |
| jumping | 97.2% | 72 |
| spinning | 99.5% | 22 |
| rocking | 96.7% | 46 |

Stage 1 recall is not the bottleneck — nearly all GT segments are covered by at least one proposal at tIoU=0.3. The challenge is Stage 2 classification accuracy and the 45:1 false positive ratio.

### 2b. Videos with 0% recall at tIoU=0.3 (completely missed GT segments)

Both Run 5 and Run 5c miss 2/229 videos entirely:

| Video | Fold | GT segs | GT class | # Preds | Notes |
|-------|------|---------|----------|---------|-------|
| `L4K0P3T4Q8_2020-03-26_19-26-58_287` | 0 | 1 | hands flapping | 41 | Single short GT segment; proposals exist but miss it |
| `C5L2D6N1W6_12-25-19` | 1 | 1 | jumping | 37 | Same pattern — proposals present, none overlap GT |

Both are single-segment videos where the detector generates proposals that don't land on the GT. These warrant manual review for annotation quality.

### 2c. Hardest videos (lowest recall@0.3 across Run 5 & 5c)

| Video | Fold | GT segs | Classes | Recall |
|-------|------|---------|---------|--------|
| `C5L2D6N1W6_12-25-19` | 1 | 1 | jumping | 0% |
| `L4K0P3T4Q8_2020-03-26_19-26-58_287` | 0 | 1 | hands flapping | 0% |
| `G4J8F6F4X0_Snapchat-1467744963` | 1 | 3 | mixed | 33% |
| `L4K0P3T4Q8_2017-12-08_12-33-13` | 0 | 2 | mixed | 50% |
| `E2P5Q8S5A7_received_-689085498 (1)` | 0 | 3 | mixed | 67% |
| `N3L7A1I2B9_20200709_082328` | 2 | 8 | mixed | 88% |
| `L0C6T6H2C6_03-08-2021` | 1 | 9 | mixed | 89% |
| `C1G1X2G4C6_IMG_7585` | 1 | 11 | mixed | 91% |

Pattern: the hardest videos tend to have few GT segments (1–3), where a single miss devastates recall. Videos with many GT segments (8–14) maintain high recall even when some are missed.

---

## 3. Stage 1 Proposal Quality: TriDet vs ActionFormer

### 3a. Proposal statistics (3-fold summary)

| Metric | ActionFormer | TriDet | Δ |
|--------|-------------|--------|---|
| Total proposals (3 folds) | 26,999 | 24,095 | −2,904 (−10.8%) |
| Mean proposals/video | 91 | 81 | −10 |
| Mean score | 0.039 | 0.032 | −0.007 |
| Median score | 0.017 | 0.012 | −0.005 |
| Max score | 0.620 | 0.727 | +0.107 |
| p95 score | 0.159 | 0.128 | −0.031 |
| Mean duration (sec) | 4.4 | 4.6 | +0.2 |
| Score > 0.1 | 2,755 (10.2%) | 1,703 (7.1%) | −1,052 |
| Score > 0.3 | 320 (1.2%) | 315 (1.3%) | −5 |
| Score > 0.5 | 20 (0.1%) | 66 (0.3%) | +46 |

TriDet generates ~11% fewer proposals with lower average scores but a wider dynamic range (higher max, fewer mid-range scores). The number of high-confidence proposals (score > 0.5) is 3× higher with TriDet, suggesting better calibration of top detections.

### 3b. Per-fold breakdown

| Fold | Detector | # Proposals | Proposals/video | Score mean | Score max |
|------|----------|-------------|-----------------|------------|-----------|
| 0 | ActionFormer | 8,154 | 81 | 0.039 | 0.620 |
| 0 | TriDet | 7,376 | 73 | 0.032 | 0.727 |
| 1 | ActionFormer | 8,911 | 95 | 0.037 | 0.549 |
| 1 | TriDet | 7,797 | 83 | 0.030 | 0.722 |
| 2 | ActionFormer | 9,934 | 96 | 0.041 | 0.545 |
| 2 | TriDet | 8,922 | 87 | 0.033 | 0.727 |

---

## 4. Proposal–GT IoU Distribution (Stage 1)

For proposals in videos that contain GT segments, the best IoU between each proposal and any GT segment.

### 4a. Summary across folds

| IoU threshold | ActionFormer (% of proposals) | TriDet (% of proposals) | Δ |
|---------------|-------------------------------|--------------------------|---|
| IoU > 0.3 | 17.6% | 18.3% | +0.7 pp |
| IoU > 0.5 | 8.8% | 9.6% | +0.8 pp |
| IoU > 0.7 | 3.6% | 4.2% | +0.6 pp |
| IoU = 0.0 (no overlap) | 60.3% | 59.5% | −0.8 pp |

TriDet has a slightly higher fraction of well-aligned proposals at all IoU thresholds, with ~1% fewer zero-overlap proposals. The improvement is consistent but small.

### 4b. Per-fold detail

| Fold | Detector | # Matched | IoU > 0.3 | IoU > 0.5 | IoU > 0.7 | IoU = 0 |
|------|----------|-----------|-----------|-----------|-----------|---------|
| 0 | AF | 5,770 | 18.5% | 9.3% | 3.8% | 59.3% |
| 0 | TD | 5,182 | 18.7% | 10.0% | 4.2% | 59.3% |
| 1 | AF | 6,556 | 18.7% | 9.5% | 3.8% | 60.2% |
| 1 | TD | 5,725 | 20.0% | 10.6% | 4.6% | 58.4% |
| 2 | AF | 7,925 | 15.8% | 7.7% | 3.2% | 61.4% |
| 2 | TD | 7,076 | 16.2% | 8.3% | 3.7% | 60.7% |

---

## 5. Error Taxonomy (Run 5 — TriDet + V-JEPA2 4-class)

Every prediction classified into one of five mutually exclusive categories based on its best-matching GT segment.

### 5a. Error distribution (3-fold aggregate)

| Error Type | Count | % | Description |
|-----------|-------|---|-------------|
| No overlap (IoU = 0) | 16,824 | 69.8% | Pure false positive — proposal is in a background region |
| Low IoU (0 < IoU < 0.3) | 4,015 | 16.7% | Partial overlap but below detection threshold |
| Wrong class (IoU ≥ 0.3) | 975 | 4.0% | Correctly localized, misclassified |
| Boundary error (correct class, 0.3 ≤ IoU < 0.5) | 1,040 | 4.3% | Right class, imprecise boundaries |
| Correct (correct class, IoU ≥ 0.5) | 1,241 | 5.2% | True positive |
| **Total** | **24,095** | | |

**86.5% of all predictions are either background false positives or have insufficient overlap.** Only 5.2% are unambiguously correct.

### 5b. Per-fold error breakdown

| Fold | Total | Correct (IoU≥0.5) | Boundary err | Wrong class | Low IoU | No overlap |
|------|-------|-------------------|--------------|-------------|---------|------------|
| 0 | 7,376 | 411 (5.6%) | 307 (4.2%) | 250 (3.4%) | 1,142 (15.5%) | 5,266 (71.4%) |
| 1 | 7,797 | 402 (5.2%) | 347 (4.5%) | 394 (5.1%) | 1,240 (15.9%) | 5,414 (69.4%) |
| 2 | 8,922 | 428 (4.8%) | 386 (4.3%) | 331 (3.7%) | 1,633 (18.3%) | 6,144 (68.9%) |

### 5c. Class confusion matrices (GT → Predicted, IoU ≥ 0.3)

**Fold 0:**

| GT ↓ Pred → | hands flapping | jumping | spinning | rocking |
|-------------|----------------|---------|----------|---------|
| hands flapping | — | 65 | 2 | 1 |
| jumping | 55 | — | 1 | 3 |
| spinning | 13 | 1 | — | 11 |
| rocking | 44 | 43 | 11 | — |

**Fold 1:**

| GT ↓ Pred → | hands flapping | jumping | spinning | rocking |
|-------------|----------------|---------|----------|---------|
| hands flapping | — | 59 | 4 | 10 |
| jumping | 147 | — | 11 | 57 |
| spinning | 11 | 8 | — | 1 |
| rocking | 33 | 50 | 3 | — |

**Fold 2:**

| GT ↓ Pred → | hands flapping | jumping | spinning | rocking |
|-------------|----------------|---------|----------|---------|
| hands flapping | — | 62 | 4 | 39 |
| jumping | 69 | — | 3 | 35 |
| spinning | 21 | 14 | — | 4 |
| rocking | 54 | 25 | 1 | — |

**Key confusion patterns (summed across 3 folds):**

| Confusion pair | Count | Interpretation |
|----------------|-------|----------------|
| hands flapping ↔ jumping | 457 | Dominant pair; visually similar brief repetitive movements |
| rocking → hands flapping | 131 | Subtle rocking misread as more common class |
| rocking → jumping | 118 | Rocking misread as jumping |
| jumping → rocking | 95 | Reverse also common |
| spinning ↔ anything | 120 | Spinning confusions distributed; small sample size amplifies noise |

### 5d. Child pose visibility by error bucket (Run 5, all folds)

Pose coverage matches `pose_filter_eval.py`: fraction of proposal frames where **any** COCO-17 body keypoint has confidence **> 0.3** (first person in H5). “Longest-run ratio” = longest contiguous child-present span / proposal length (relevant to variant B cropping).

| Error bucket | n | Mean presence | Median | p25 | p75 | frac presence = 0 | frac < 0.5 | Mean longest-run ratio |
|--------------|---:|--------------:|-------:|----:|----:|-------------------:|-----------:|-----------------------:|
| Correct (class OK, IoU ≥ 0.5) | 1,241 | **0.969** | 1.000 | 1.000 | 1.000 | 2.0% | 2.2% | 0.953 |
| Boundary error (class OK, 0.3 ≤ IoU < 0.5) | 1,040 | **0.957** | 1.000 | 1.000 | 1.000 | 2.9% | 3.3% | 0.932 |
| Wrong class (IoU ≥ 0.3) | 975 | **0.949** | 1.000 | 1.000 | 1.000 | 3.6% | 3.8% | 0.921 |
| Low IoU (0 < IoU < 0.3) | 4,015 | **0.934** | 1.000 | 1.000 | 1.000 | 4.4% | 5.6% | 0.900 |
| **No overlap (IoU = 0)** | 16,824 | **0.831** | **1.000** | 0.988 | 1.000 | **12.9%** | **16.1%** | **0.814** |

**Takeaways**

- **Medians are 1.0 for every bucket** — most proposals sit on timelines where the pose pipeline sees a person almost every frame. Failing proposals are **not** mostly “empty room” by this definition.
- **Pure background FPs (no overlap) are where visibility is weakest:** lower **mean** presence (0.831 vs ~0.93–0.97 for other errors), **~3× higher** rate of zero presence (12.9% vs ~2–4%), and **16.1%** with presence < 0.5 vs **~3–6%** for localized errors. So the **no-overlap mass does concentrate somewhat on low-visibility segments**, but the **majority still have high median coverage** — they are often **person-present but not overlapping GT** (wrong time / wrong behavior in frame), not “no child.”
- **Correct + boundary + wrong-class rows are visibility-rich** — Stage 2 mistakes and boundary issues are **not** explained by missing pose; they occur when the child is usually visible.
- **Script:** `two-stg/error_bucket_pose_visibility.py` (re-run after changing pose rules).

---

## 6. Fold Variance Analysis

### 6a. Per-fold mAP breakdown (all three runs)

| Run | Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|-----|------|---------|---------|---------|---------|
| Run 1 (AF+VJEPA) | 0 | 30.3% | 23.4% | 14.0% | 23.2% |
| Run 1 (AF+VJEPA) | 1 | 22.9% | 15.9% | 6.3% | 15.8% |
| Run 1 (AF+VJEPA) | 2 | 22.7% | 14.0% | 6.9% | 14.5%* |
| Run 5 (TD+VJEPA) | 0 | 33.8% | 28.5% | 15.8% | 26.7% |
| Run 5 (TD+VJEPA) | 1 | 22.4% | 13.8% | 6.3% | 13.4% |
| Run 5 (TD+VJEPA) | 2 | 22.6% | 15.5% | 8.8% | 15.8% |
| Run 5c (TD+3way) | 0 | 24.7% | 20.4% | 11.5% | 19.6% |
| Run 5c (TD+3way) | 1 | 24.5% | 18.7% | 7.1% | 16.2% |
| Run 5c (TD+3way) | 2 | 22.9% | 12.5% | 6.6% | 14.2% |

*Run 1 fold 2 from `metrics.json` (custom eval); OpenTAD metrics not generated for this fold.

**Fold 0 is consistently the best fold** across all runs — by 8–13 pp avg_mAP in Run 5, 5–6 pp in Run 5c. This "easy fold" effect is a major source of variance and must be accounted for in any reported numbers.

### 6b. GT segment distribution per fold

| Fold | Total GT | hands flapping | jumping | spinning | rocking | Val videos |
|------|----------|----------------|---------|----------|---------|------------|
| 0 | 158 | 89 (56%) | 34 (22%) | 9 (6%) | 26 (16%) | 101 |
| 1 | 181 | 99 (55%) | 52 (29%) | 12 (7%) | 18 (10%) | 94 |
| 2 | 176 | 80 (45%) | 50 (28%) | 14 (8%) | 32 (18%) | 103 |

Fold 0's GT is not obviously different from other folds. The performance gap likely stems from intrinsic video characteristics (clarity of RMM behaviors, video quality, or the training distribution of folds 1+2 being more representative).

### 6c. Per-fold deltas (Run 5 − Run 1)

| Fold | Δ avg_mAP | Δ mAP@0.3 | Δ mAP@0.7 |
|------|-----------|-----------|-----------|
| 0 | **+3.5 pp** | +3.5 pp | +1.8 pp |
| 1 | −2.4 pp | −0.5 pp | 0.0 pp |
| 2 | +1.3 pp | −0.1 pp | +1.9 pp |

TriDet's improvement is driven primarily by fold 0. Fold 1 actually regresses. This non-robustness across splits weakens the claim that TriDet is meaningfully better than ActionFormer for this dataset.

---

## 7. Final Score Distribution (after Stage 2)

Stage 2 scores are `det_score × max(class_probs)`. Distribution reveals classifier confidence.

### 7a. Cross-run class distribution shift

V-JEPA2-only (Run 5) vs 3-way fusion (Run 5c) produce dramatically different class distributions from identical proposals:

| Class | Run 5 (VJEPA) mean % of preds | Run 5c (3-way) mean % of preds |
|-------|--------------------------------|--------------------------------|
| hands flapping | 46% | 24% |
| jumping | 26% | 26% |
| spinning | 6% | 9% |
| rocking | 22% | 41% |

The 3-way MLP shifts ~22 pp of prediction mass from hands flapping to rocking, driven by the skeleton modalities (PoseC3D/STGCN++) which encode distinctive kinematic signatures for rocking. However, this redistribution is uncalibrated (MLP trained on GT-clip scores, not proposal scores) and leads to worse overall performance.

### 7b. Score statistics per run (3-fold pooled)

| Run | Mean score | Median score | p95 | Max |
|-----|-----------|-------------|-----|-----|
| Run 1 (AF+VJEPA) | 0.031 | 0.012 | 0.126 | 0.619 |
| Run 5 (TD+VJEPA) | 0.025 | 0.009 | 0.106 | 0.716 |
| Run 5c (TD+3way) | 0.027 | 0.010 | 0.111 | 0.720 |

TriDet proposals yield lower mean scores (fewer mid-range confident predictions) but higher top scores. The 3-way fusion slightly lifts average confidence.

---

## 8. Temporal Boundary Analysis

### 8a. TP boundary quality (Run 5, correct class + IoU ≥ 0.3)

| Fold | # TPs | Mean IoU | Median IoU | p25 IoU | p75 IoU |
|------|-------|----------|------------|---------|---------|
| 0 | 718 | 0.562 | 0.535 | 0.408 | 0.706 |
| 1 | 749 | 0.551 | 0.520 | 0.396 | 0.700 |
| 2 | 814 | 0.545 | 0.513 | 0.391 | 0.685 |
| **Mean** | **760** | **0.553** | **0.523** | **0.398** | **0.697** |

Even among correctly classified detections, boundary quality is moderate:
- 25% of TPs have IoU < 0.40 (barely above the 0.3 threshold)
- Median IoU of 0.52 means half of all "correct" predictions would fail at tIoU=0.6
- This explains the steep mAP drop from @0.3 to @0.7: the boundary precision, not just recall, degrades

---

## 9. Recall & Precision Summary

### 9a. Per-class recall at tIoU=0.3 (3-fold mean, from custom metrics)

| Class | Run 5 (TD+VJEPA) | Run 5c (TD+3way) | Run 1 (AF+VJEPA) |
|-------|-------------------|-------------------|-------------------|
| hands flapping | 93.6% | 72.4% | 92.2%* |
| jumping | 75.0% | 82.1% | 83.3%* |
| spinning | 69.5% | 85.1% | 78.0%* |
| rocking | 82.2% | 87.4% | 85.7%* |

*Run 1 recall from custom `metrics.json` (fold 0 & 2 only; fold 1 uses different metric format).

### 9b. Precision context

With ~24,000 predictions for ~515 GT segments (3 folds combined), raw precision is extremely low:

| Metric | Run 5 | Run 5c |
|--------|-------|--------|
| Total predictions | 24,095 | 24,095 |
| Total GT segments | 515 | 515 |
| Prediction:GT ratio | 46.8:1 | 46.8:1 |
| Correct predictions (IoU≥0.5) | 1,241 (5.2%) | ~1,200 (5.0%) |

---

## 10. Key Findings

### Finding 1: TriDet improves boundary precision but not coarse detection

Run 5 vs Run 1 (same Stage 2):
- **mAP@0.7:** +1.24 pp (10.29% vs 9.05%) — confirms TriDet's Trident-head boundary modeling helps
- **mAP@0.5:** +1.51 pp (19.25% vs 17.74%)
- **mAP@0.3:** +1.00 pp (26.29% vs 25.29%)
- **avg_mAP:** +0.82 pp (18.65% vs 17.83%)

The improvement is real but modest and driven primarily by fold 0 (+3.5 pp), with fold 1 showing regression (−2.4 pp). High fold variance (σ = 7.07 pp for Run 5) means this difference is not statistically robust.

### Finding 2: Class-level improvements are uneven

- **Rocking** benefits most from TriDet (+5.1 pp @0.3, +2.4 pp @0.5, +2.0 pp @0.7). As the hardest class with lowest AP, TriDet's IoU-weighted loss may help with rocking's longer, more ambiguous segments.
- **Hands flapping** trades coarse recall for boundary precision (−1.9 pp @0.3 but +4.6 pp @0.7).
- **Spinning** gets worse across the board (−4.0 pp @0.3, −8.7 pp @0.7), but with only 9–14 GT segments per fold, a single TP swing changes AP by 5–10%.
- **Jumping** follows the hands flapping pattern: small loss at coarse thresholds, gain at tight.

### Finding 3: ~70% of predictions are pure background false positives

The error taxonomy shows 69.8% of all predictions have zero IoU with any GT segment. Another 16.7% have partial overlap below 0.3. Only 9.5% of predictions overlap GT at IoU ≥ 0.3, and of those, 4.0% are misclassified.

The 47:1 proposal-to-GT ratio is a fundamental limitation of the binary proposal generation design, which optimizes Stage 1 recall (98.3%) at massive precision cost. Improving mAP requires either (a) better proposal filtering/scoring to reduce the false positive flood, or (b) better Stage 2 classification to correctly rank the few true positives above the many false positives.

### Finding 4: 3-way fusion redistributes class predictions dramatically

Run 5c vs Run 5 shows the skeleton modalities shift 22 pp of prediction mass from hands flapping to rocking. Jumping and rocking (distinctive kinematics) benefit from PoseC3D/STGCN++; hands flapping and spinning (visual-dominant actions) are hurt. The net effect is negative because:
1. The MLP was trained on GT-clip skeleton scores but receives noisier proposal-level scores (distribution mismatch)
2. The redistribution overcorrects — too many false rocking predictions
3. Hands flapping, the largest class (55% of GT), loses the most, dragging overall mAP down

### Finding 5: Dominant confusion pair: hands flapping ↔ jumping

457 confusions between these two classes across 3 folds (at IoU ≥ 0.3), more than all other confusion pairs combined. Both are brief, repetitive movements that look similar in V-JEPA2 feature space. Rocking is systematically misclassified as hands flapping (131 times) or jumping (118 times) — the classifier defaults to more common classes when uncertain.

### Finding 6: Stage 1 proposal quality is nearly identical between detectors

Despite TriDet's architectural advantages (SGP blocks, Trident-head, IoU-weighted loss):
- IoU distributions are within 1 pp at all thresholds
- Both detectors have ~60% zero-overlap proposals
- TriDet generates 11% fewer proposals with slightly better calibration (3× more score > 0.5 proposals)

The bottleneck is not the detector architecture on this dataset. Feature quality (V-JEPA2 representations) and dataset characteristics (small GT count, imbalanced classes) dominate.

### Finding 7: Two videos are consistently missed — potential data issues

`L4K0P3T4Q8_2020-03-26_19-26-58_287` and `C5L2D6N1W6_12-25-19` have 0% recall despite 37–41 proposals each. Both are single-GT-segment videos. The proposals exist but none land on the GT, suggesting either:
- Very short GT segments that fall between proposal granularity levels
- Potentially questionable annotations
- Atypical video content where temporal patterns differ from training distribution

### Finding 8: Fold 0 is systematically easier

Fold 0 outperforms by 8–13 pp avg_mAP in Run 5 (26.7% vs 13.4–15.8%). GT distribution does not explain this. The "easy fold" effect inflates reported gains when fold 0 is the outlier benefiting from TriDet (+3.5 pp) while folds 1–2 show marginal or negative improvement.

### Finding 9: TP boundary quality is moderate

Among true positives (correct class, IoU ≥ 0.3), mean IoU is 0.553 with p25 at 0.398. A quarter of "correct" predictions barely clear the 0.3 threshold. This explains the 2.5× drop from mAP@0.3 (26.3%) to mAP@0.7 (10.3%) — even successful detections lack tight boundaries.

### Finding 10: Rocking is the hardest class by a large margin

Per-class AP at tIoU=0.3 averages 5–12% for rocking vs 25–40% for other classes. Contributing factors:
- Rare: 14–32 GT segments per fold (vs 80–99 for hands flapping)
- Visually subtle: body sway is harder to distinguish than flapping/jumping/spinning
- Systematically confused with hands flapping (131 times) and jumping (118 times)
- Potentially inconsistent annotations: rocking can overlap with other self-stimulatory behaviors

---

## 11. Data Quality Concerns

1. **Zero-recall videos** (`L4K0P3T4Q8_2020-03-26_19-26-58_287`, `C5L2D6N1W6_12-25-19`): Warrant manual annotation review.
2. **Single-segment videos** are disproportionately represented in both the hardest and easiest lists, indicating high per-video variance driven by annotation density.
3. **Spinning scarcity** (9–14 GT segments per fold): Per-class AP is statistically unreliable — a single TP gain/loss swings AP by 5–10%.
4. **Rocking annotation consistency**: Low AP + high confusion rates suggest potential inter-annotator disagreement. Recommend inter-rater reliability analysis.
5. **47:1 proposal-to-GT ratio**: The binary proposal approach generates far more proposals than necessary. Score thresholding or learned proposal pruning could reduce false positives without significantly hurting recall.

---

## 12. Recommendations for Paper

1. **Report per-class AP** alongside aggregate mAP — the aggregate masks critical class disparities (rocking at 5–12% vs hands flapping at 38–40%).
2. **Report confidence intervals or variance bars** — σ = 3–7 pp across folds means small differences between methods may not be significant. Consider statistical tests (paired t-test or Wilcoxon over fold-level mAPs).
3. **Discuss the fold 0 effect** — it drives the majority of TriDet's apparent gains. Report per-fold numbers in supplementary.
4. **Include error taxonomy** — 70% background FP rate motivates future work on proposal filtering.
5. **Include confusion analysis** — hands flapping ↔ jumping dominance motivates multimodal approaches and suggests that V-JEPA2 features alone may be insufficient for fine-grained RMM discrimination.
6. **Frame TriDet's contribution accurately**: +1.24 pp at mAP@0.7, +0.82 pp avg_mAP. Better boundary precision but not a game changer. The classification bottleneck dominates.
7. **3-way fusion needs recalibration**: The potential is there (jumping +8.7 pp, rocking +2.1 pp from skeleton signal) but the MLP distribution mismatch nullifies the gains. Retraining on proposal-style skeleton scores is the clear next step.
