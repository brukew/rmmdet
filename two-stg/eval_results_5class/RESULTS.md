# Two-Stage TAL Results (5-Class Classifier)

Binary ActionFormer detects RMM segments → V-JEPA2 **5-class** classifier (4 RMM + background) assigns labels → scores computed as `det_score × max(rmm_class_probs)`, ignoring background probability. This naturally downweights proposals where the classifier predicts background, without hard filtering.

## Models Used

All paths relative to `/orcd/data/satra/001/users/brukew/actreg/`.

- **Detector:** `OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold{0,1,2}/gpu1_id99/result_detection.json`
- **Classifier:** `v-jepa/runs/vjepa2_tal_cv_5class_balanced/fold_{0,1,2}` (V-JEPA2, 5-class with background)

## Results

| Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|
| 0 | 30.97% | 22.95% | 12.17% | 22.08% |
| 1 | 18.45% | 12.56% | 5.98% | 12.54% |
| 2 | 19.23% | 13.36% | 6.21% | 13.57% |
| **Mean ± Std** | **22.88% ± 5.72%** | **16.29% ± 4.72%** | **8.12% ± 2.87%** | **16.07% ± 4.28%** |

Per-class AP@0.3 (mean ± std across folds):

| Class | AP@0.3 |
|-------|--------|
| hands flapping | 35.5% ± 8.4% |
| jumping | 31.4% ± 14.8% |
| spinning | 14.9% ± 11.9% |
| rocking | 9.7% ± 6.0% |

## Background Probability Analysis

The 5-class classifier outputs a background probability for each proposal. Higher bg_prob means the classifier thinks the proposal is not an RMM event.

| Statistic | Value |
|-----------|-------|
| Total predictions | 26,999 |
| bg_prob mean | 0.715 |
| bg_prob median | 0.888 |
| bg_prob std | 0.329 |

| Threshold | Count | Percentage |
|-----------|-------|------------|
| bg_prob > 0.5 | 20,215 | 74.9% |
| bg_prob > 0.7 | 18,089 | 67.0% |
| bg_prob > 0.9 | 12,944 | 47.9% |

**Key insight:** The classifier predicts background for ~75% of proposals (bg_prob > 0.5), suggesting it's highly conservative about RMM classification.

## Deep Dive: Score Reduction Analysis

Comparing matched predictions between 4-class and 5-class classifiers:

### Overall Score Impact

| Metric | 4-class | 5-class | Ratio |
|--------|---------|---------|-------|
| Mean score | 0.0310 | 0.0114 | **36.7%** |
| Median score | 0.0121 | 0.0011 | 9.1% |
| Max score | 0.6188 | 0.5890 | 95.2% |

### Score Reduction by Background Probability

| bg_prob Range | Predictions | Median Score Ratio (5cls/4cls) |
|---------------|-------------|-------------------------------|
| [0.0, 0.3) | 5,428 | **0.869** (mild reduction) |
| [0.3, 0.5) | 3,477 | 0.591 |
| [0.5, 0.7) | 2,553 | 0.361 |
| [0.7, 0.9) | 37,331 | 0.192 |
| [0.9, 1.0) | 25,814 | **0.034** (severe reduction) |

When bg_prob > 0.9, scores are reduced to just **3.4%** of the 4-class scores.

### Label Agreement

- **Same label:** 41,889 (56.1%)
- **Different label:** 32,714 (43.9%)

When labels differ, the median bg_prob is **0.933** — indicating the 5-class classifier is uncertain about these proposals.

### Top Class Confusion Patterns (when 4-class ≠ 5-class)

| 4-class Label | 5-class Label | Count | % of Disagreements |
|---------------|---------------|-------|-------------------|
| jumping | hands flapping | 14,073 | 43.0% |
| jumping | spinning | 9,396 | 28.7% |
| hands flapping | rocking | 2,728 | 8.3% |
| hands flapping | jumping | 1,326 | 4.1% |
| rocking | jumping | 1,166 | 3.6% |

### Per-Class Background Confusion

| Class (4-class) | High bg_prob (>0.7) | Score Ratio when bg>0.7 |
|-----------------|---------------------|------------------------|
| **jumping** | **94.0%** | 0.099 (worst affected) |
| rocking | 91.9% | 0.177 |
| hands flapping | 63.6% | 0.039 |
| spinning | 52.1% | 0.074 |

**Critical finding:** The 5-class classifier predicts high background probability for **94% of jumping** proposals, severely downweighting this class. When 4-class predicts "jumping" but 5-class has high bg_prob:
- 37% relabeled as "hands flapping"
- 35% kept as "jumping" (but with reduced score)
- 26% relabeled as "spinning"

## Comparison: 5-Class vs 4-Class

| Model | avg_mAP | mAP@0.3 | Notes |
|-------|---------|---------|-------|
| **Two-Stage 4-class** | **17.83%** | 25.29% | Better performance |
| Two-Stage 5-class (this) | 16.07% | 22.88% | Conservative classifier |
| ActionFormer Balanced | 16.70% | 23.97% | End-to-end 4-class |

**Why does 5-class underperform?**

1. **Jumping class is devastated:** 94% of jumping predictions have bg_prob > 0.7, reducing scores to just 10% of the 4-class scores. The classifier confuses jumping with hands flapping (37%) and spinning (26%).

2. **Severe score reduction:** When bg_prob > 0.9 (48% of predictions), scores drop to 3.4% of 4-class values. This pushes many true positives down in the ranking.

3. **44% label disagreement:** The 5-class classifier often predicts different RMM types than the 4-class classifier, with most disagreements occurring when bg_prob is high (median 0.933).

4. **Training distribution mismatch:** The 5-class classifier was trained on windows with explicit background labels (~80% background). When applied to ActionFormer proposals (which are already filtered for RMM by Stage 1), the classifier over-predicts background.

## Conclusion

The **4-class classifier is preferred** for two-stage TAL. The 5-class model's background predictions don't translate well to the two-stage setting where Stage 1 already filters for RMM proposals.

Potential alternatives to explore:
- Hard filtering proposals where bg_prob > threshold
- Training a 4-class classifier on ActionFormer proposals specifically
- Using bg_prob as a separate gating factor: `det_score × (1 - bg_prob) × max(rmm_probs)`

## Artifacts

- Per-fold metrics/predictions: `two-stg/eval_results_5class/fold{0,1,2}/metrics.json`
- Predictions include `bg_prob` column for post-hoc analysis
- Script: `two-stg/eval_two_stage_tal.py --num-classes 5`
