# Classifier Effect & Two-Stage Analysis

All metrics are 3-fold cross-validation. GT-overlapping = best IoU ≥ 0.3 with any GT segment. TP = correct class + IoU ≥ 0.3.

---

# Part 1: MLP Effect

Pairs share the **same detector proposals** (same start/end times). Only the Stage-2 classifier differs: V-JEPA2 alone vs 3-way MLP fusion (V-JEPA2 + PoseC3D + STGCN++).

## 1a. Classification Accuracy on GT-Overlapping Proposals

For proposals with best IoU ≥ 0.3 to any GT segment, what fraction get the **correct class label**?

| Run | GT-overlap proposals | Correct label | Accuracy |
|-----|---------------------|---------------|----------|
| AF + V-JEPA2 (Run 1) | 3,551 | 2,469 | **69.5%** |
| AF + MLP (Run 3c) | 3,551 | 2,242 | **63.1%** |
| TD + V-JEPA2 (Run 5) | 3,256 | 2,281 | **70.1%** |
| TD + MLP (Run 5c) | 3,256 | 2,038 | **62.6%** |
| AF E2E (baseline) | 8,204 | 2,468 | **30.1%** |
| TD E2E (Run 6) | 8,309 | 2,469 | **29.7%** |

### Per-class accuracy (IoU ≥ 0.3, 3-fold pooled)

| GT Class | AF + V-JEPA2 (Run 1) | AF + MLP (Run 3c) | TD + V-JEPA2 (Run 5) | TD + MLP (Run 5c) | AF E2E (baseline) | TD E2E (Run 6) |
|----------|------||------||------||------||------||------|
| hands flapping | 82.8% | 56.8% | 84.3% | 55.9% | 31.3% | 30.3% |
| jumping | 56.8% | 69.0% | 56.5% | 67.6% | 29.3% | 29.7% |
| rocking | 51.4% | 66.6% | 50.8% | 66.7% | 28.8% | 29.1% |
| spinning | 67.8% | 74.8% | 70.0% | 76.4% | 28.0% | 28.1% |

### Balanced accuracy (mean of per-class accuracies)

Neutralizes the majority-class effect. Balanced acc = (1/C) Σ per-class accuracy.

| Run | Overall accuracy | Balanced accuracy |
|-----|:---------------:|:-----------------:|
| AF + V-JEPA2 (Run 1) | 69.5% | **64.7%** |
| AF + MLP (Run 3c) | 63.1% | **66.8%** |
| TD + V-JEPA2 (Run 5) | 70.1% | **65.4%** |
| TD + MLP (Run 5c) | 62.6% | **66.6%** |
| AF E2E (baseline) | 30.1% | **29.4%** |
| TD E2E (Run 6) | 29.7% | **29.3%** |

## 1b. Head-to-Head Label Comparison

Same proposals, two classifiers. When they **disagree**, which is correct?

### ActionFormer proposals: AF + V-JEPA2 (Run 1) vs AF + MLP (Run 3c)

| Outcome | Count | % of GT-overlapping |
|---------|------:|--------------------:|
| **Agree, correct** | 1,978 | 49.5% |
| **Agree, wrong** | 482 | 12.1% |
| **V-JEPA2 right, MLP wrong** | 799 | 20.0% |
| **MLP right, V-JEPA2 wrong** | 546 | 13.7% |
| **Both wrong, different labels** | 194 | 4.9% |
| **Total GT-overlapping** | 3,999 | 100% |
| Total proposals matched | 74,603 | |

**Label agreement rate:** 61.5% (agree on 2,460 / 3,999 GT-overlapping proposals)

**Net MLP advantage:** -253 proposals (MLP uniquely correct: 546, V-JEPA2 uniquely correct: 799)

#### Per-class disagreement detail (ActionFormer)

When the two classifiers **disagree**, what class does each pick?

| GT class | V-JEPA2 right / MLP wrong | MLP right / V-JEPA2 wrong | Net MLP |
|----------|:-------------------------:|:-------------------------:|:-------:|
| hands flapping | 588 | 84 | -504 |
| jumping | 126 | 265 | +139 |
| rocking | 53 | 144 | +91 |
| spinning | 32 | 53 | +21 |

### TriDet proposals: TD + V-JEPA2 (Run 5) vs TD + MLP (Run 5c)

| Outcome | Count | % of GT-overlapping |
|---------|------:|--------------------:|
| **Agree, correct** | 2,036 | 50.0% |
| **Agree, wrong** | 516 | 12.7% |
| **V-JEPA2 right, MLP wrong** | 835 | 20.5% |
| **MLP right, V-JEPA2 wrong** | 510 | 12.5% |
| **Both wrong, different labels** | 175 | 4.3% |
| **Total GT-overlapping** | 4,072 | 100% |
| Total proposals matched | 70,327 | |

**Label agreement rate:** 62.7% (agree on 2,552 / 4,072 GT-overlapping proposals)

**Net MLP advantage:** -325 proposals (MLP uniquely correct: 510, V-JEPA2 uniquely correct: 835)

#### Per-class disagreement detail (TriDet)

When the two classifiers **disagree**, what class does each pick?

| GT class | V-JEPA2 right / MLP wrong | MLP right / V-JEPA2 wrong | Net MLP |
|----------|:-------------------------:|:-------------------------:|:-------:|
| hands flapping | 637 | 62 | -575 |
| jumping | 133 | 258 | +125 |
| rocking | 40 | 147 | +107 |
| spinning | 25 | 43 | +18 |

## 1c. AUROC: Does the Final Score Separate TPs from FPs?

Binary classification: is this prediction a TP (correct class + IoU ≥ 0.3)? Higher AUROC = final score better ranks TPs above FPs.

| Run | Total preds | TPs | FPs | AUROC |
|-----|------------:|----:|----:|------:|
| AF + V-JEPA2 (Run 1) | 26,999 | 2,469 | 24,530 | **0.5514** |
| AF + MLP (Run 3c) | 26,999 | 2,242 | 24,757 | **0.5332** |
| TD + V-JEPA2 (Run 5) | 24,095 | 2,281 | 21,814 | **0.5818** |
| TD + MLP (Run 5c) | 24,095 | 2,038 | 22,057 | **0.5655** |
| AF E2E (baseline) | 51,482 | 2,468 | 49,014 | **0.5828** |
| TD E2E (Run 6) | 51,053 | 2,469 | 48,584 | **0.5986** |

### Per-class AUROC

| Class | AF + V-JEPA2 (Run 1) | AF + MLP (Run 3c) | TD + V-JEPA2 (Run 5) | TD + MLP (Run 5c) | AF E2E (baseline) | TD E2E (Run 6) |
|-------|------||------||------||------||------||------|
| hands flapping | 0.564 | 0.550 | 0.589 | 0.580 | 0.551 | 0.573 |
| jumping | 0.551 | 0.543 | 0.575 | 0.566 | 0.621 | 0.626 |
| rocking | 0.493 | 0.496 | 0.551 | 0.533 | 0.474 | 0.512 |
| spinning | 0.518 | 0.508 | 0.551 | 0.536 | 0.663 | 0.648 |

## 1d. Score Distributions (TP vs FP)

| Run | TP mean score | FP mean score | TP/FP ratio | TP median | FP median |
|-----|:------------:|:------------:|:-----------:|:---------:|:---------:|
| AF + V-JEPA2 (Run 1) | 0.0544 | 0.0286 | 1.9× | 0.0156 | 0.0119 |
| AF + MLP (Run 3c) | 0.0562 | 0.0313 | 1.8× | 0.0155 | 0.0135 |
| TD + V-JEPA2 (Run 5) | 0.0537 | 0.0226 | 2.4× | 0.0131 | 0.0083 |
| TD + MLP (Run 5c) | 0.0560 | 0.0247 | 2.3× | 0.0130 | 0.0093 |
| AF E2E (baseline) | 0.0661 | 0.0355 | 1.9× | 0.0304 | 0.0211 |
| TD E2E (Run 6) | 0.0583 | 0.0233 | 2.5× | 0.0187 | 0.0124 |

## 1e. What Does the MLP Actually Change? (Label Redistribution)

For GT-overlapping proposals (IoU ≥ 0.3), how does each classifier label them?

### ActionFormer proposals

| GT Class | n GT-overlap | V-JEPA2 assigns correct | MLP assigns correct | V-JEPA2 most common wrong | MLP most common wrong |
|----------|:-----------:|:----------------------:|:-------------------:|:------------------------:|:---------------------:|
| hands flapping | 1967 | 1636 (83%) | 1132 (58%) | jumping (252) | jumping (486) |
| jumping | 1071 | 590 (55%) | 729 (68%) | hands flapping (350) | rocking (189) |
| rocking | 614 | 311 (51%) | 402 (65%) | hands flapping (158) | hands flapping (91) |
| spinning | 347 | 240 (69%) | 261 (75%) | hands flapping (56) | jumping (40) |

### TriDet proposals

| GT Class | n GT-overlap | V-JEPA2 assigns correct | MLP assigns correct | V-JEPA2 most common wrong | MLP most common wrong |
|----------|:-----------:|:----------------------:|:-------------------:|:------------------------:|:---------------------:|
| hands flapping | 1941 | 1659 (85%) | 1084 (56%) | jumping (218) | jumping (532) |
| jumping | 1134 | 637 (56%) | 762 (67%) | hands flapping (365) | rocking (205) |
| rocking | 639 | 313 (49%) | 420 (66%) | hands flapping (163) | jumping (93) |
| spinning | 358 | 262 (73%) | 280 (78%) | hands flapping (47) | rocking (40) |

---

# Part 2: Error Taxonomy Across All Runs

Every prediction classified into one of five categories based on its best-matching GT segment.

## 2a. Error Breakdown (3-fold pooled)

| Run | Total | Correct (IoU≥0.5, right class) | Boundary (0.3≤IoU<0.5, right class) | Wrong class (IoU≥0.3) | Low IoU (0<IoU<0.3) | No overlap (IoU=0) |
|-----|------:|-----:||-----:||-----:||-----:||-----:|
| AF + V-JEPA2 (Run 1) | 26,999 | 1,280 (4.7%) | 1,189 (4.4%) | 1,082 (4.0%) | 4,463 (16.5%) | 18,985 (70.3%) |
| AF + MLP (Run 3c) | 26,999 | 1,197 (4.4%) | 1,045 (3.9%) | 1,309 (4.8%) | 4,463 (16.5%) | 18,985 (70.3%) |
| TD + V-JEPA2 (Run 5) | 24,095 | 1,241 (5.2%) | 1,040 (4.3%) | 975 (4.0%) | 4,015 (16.7%) | 16,824 (69.8%) |
| TD + MLP (Run 5c) | 24,095 | 1,143 (4.7%) | 895 (3.7%) | 1,218 (5.1%) | 4,015 (16.7%) | 16,824 (69.8%) |
| AF E2E (baseline) | 51,482 | 1,281 (2.5%) | 1,187 (2.3%) | 5,736 (11.1%) | 9,776 (19.0%) | 33,502 (65.1%) |
| TD E2E (Run 6) | 51,053 | 1,314 (2.6%) | 1,155 (2.3%) | 5,840 (11.4%) | 9,830 (19.3%) | 32,914 (64.5%) |

### Percentage-only view

| Run | Correct | Boundary | Wrong class | Low IoU | No overlap |
|-----|--------:|---------:|------------:|--------:|-----------:|
| AF + V-JEPA2 (Run 1) | 4.7% | 4.4% | 4.0% | 16.5% | 70.3% |
| AF + MLP (Run 3c) | 4.4% | 3.9% | 4.8% | 16.5% | 70.3% |
| TD + V-JEPA2 (Run 5) | 5.2% | 4.3% | 4.0% | 16.7% | 69.8% |
| TD + MLP (Run 5c) | 4.7% | 3.7% | 5.1% | 16.7% | 69.8% |
| AF E2E (baseline) | 2.5% | 2.3% | 11.1% | 19.0% | 65.1% |
| TD E2E (Run 6) | 2.6% | 2.3% | 11.4% | 19.3% | 64.5% |

## 2b. Per-Class Error Breakdown (IoU ≥ 0.3 proposals only)

Among predictions that overlap GT at IoU ≥ 0.3, what fraction are correct vs wrong class, broken down by GT class?

### AF + V-JEPA2 (Run 1)

| GT Class | Correct class | Wrong class | Accuracy |
|----------|:------------:|:-----------:|:--------:|
| hands flapping | 1432 | 297 | 82.8% |
| jumping | 538 | 409 | 56.8% |
| rocking | 295 | 279 | 51.4% |
| spinning | 204 | 97 | 67.8% |

### AF + MLP (Run 3c)

| GT Class | Correct class | Wrong class | Accuracy |
|----------|:------------:|:-----------:|:--------:|
| hands flapping | 982 | 747 | 56.8% |
| jumping | 653 | 294 | 69.0% |
| rocking | 382 | 192 | 66.6% |
| spinning | 225 | 76 | 74.8% |

### TD + V-JEPA2 (Run 5)

| GT Class | Correct class | Wrong class | Accuracy |
|----------|:------------:|:-----------:|:--------:|
| hands flapping | 1317 | 246 | 84.3% |
| jumping | 495 | 381 | 56.5% |
| rocking | 273 | 264 | 50.8% |
| spinning | 196 | 84 | 70.0% |

### TD + MLP (Run 5c)

| GT Class | Correct class | Wrong class | Accuracy |
|----------|:------------:|:-----------:|:--------:|
| hands flapping | 874 | 689 | 55.9% |
| jumping | 592 | 284 | 67.6% |
| rocking | 358 | 179 | 66.7% |
| spinning | 214 | 66 | 76.4% |

### AF E2E (baseline)

| GT Class | Correct class | Wrong class | Accuracy |
|----------|:------------:|:-----------:|:--------:|
| hands flapping | 1251 | 2745 | 31.3% |
| jumping | 623 | 1502 | 29.3% |
| rocking | 398 | 986 | 28.8% |
| spinning | 196 | 503 | 28.0% |

### TD E2E (Run 6)

| GT Class | Correct class | Wrong class | Accuracy |
|----------|:------------:|:-----------:|:--------:|
| hands flapping | 1194 | 2749 | 30.3% |
| jumping | 635 | 1506 | 29.7% |
| rocking | 425 | 1034 | 29.1% |
| spinning | 215 | 551 | 28.1% |

## 2c. E2E vs Two-Stage: Where Do the Errors Differ?

Direct comparison of ActionFormer E2E baseline against its two-stage counterparts (Runs 1 and 3c), all using ActionFormer proposals.

| Metric | AF E2E (baseline) | AF + V-JEPA2 (Run 1) | AF + MLP (Run 3c) |
|--------|-----:||-----:||-----:|
| Correct (IoU≥0.5, right class) | 2.5% | 4.7% | 4.4% |
| Boundary (0.3≤IoU<0.5, right class) | 2.3% | 4.4% | 3.9% |
| Wrong class (IoU≥0.3) | 11.1% | 4.0% | 4.8% |
| Low IoU (0<IoU<0.3) | 19.0% | 16.5% | 16.5% |
| No overlap (IoU=0) | 65.1% | 70.3% | 70.3% |
| **Total predictions** | 51,482 | 26,999 | 26,999 |

### Classification accuracy: E2E vs Two-Stage (IoU ≥ 0.3)

| GT Class | AF E2E (baseline) | AF + V-JEPA2 (Run 1) | AF + MLP (Run 3c) |
|----------|-----:||-----:||-----:|
| hands flapping | 31.3% | 82.8% | 56.8% |
| jumping | 29.3% | 56.8% | 69.0% |
| rocking | 28.8% | 51.4% | 66.6% |
| spinning | 28.0% | 67.8% | 74.8% |

### TriDet two-stage: Run 5 vs Run 5c

| Metric | TD + V-JEPA2 (Run 5) | TD + MLP (Run 5c) |
|--------|-----:||-----:|
| Correct (IoU≥0.5, right class) | 5.2% | 4.7% |
| Boundary (0.3≤IoU<0.5, right class) | 4.3% | 3.7% |
| Wrong class (IoU≥0.3) | 4.0% | 5.1% |
| Low IoU (0<IoU<0.3) | 16.7% | 16.7% |
| No overlap (IoU=0) | 69.8% | 69.8% |
| **Total predictions** | 24,095 | 24,095 |

---

# Summary

*(Key findings from this analysis — review and interpret for paper.)*
