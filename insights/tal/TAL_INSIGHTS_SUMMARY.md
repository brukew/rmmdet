# TAL Model Behavior Analysis: Key Insights

**Date:** January 5, 2026  
**Analysis:** Top-2 accuracy patterns and Fusion Paradox investigation

---

## Executive Summary

Three major findings from analyzing window-level classification behavior:

1. **The ~10% Top-1 to Top-2 Gap reveals model uncertainty at the RMM/background boundary**
   - Models know the right answer but lack confidence
   - Rare classes (rocking: +61.7%, spinning: +60.0%) show massive recall gains at top-2
   - 60-70% of RMM→BG errors have the correct class as 2nd choice

2. **The Fusion Paradox: Lower accuracy (47%) but higher TAL mAP (6.36%)**
   - Fusion predicts background only 42.5% vs 84-87% for single models
   - This dramatically improves RMM recall (55-68%) at the cost of background precision
   - TAL mAP only evaluates RMM classes, so background errors don't directly hurt

3. **False Positives are High-Confidence Errors — Thresholding Won't Help**
   - BG→RMM errors have mean confidence 0.86-0.90 (nearly same as true positives!)
   - Simple score thresholding cannot effectively filter FPs
   - 75% of FPs are `hands_flapping` — visual similarity with normal gesturing
   - Focus on temporal consistency (smoothing/merging) rather than confidence filtering

---

## Section 1: Top-2 Accuracy Analysis

### 1.1 Overall Top-1 vs Top-2 Performance

| Model | Top-1 Acc | Top-2 Acc | Gap |
|-------|-----------|-----------|-----|
| V-JEPA | 85.0% | 95.4% | **+10.3%** |
| PoseC3D | 85.3% | 96.5% | **+11.2%** |
| Fusion | 47.0% | 93.3% | **+46.3%** |

**Key insight:** All models achieve >93% top-2 accuracy, meaning the correct class is almost always in the top 2 predictions. The ~10% gap for single models indicates systematic uncertainty.

### 1.2 Background False Positive Recovery

When background windows are misclassified as RMM, is background the 2nd choice?

| Model | BG→RMM Errors | BG as 2nd Choice | Recovery Rate |
|-------|---------------|------------------|---------------|
| V-JEPA | 523 | 379 | **72.5%** |
| PoseC3D | 269 | 266 | **98.9%** |

**What RMM classes are background confused with?**

| Confused As | V-JEPA | PoseC3D |
|-------------|--------|---------|
| hands_flapping | 74.8% | 65.8% |
| jumping | 15.3% | 18.6% |
| rocking | 7.3% | 4.8% |
| spinning | 2.7% | 10.8% |

**Insight:** Background is most often confused with `hands_flapping` (likely due to similar arm movements in normal activities). PoseC3D has near-perfect recovery (98.9%), suggesting its errors are highly uncertain.

### 1.3 RMM→Background Error Analysis

When RMM windows are misclassified as background:

**V-JEPA:**
| Class | Misclassified as BG | Recovery Rate |
|-------|---------------------|---------------|
| hands_flapping | 87/292 (29.8%) | 56.3% |
| jumping | 40/90 (44.4%) | 62.5% |
| rocking | 41/47 (87.2%) | 70.7% |
| spinning | 20/25 (80.0%) | 75.0% |
| **TOTAL** | 188 | **62.8%** |

**PoseC3D:**
| Class | Misclassified as BG | Recovery Rate |
|-------|---------------------|---------------|
| hands_flapping | 133/276 (48.2%) | 62.4% |
| jumping | 48/83 (57.8%) | 62.5% |
| rocking | **47/47 (100%)** | 27.7% |
| spinning | 13/25 (52.0%) | 61.5% |
| **TOTAL** | 241 | **55.6%** |

**Critical finding:** PoseC3D classifies **100% of rocking** as background! This explains its 0% F1 for rocking. The low recovery rate (27.7%) suggests these are genuinely ambiguous cases.

### 1.4 Per-Class Recall: Top-1 vs Top-2

| Class | V-JEPA Top-1 | V-JEPA Top-2 | Gap | PoseC3D Top-1 | PoseC3D Top-2 | Gap |
|-------|--------------|--------------|-----|---------------|---------------|-----|
| hands_flapping | 64.7% | 84.6% | +19.9% | 48.9% | 79.3% | +30.4% |
| jumping | 46.7% | 80.0% | +33.3% | 31.3% | 69.9% | +38.6% |
| **rocking** | 6.4% | **68.1%** | **+61.7%** | 0.0% | 27.7% | +27.7% |
| **spinning** | 16.0% | **76.0%** | **+60.0%** | 48.0% | 80.0% | +32.0% |
| background | 88.3% | 96.8% | +8.5% | 91.5% | 99.9% | +8.4% |

**Breakthrough insight:** Rare classes (rocking, spinning) have the largest recall gains from top-1 to top-2:
- V-JEPA rocking: **6.4% → 68.1%** (+61.7%) — models "know" rocking but lack confidence
- V-JEPA spinning: **16.0% → 76.0%** (+60.0%) — same pattern

This suggests:
1. Models learn subtle features of rare classes
2. But background dominance during training causes confidence collapse
3. Lower thresholds or calibration could dramatically improve rare class detection

---

## Section 2: Fusion Paradox Investigation

### 2.1 Prediction Distribution Shift

| Class | Ground Truth | V-JEPA | PoseC3D | Fusion |
|-------|--------------|--------|---------|--------|
| hands_flapping | 5.9% | 11.9% | 8.9% | **12.6%** |
| jumping | 1.8% | 2.8% | 2.3% | **12.1%** |
| rocking | 1.0% | 0.9% | 0.4% | **28.3%** |
| spinning | 0.5% | 0.4% | 1.2% | **4.5%** |
| **background** | **90.8%** | 84.0% | 87.2% | **42.5%** |

**The smoking gun:** Fusion predicts background only **42.5%** of the time vs 84-87% for single models. This is a **dramatic shift** toward RMM predictions.

### 2.2 Error Type Distribution

| Model | Correct | BG→RMM | RMM→BG | RMM→Wrong RMM |
|-------|---------|--------|--------|---------------|
| V-JEPA | 85.0% | 10.6% | 3.8% | 0.6% |
| PoseC3D | 85.3% | 7.5% | 6.7% | 0.5% |
| **Fusion** | **47.0%** | **49.7%** | **1.4%** | 1.9% |

**Error trade-off:**
- Fusion has **10x higher BG→RMM rate** (49.7% vs ~10%)
- But **dramatically lower RMM→BG rate** (1.4% vs 4-7%)
- This is the key to understanding the paradox!

### 2.3 Per-Class Precision/Recall

**V-JEPA:**
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| hands_flapping | 32.0% | 64.7% | 42.9% |
| jumping | 30.2% | 46.7% | 36.7% |
| rocking | 7.1% | 6.4% | 6.7% |
| spinning | 22.2% | 16.0% | 18.6% |
| background | 95.5% | 88.3% | 91.8% |

**PoseC3D:**
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| hands_flapping | 42.1% | 48.9% | 45.2% |
| jumping | 31.3% | 31.3% | 31.3% |
| rocking | 0.0% | 0.0% | 0.0% |
| spinning | 28.6% | 48.0% | 35.8% |
| background | 92.3% | 91.5% | 91.9% |

**Fusion:**
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| hands_flapping | 30.9% | **65.8%** | 42.0% |
| jumping | 10.2% | **67.8%** | 17.8% |
| rocking | 1.9% | **55.3%** | 3.6% |
| spinning | 5.8% | **52.0%** | 10.4% |
| background | **96.8%** | 45.3% | 61.7% |

**Key trade-off:**
- Fusion achieves **55-68% recall** for all RMM classes (vs 0-65% for single models)
- But precision tanks (2-31% for RMMs)
- Background recall drops to 45.3%, but precision stays high (96.8%)

### 2.4 Confidence Distribution Analysis

| Metric | V-JEPA | PoseC3D | Fusion |
|--------|--------|---------|--------|
| Mean max confidence | 0.975 | 0.957 | **0.579** |
| Mean BG score | 0.839 | 0.859 | **0.365** |
| Mean max RMM score | 0.156 | 0.136 | **0.464** |
| % where BG > max RMM | 84.0% | 87.2% | **42.5%** |

**Confidence calibration:**
- Single models are highly confident (97.5% max confidence)
- Fusion is much less confident (57.9% max)
- This spreads probability mass across classes, letting RMMs "win" more often

---

## Section 3: Deep Dive — Can Thresholding Filter False Positives?

### 3.1 Critical Finding: FPs are HIGH-Confidence Errors

| Error Type | V-JEPA Mean Conf | PoseC3D Mean Conf |
|------------|------------------|-------------------|
| BG Correct (TN) | 0.988 | 0.972 |
| **BG→RMM (FP)** | **0.905** | **0.862** |
| RMM Correct (TP) | ~0.90 | ~0.87 |

**Key insight:** False positives have nearly identical confidence to true positives!
- BG→RMM errors are **not low-confidence mistakes** — the model is confidently wrong
- Simple max-score thresholding **cannot effectively separate** FPs from TPs

### 3.2 FP Distribution by Predicted Class

| Predicted As | V-JEPA | PoseC3D |
|--------------|--------|---------|
| **hands_flapping** | **74.8%** | **65.8%** |
| jumping | 15.3% | 18.6% |
| rocking | 7.3% | 4.8% |
| spinning | 2.7% | 10.8% |

**hands_flapping dominates FPs** — likely due to visual similarity between normal gesturing/waving and the RMM behavior. This is a fundamental model limitation, not a calibration issue.

### 3.3 Margin Thresholding Analysis

**Margin = max_RMM_score - BG_score**

| Prediction Type | V-JEPA Margin | PoseC3D Margin |
|-----------------|---------------|----------------|
| BG Correct | -0.977 | -0.95 |
| **BG→RMM (FP)** | **0.838** | **0.81** |
| RMM Correct (TP) | 0.889 | 0.85 |

FPs and TPs have similar margins — margin thresholding offers limited help:

| Margin Threshold | FPs Filtered | TPs Lost |
|------------------|--------------|----------|
| >0.3 | 6% | 4% |
| >0.5 | 14% | 9% |
| >0.7 | 21% | 15% |

**Trade-off is steep:** Filtering 21% of FPs costs 15% of TPs.

### 3.4 Why Simple Thresholding Fails

1. **Not uncertainty, but genuine confusion**: The model truly believes BG windows are RMM
2. **Visual similarity**: hands_flapping looks like normal gesturing
3. **Confident errors**: Mean FP confidence ~0.86-0.90, nearly matching TP confidence
4. **Entropy is not discriminative**: FPs and TPs have similar prediction entropy

---

## Why Fusion Has Lower Accuracy but Higher TAL mAP

### The Complete Picture

```
Single Models (V-JEPA, PoseC3D):
├── High window accuracy (~85%)
├── Predicts BG ~85% of time
├── Few BG→RMM errors (7-11%)
├── Many RMM→BG errors (4-7%)
└── TAL Result: Misses many RMM segments → Lower recall → Lower mAP

Fusion Model:
├── Low window accuracy (47%)
├── Predicts BG only 42.5% of time
├── Many BG→RMM errors (50%)
├── Few RMM→BG errors (1.4%)
└── TAL Result: Catches more RMM segments → Higher recall → Higher mAP
```

### TAL mAP Mechanics

TAL mAP only evaluates RMM classes (not background):
1. **BG→RMM errors** create false positive detections → hurts RMM precision
2. **RMM→BG errors** create false negative detections → hurts RMM recall

For low-prevalence classes, **recall matters more than precision** because:
- Missing a true positive (FN) directly reduces mAP
- False positives (FP) are diluted across many background windows

### Why Fusion Wins

Fusion's aggressive RMM prediction strategy:
1. **Catches more true RMM segments** (1.4% RMM→BG vs 4-7%)
2. **Creates many false positives** (50% BG→RMM)
3. **But**: False positives can be filtered by postprocessing (threshold, smoothing, merging)
4. **And**: TAL evaluation uses confidence-weighted mAP, so high-confidence TPs outweigh low-confidence FPs

---

## Recommendations

### For Window-Level Classification
- Use **lower confidence thresholds** for rare classes (rocking, spinning)
- Consider **temperature scaling** to calibrate confidence
- The top-2 analysis shows models "know" the answer — they just need encouragement
- **Don't rely on confidence thresholding** to filter FPs — they're high-confidence errors

### For TAL Postprocessing
- **Fusion wins** because recall matters more than precision for imbalanced detection
- **Simple score thresholding won't help** — FPs have similar confidence to TPs
- Focus on **temporal consistency** filtering:
  - Smoothing: Isolated FP windows get averaged out
  - Merging: Short FP segments get merged into longer BG regions
  - Minimum duration: Filter segments shorter than typical RMM duration
- Consider **class-specific strategies** (stricter filtering for hands_flapping which causes 75% of FPs)

### For Rare Classes
- **Rocking is extremely hard**: 0% recall for PoseC3D, 6.4% for V-JEPA
- But V-JEPA achieves **68.1% top-2 recall** for rocking — the features exist!
- Training strategies: focal loss, oversampling, or cascade classifiers may help

### Why Fusion Works (Despite Lower Accuracy)
- Fusion doesn't work because of better calibration
- It works because **different models have different error patterns**
- V-JEPA and PoseC3D make different mistakes → combining them catches more true RMMs
- The aggressive RMM prediction strategy (42.5% BG vs 85% for single models) trades window accuracy for TAL recall

---

## Section 4: PoseC3D Balanced Sampling Analysis

### Configuration
- **Class sampling probability:** `[1.0, 1.0, 1.93, 9.12, 0.1]` — heavily undersample background, oversample rare classes
- **Class weights:** `[3.3, 7.0, 13.4, 63.5, 0.22]` — inverse frequency weighting

### Results (3-fold CV)

| Metric | PoseC3D (Original) | PoseC3D (Balanced) |
|--------|--------------------|--------------------|
| Top-1 Accuracy | 85.9% | **81.1%** |
| Top-2 Accuracy | 96.2% | 96.3% |
| Macro F1 | 37.6% | 35.8% |
| RMM Recall | ~49% | **~51%** |
| BG False Alarm | ~17% | **~14%** |

### Per-Class F1

| Class | Original | Balanced |
|-------|----------|----------|
| hands_flapping | 37.5% | 37.5% |
| jumping | 27.9% | 27.9% |
| **rocking** | **0%** | **2.8%** |
| spinning | 21.5% | 21.5% |

### Key Findings
1. **Accuracy drops** (81% vs 86%) due to background undersampling
2. **Rocking slightly improves** (0% → 2.8% F1) but remains extremely difficult
3. **High fold variance** for rare classes (spinning: 0-52% F1 across folds)
4. **Trade-off**: May help TAL mAP by reducing RMM→BG errors, but at cost of more BG→RMM FPs
5. **Massive overfitting**: 99% train accuracy vs 81% val accuracy despite balanced sampling

---

## Files

- Analysis notebook: `insights/tal/tal_model_analysis.ipynb`
- Model comparisons: `tal/TAL_MODEL_COMPARISON.md`, `tal/TAL_DET_MODEL_COMPARISON.md`

