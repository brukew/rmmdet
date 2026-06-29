# Analysis Notes — Classifier Effect & Two-Stage TAL

**Date:** 2026-03-26
**Script:** `two-stg/analyze_classifier_effect.py`
**Output:** `two-stg/analysis_classifier_effect.md`

---

## Bug Fix: ID2LABEL Mapping

The `ID2LABEL` mapping in `analyze_runs_5_5c.py` and the new analysis script had **spinning (class_id=3) and rocking (class_id=2) swapped** compared to `tal/tal_map_eval.py:LABEL_MAP_4CLASS`. The correct mapping is:

```
{0: "hands flapping", 1: "jumping", 2: "rocking", 3: "spinning"}
```

This did not affect Runs 5/5c (which store explicit `label` strings in predictions.csv) but **silently corrupted Run 1's per-class numbers** (which only stores `class_id`). Both scripts are now fixed.

---

## MLP Effect

**Key finding: V-JEPA2 has a majority-class bias; MLP is more balanced but loses overall.**

| Metric | V-JEPA2 | MLP |
|--------|---------|-----|
| Overall accuracy (AF proposals) | **69.5%** | 63.1% |
| Overall accuracy (TD proposals) | **70.1%** | 62.6% |
| Hands flapping accuracy | **83–84%** | 56–57% |
| Jumping accuracy | 56–57% | **67–69%** |
| Rocking accuracy | 50–51% | **66–67%** |
| Spinning accuracy | 68–70% | **75–76%** |

V-JEPA2's overall lead comes entirely from hands flapping (the largest class, ~38% of GT). The MLP is better on the other three classes.

**Head-to-head disagreement (when they pick different labels):**
- V-JEPA2 wins hands flapping by 504–575 proposals
- MLP wins jumping by 125–139, rocking by 91–107, spinning by 18–21
- Net: V-JEPA2 wins by 253 (AF) and 325 (TD) — driven entirely by hands flapping

**AUROC (does the final score separate TPs from FPs):**
- V-JEPA2 > MLP on both detectors (0.551 vs 0.533 on AF; 0.582 vs 0.566 on TD)
- All AUROCs are low (~0.55) — classifiers are poor rankers; detector confidence dominates

**Why worse accuracy → worse mAP:**
1. Majority class (hands flapping) is where V-JEPA2 dominates; losing there drags aggregate mAP
2. V-JEPA2 has slightly better TP/FP score separation (better ranking)

**Balanced accuracy (mean of per-class accuracies) flips the story:**

| Run | Overall accuracy | Balanced accuracy |
|-----|:---:|:---:|
| AF + V-JEPA2 (Run 1) | **69.5%** | 64.7% |
| AF + MLP (Run 3c) | 63.1% | **66.8%** |
| TD + V-JEPA2 (Run 5) | **70.1%** | 65.4% |
| TD + MLP (Run 5c) | 62.6% | **66.6%** |
| AF E2E (baseline) | 30.1% | 29.4% |
| TD E2E (Run 6) | 29.7% | 29.3% |

**The MLP is a better balanced classifier.** V-JEPA2 wins overall accuracy by 6–8pp because it defaults to hands flapping, but the MLP wins balanced accuracy by 1–2pp because it treats all four classes more evenly. This is a classic precision/balanced-accuracy tradeoff in imbalanced classification: V-JEPA2 optimizes for the common case; the MLP produces a fairer classifier that mAP doesn't reward because hands flapping dominates GT frequency.

---

## Two-Stage vs E2E

**Key finding: the binary first stage is a powerful filter; E2E models are the best rankers.**

| Metric | AF E2E | TD E2E (Run 6) | AF Two-Stage (V-JEPA2) | TD Two-Stage (V-JEPA2) |
|--------|--------|----------------|------------------------|------------------------|
| Total predictions | 51,482 | 51,053 | 26,999 | 24,095 |
| Correct (IoU≥0.5) | 2.5% | 2.6% | 4.7% | 5.2% |
| Wrong class (IoU≥0.3) | 11.1% | 11.4% | 4.0% | 4.0% |
| No overlap | 65.1% | 64.5% | 70.3% | 69.8% |
| Per-class accuracy | ~29% (near chance) | ~29% (near chance) | 51–83% | 51–84% |

Both E2E models produce ~2× more predictions with ~3× the wrong-class rate. Their per-class accuracy (~29–30%) is near chance for 4 classes, confirming they barely classify and rely on detection confidence for ranking.

The binary Stage 1 halves predictions while preserving TP count, then V-JEPA2 classifies at ~70% accuracy — far better than E2E's ~30%.

**AUROC (ranking quality):**

| Run | AUROC |
|-----|:-----:|
| TD E2E (Run 6) | **0.5986** |
| AF E2E (baseline) | 0.5828 |
| TD + V-JEPA2 (Run 5) | 0.5818 |
| AF + V-JEPA2 (Run 1) | 0.5514 |
| TD + MLP (Run 5c) | 0.5655 |
| AF + MLP (Run 3c) | 0.5332 |

TD E2E has the best AUROC — it ranks TPs above FPs better than any other run. This is the paradox: E2E models can't classify but can rank. Two-stage models classify well but rank worse because `det_score × cls_conf` is a noisier scoring function.

**TP/FP score separation:**

| Run | TP mean | FP mean | TP/FP ratio |
|-----|:-------:|:-------:|:-----------:|
| TD E2E (Run 6) | 0.058 | 0.023 | **2.5×** |
| TD + V-JEPA2 (Run 5) | 0.054 | 0.023 | 2.4× |
| TD + MLP (Run 5c) | 0.056 | 0.025 | 2.3× |
| AF E2E (baseline) | 0.066 | 0.036 | 1.9× |
| AF + V-JEPA2 (Run 1) | 0.054 | 0.029 | 1.9× |
| AF + MLP (Run 3c) | 0.056 | 0.031 | 1.8× |

TD E2E achieves the best TP/FP score ratio, reinforcing its ranking advantage.

**Error taxonomy — E2E vs Two-Stage side-by-side:**

| Error Type | AF E2E | AF + V-JEPA2 | AF + MLP | TD E2E | TD + V-JEPA2 | TD + MLP |
|------------|:------:|:------------:|:--------:|:------:|:------------:|:--------:|
| Correct | 2.5% | 4.7% | 4.4% | 2.6% | 5.2% | 4.7% |
| Boundary | 2.3% | 4.4% | 3.9% | 2.3% | 4.3% | 3.7% |
| Wrong class | 11.1% | 4.0% | 4.8% | 11.4% | 4.0% | 5.1% |
| Low IoU | 19.0% | 16.5% | 16.5% | 19.3% | 16.7% | 16.7% |
| No overlap | 65.1% | 70.3% | 70.3% | 64.5% | 69.8% | 69.8% |

The pattern is consistent across both detectors: two-stage doubles the correct rate and cuts wrong-class errors by ~3×, while E2E retains a ranking edge.

---

## Resolved: TriDet E2E (Run 6) Predictions

TriDet E2E was originally trained with `save_dict=False`, so no `result_detection.json` was generated. Re-ran test with `--cfg-options post_processing.save_dict=True`.

SLURM jobs 11018241–43 completed successfully (2026-03-26). Note: `best.pth` is saved at the val loss minimum, not peak mAP epoch, so fold-level mAP differs slightly from RUNS.md:

| Fold | best.pth mAP | RUNS.md (peak epoch) |
|------|:------------:|:--------------------:|
| 0 | 18.46% | 20.89% |
| 1 | 18.56% | 22.90% |
| 2 | 15.17% | 15.37% |

Output: `OpenTAD/exps/sails_rmm/tridet_vjepa_balanced_fold{0,1,2}/gpu1_id99/result_detection.json`
