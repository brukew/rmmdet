# VLM Prompt Ablation Study — RMM Type Classification

**Date**: 2026-03-22
**Model**: Qwen3-VL-30B-A3B-Instruct (MoE, native video input)
**Task**: 4-class RMM type classification (hands flapping, rocking, spinning, jumping)
**Data**: fold_0_val.csv, n=218 clips, canonical pre-cut clips
**Config**: do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=64

---

## Prompts Tested

### P1 — Full definitions (baseline)
```
Identify the repetitive motor movement (RMM) shown in the video.
Choose exactly one option from the list and reply as `RMM: <option>` with no extra words.

- hands flapping: repetitive movement of the hands at the wrists either vertically or horizontally.
- rocking: repetitive front-to-back or side-to-side movement of the body.
- spinning: repetitive turning of the body in a circular motion.
- jumping: repetitive bouncing of the body involving bending the knees and feet leaving or nearly leaving the floor.
```

### P2 — Structured with clinical context
```
You are analyzing a child's repetitive motor behavior. Watch the video carefully and classify the movement.

Categories:
- hands flapping: both hands flap repeatedly at the wrists (vertical or horizontal)
- rocking: body rocks front-to-back or side-to-side repeatedly
- spinning: body turns in circles repeatedly
- jumping: body bounces up repeatedly, knees bend, feet near or off the floor

Reply with only: `RMM: <category>`
```

### P4 — Zero-shot minimal
```
Classify the repetitive motor movement in this video as one of: hands flapping, rocking, spinning, jumping.
Reply as `RMM: <answer>` only.
```

### P5 — Expert role + chain-of-thought
```
You are a certified pediatric behavioral analyst specializing in repetitive motor movements (RMMs)
in children with autism spectrum disorder. You have 20 years of clinical experience coding movement
behaviors from video.

Watch the video carefully. Before answering, briefly reason through what you observe:
1. Which body part(s) are moving?
2. Is the movement bilateral or unilateral?
3. What is the movement's direction and pattern?

Then classify as exactly one of:
- hands flapping: repetitive movement of the hands at the wrists either vertically or horizontally.
- rocking: repetitive front-to-back or side-to-side movement of the body.
- spinning: repetitive turning of the body in a circular motion.
- jumping: repetitive bouncing of the body involving bending the knees and feet leaving or nearly leaving the floor.

End your response with: `RMM: <answer>`
```

### P6 — Contrastive / disambiguation-focused
```
You are an expert annotator classifying repetitive motor movements in children.
The hardest distinctions are:
- "hands flapping" (BOTH hands) vs "one hand flap" (only ONE hand) — watch carefully for unilateral vs bilateral
- "jumping" involves the WHOLE body leaving or nearly leaving the floor — not just bouncing in place
- "rocking" is a weight-shift movement, not circular

Given this video, classify the movement as exactly one of:
- hands flapping: repetitive movement of the hands at the wrists either vertically or horizontally.
- rocking: repetitive front-to-back or side-to-side movement of the body.
- spinning: repetitive turning of the body in a circular motion.
- jumping: repetitive bouncing of the body involving bending the knees and feet leaving or nearly leaving the floor.

Reply only as: `RMM: <answer>`
```

---

## Evaluation Note — GT Label Mismatch

The val set (n=218) contains **29 clips labelled "one hand flap"** in the ground truth. All 4-class prompts (P1/P2/P4/P5/P6) cannot predict this label, so those 29 clips are always counted as wrong — artificially deflating all metrics.

Two sets of metrics are reported:
- **Raw**: ground truth used as-is (penalizes "one hand flap" clips)
- **Merged**: GT "one hand flap" → "hands flapping" before scoring (fair 4-class evaluation)

The **Merged** figures should be treated as canonical for 4-class comparisons.

---

## Results

Two sets of runs were performed:
- **Run A** (`max_pixels=151200`, 3 runs each): over-constrained resolution
- **Run B** (no `max_pixels`, default `VIDEO_TOTAL_PIXELS` budget, 3 runs each): full resolution per frame

### Run A — max_pixels=151200, Raw GT (avg ± std)

| Prompt | Top-1 Acc | Macro F1 | Cohen's κ |
|--------|-----------|----------|-----------|
| P1 Full definitions | 0.506 ± 0.007 | 0.505 ± 0.010 | 0.317 ± 0.012 |
| P2 Structured+context | 0.554 ± 0.011 | 0.517 ± 0.025 | 0.313 ± 0.014 |
| P4 Zero-shot minimal | 0.549 ± 0.012 | 0.513 ± 0.018 | 0.326 ± 0.016 |
| P5 Expert CoT | 0.520 ± 0.005 | 0.468 ± 0.016 | 0.293 ± 0.005 |
| P6 Contrastive | 0.543 ± 0.010 | 0.511 ± 0.020 | 0.314 ± 0.014 |

### Run B — no max_pixels, Raw GT (avg ± std)

| Prompt | Top-1 Acc | Macro F1 | Cohen's κ |
|--------|-----------|----------|-----------|
| P1 Full definitions | 0.523 ± 0.017 | 0.521 ± 0.023 | 0.335 ± 0.022 |
| **P2 Structured+context** | **0.569 ± 0.009** | **0.539 ± 0.014** | **0.338 ± 0.012** |
| P4 Zero-shot minimal | 0.540 ± 0.005 | 0.486 ± 0.009 | 0.309 ± 0.009 |
| P5 Expert CoT | 0.534 ± 0.019 | 0.504 ± 0.030 | 0.312 ± 0.028 |
| P6 Contrastive | 0.549 ± 0.007 | 0.519 ± 0.007 | 0.318 ± 0.014 |

### Run B — no max_pixels, Merged GT (one hand flap → hands flapping) (avg ± std)

| Prompt | Top-1 Acc | Macro F1 | Cohen's κ |
|--------|-----------|----------|-----------|
| P1 Full definitions | 0.599 ± 0.016 | 0.534 ± 0.017 | 0.394 ± 0.022 |
| **P2 Structured+context** | **0.672 ± 0.014** | **0.554 ± 0.021** | **0.416 ± 0.022** |
| P4 Zero-shot minimal | 0.649 ± 0.013 | 0.524 ± 0.020 | 0.399 ± 0.023 |
| P5 Expert CoT | 0.637 ± 0.016 | 0.515 ± 0.028 | 0.396 ± 0.027 |
| P6 Contrastive | 0.649 ± 0.011 | 0.537 ± 0.015 | 0.392 ± 0.018 |

Removing the pixel cap improved all prompts. **P2 wins across all metrics** at full resolution. Under fair (merged) evaluation, true top-1 accuracy is ~6–10pp higher across all prompts.

> Note: Top-2 accuracy equals Top-1 for all runs — bug in top-2 metric for single-output (non-window) inference mode. To fix.

---

## Per-Class Metrics (best run per prompt)

### P4 Zero-shot — run 1 (top1=0.560)
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping | 0.596 | 0.827 | 0.692 |
| rocking | 0.600 | 0.387 | 0.471 |
| spinning | 0.353 | 0.800 | 0.490 |
| jumping | 0.607 | 0.378 | 0.466 |

### P6 Contrastive — run 2 (top1=0.550)
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping | 0.600 | 0.857 | 0.706 |
| rocking | 0.667 | 0.194 | 0.300 |
| spinning | 0.500 | 0.733 | 0.595 |
| jumping | 0.576 | 0.422 | 0.487 |

---

## Key Findings

1. **Removing max_pixels helps consistently** — all prompts improved when using full resolution (default `VIDEO_TOTAL_PIXELS` budget).
2. **P2 (structured+context) wins** at full resolution — raw top1=0.569, F1=0.539, κ=0.338; merged top1=0.672, F1=0.554, κ=0.416.
3. **GT label mismatch inflates error** — 29 "one hand flap" clips in the val set are never correctly matchable in 4-class runs. Under fair (merged) evaluation, all prompts gain ~6–10pp in accuracy and ~8pp in kappa.
4. **P4 (zero-shot minimal) dropped** with more resolution — extra visual detail may require more instructional context to interpret correctly.
5. **P5 (expert CoT) underperforms** — likely because `max_new_tokens=64` truncates the chain-of-thought before the final `RMM:` label. Needs `max_new_tokens≥256` to work properly.
6. **P6 (contrastive)** improves spinning recall (0.733 vs 0.800 for P4) but hurts rocking recall badly (0.194).
7. All prompts show high hands-flapping recall (>0.82) due to class imbalance (55% of val set).
8. **Spinning is consistently the hardest class** — zero recall in earlier 5-class frame-sampled runs; here improved to 0.73-0.80 with native video input.

---

## P2 Variant Runs (single run each, 2026-03-23)

Four variants of P2 were tested to explore prompt modifications. All use Qwen3-VL-30B-A3B-Instruct, native video input, no `max_pixels`, `do_sample=True`.

### Summary (Merged GT)

| Config | Labels | Top-1 Acc | Macro F1 | Cohen's κ |
|--------|--------|-----------|----------|-----------|
| **P2 baseline** (avg, ref) | 4-class | 0.672 | 0.554 | 0.416 |
| **p2_merged_flap** | 4-class | **0.688** | **0.585** | **0.466** |
| p2_5class | 5-class (merged) | 0.661 | 0.541 | 0.421 |
| p2_none_class | 4-class + none | 0.463 | 0.477 | 0.269 |

> Raw (unmerged) metrics: p2_merged_flap=0.583/0.561/0.374, p2_5class=0.564/0.465/0.365, p2_none_class=0.417/0.380/0.258

### Per-Class Metrics (Merged GT)

#### p2_merged_flap
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping (n=127) | 0.794 | 0.850 | 0.821 |
| rocking (n=31) | 0.517 | 0.484 | 0.500 |
| spinning (n=15) | 0.480 | 0.800 | 0.600 |
| jumping (n=45) | 0.556 | 0.333 | 0.417 |

#### p2_5class (one hand flap → hands flapping post-hoc)
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping (n=127) | 0.785 | 0.835 | 0.809 |
| rocking (n=31) | 0.469 | 0.484 | 0.476 |
| spinning (n=15) | 0.417 | 0.667 | 0.513 |
| jumping (n=45) | 0.500 | 0.289 | 0.366 |

#### p2_none_class (merged GT)
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping (n=127) | 0.810 | 0.535 | 0.645 |
| rocking (n=31) | 0.500 | 0.065 | 0.114 |
| spinning (n=15) | 0.579 | 0.733 | 0.647 |
| jumping (n=45) | 0.571 | 0.444 | 0.500 |

#### p2_frame_sampled (4 windows × 8 frames, qwen3 wrapper, merged GT)
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| hands flapping (n=127) | 0.778 | 0.882 | 0.827 |
| rocking (n=31) | 0.420 | 0.677 | 0.519 |
| spinning (n=15) | 0.714 | 0.333 | 0.455 |
| jumping (n=45) | 0.588 | 0.222 | 0.323 |

### Summary (updated with frame-sampled)

| Config | Labels | Top-1 Acc (merged) | Macro F1 (merged) | Cohen's κ (merged) |
|--------|--------|--------------------|-------------------|---------------------|
| **P2 baseline** (avg, ref) | 4-class native video | 0.672 | 0.554 | 0.416 |
| **p2_merged_flap** | 4-class native video | **0.688** | **0.585** | **0.466** |
| p2_frame_sampled | 4-class 4w×8f sampled | 0.679 | 0.531 | 0.431 |
| p2_5class (merged) | 5-class → 4-class | 0.661 | 0.541 | 0.421 |
| p2_none_class | 4-class + none | 0.463 | 0.477 | 0.269 |

### Key Takeaways

- **p2_merged_flap is the best single run** (κ=0.466) — merging bilateral/unilateral description into the hands flapping definition outperforms baseline. Best spinning recall (0.80) and hands flapping F1 (0.821).
- **p2_frame_sampled** (4w×8f) is competitive with native video (κ=0.431 vs 0.416 baseline) but falls behind p2_merged_flap. Rocking recall notably better (0.677 vs 0.484) but spinning recall collapses (0.333 vs 0.800) — frame sampling misses the rotational pattern that native video captures.
- **p2_5class** — adding "one hand flap" as a separate class hurts macro F1 (only 5/29 one-hand clips identified correctly, recall=0.17). Post-hoc merging recovers accuracy (0.661) and kappa (0.421), confirming the model sees hand movement correctly but can't reliably distinguish laterality.
- **p2_none_class** — "none" escape class severely hurts performance (κ=0.269). Model routes 58 clips to "none" despite none existing in GT. Rocking recall collapses to 0.065. The "none" option acts as a hedge rather than a genuine uncertainty signal.

---

## Cross-Fold Merged-Flap Runs (2026-03-24)

Three prompt styles with the merged hands flapping description tested across all 3 folds, 2 runs each (6 runs per prompt). All use Qwen3-VL-30B-A3B-Instruct, native video input, no `max_pixels` cap, `do_sample=True` (temp=0.7).

**Hands flapping definition used**: "one or both hands flap repeatedly at the wrists, either vertically or horizontally"

**Fold class distributions (n=218 each):**

| Class | Fold 0 | Fold 1 | Fold 2 |
|-------|--------|--------|--------|
| hands flapping | 98 (+ 29 one hand flap → merged) | 79 (+ 34 → merged) | 69 (+ 34 → merged) |
| jumping | 45 | 65 | 60 |
| rocking | 31 | 24 | 41 |
| spinning | 15 | 16 | 14 |

### Summary (merged GT, avg ± std across 6 runs / 3 folds)

| Prompt | Top-1 Acc | Macro F1 | Cohen's κ |
|--------|-----------|----------|-----------|
| **p2_merged_flap** | **0.622 ± 0.051** | **0.536 ± 0.070** | **0.375 ± 0.071** |
| p4_merged_flap | 0.624 ± 0.040 | 0.477 ± 0.068 | 0.338 ± 0.069 |
| p5_merged_flap | 0.568 ± 0.050 | 0.467 ± 0.049 | 0.331 ± 0.064 |

> High std reflects genuine fold difficulty differences, not sampling noise — fold 1 is consistently harder for all prompts.

### Per-Fold Breakdown (merged GT, avg ± std over 2 runs)

| Prompt | Fold 0 Acc | Fold 1 Acc | Fold 2 Acc |
|--------|-----------|-----------|-----------|
| p2_merged_flap | 0.677 ± 0.025 | 0.557 ± 0.002 | 0.631 ± 0.007 |
| p4_merged_flap | 0.661 ± 0.000 | 0.569 ± 0.000 | 0.642 ± 0.009 |
| p5_merged_flap | 0.633 ± 0.000 | 0.512 ± 0.007 | 0.560 ± 0.009 |

| Prompt | Fold 0 F1 | Fold 1 F1 | Fold 2 F1 |
|--------|----------|----------|----------|
| p2_merged_flap | 0.568 ± 0.035 | 0.443 ± 0.002 | 0.597 ± 0.003 |
| p4_merged_flap | 0.471 ± 0.004 | 0.400 ± 0.000 | 0.562 ± 0.023 |
| p5_merged_flap | 0.506 ± 0.000 | 0.401 ± 0.005 | 0.493 ± 0.021 |

| Prompt | Fold 0 κ | Fold 1 κ | Fold 2 κ |
|--------|---------|---------|---------|
| p2_merged_flap | 0.416 ± 0.050 | 0.284 ± 0.003 | 0.427 ± 0.009 |
| p4_merged_flap | 0.339 ± 0.001 | 0.254 ± 0.005 | 0.421 ± 0.018 |
| p5_merged_flap | 0.400 ± 0.002 | 0.248 ± 0.015 | 0.344 ± 0.017 |

### Per-Class Metrics (merged GT, all folds, avg ± std)

#### p2_merged_flap

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| hands flapping | 0.682 ± 0.037 | 0.821 ± 0.038 | 0.745 ± 0.038 |
| rocking | 0.459 ± 0.126 | 0.437 ± 0.079 | 0.440 ± 0.089 |
| spinning | 0.491 ± 0.104 | 0.679 ± 0.090 | 0.565 ± 0.088 |
| jumping | 0.598 ± 0.093 | 0.299 ± 0.112 | 0.393 ± 0.120 |

#### p4_merged_flap

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| hands flapping | 0.648 ± 0.031 | 0.916 ± 0.017 | 0.759 ± 0.026 |
| rocking | 0.537 ± 0.151 | 0.223 ± 0.087 | 0.305 ± 0.098 |
| spinning | 0.385 ± 0.059 | 0.589 ± 0.037 | 0.464 ± 0.049 |
| jumping | 0.747 ± 0.112 | 0.262 ± 0.112 | 0.381 ± 0.138 |

#### p5_merged_flap

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| hands flapping | 0.703 ± 0.065 | 0.724 ± 0.034 | 0.713 ± 0.047 |
| rocking | 0.392 ± 0.125 | 0.258 ± 0.056 | 0.303 ± 0.074 |
| spinning | 0.272 ± 0.049 | 0.737 ± 0.135 | 0.397 ± 0.072 |
| jumping | 0.566 ± 0.076 | 0.389 ± 0.106 | 0.454 ± 0.086 |

### Individual Runs

| Run ID | Prompt | Fold | n | Top-1 Acc | Macro F1 | Balanced Acc | κ |
|--------|--------|------|---|-----------|----------|--------------|---|
| 20260324_0109_itxl | p2_merged_flap | fold_0 | 218 | 0.6514 | 0.5327 | 0.5563 | 0.3661 |
| 20260324_0109_wvcj | p2_merged_flap | fold_0 | 218 | 0.7018 | 0.6026 | 0.6192 | 0.4654 |
| 20260324_0109_wjbe | p2_merged_flap | fold_1 | 218 | 0.5596 | 0.4454 | 0.4909 | 0.2865 |
| 20260324_0127_gtsh | p2_merged_flap | fold_1 | 218 | 0.5550 | 0.4412 | 0.4950 | 0.2809 |
| 20260324_0130_lyfj | p2_merged_flap | fold_2 | 218 | 0.6376 | 0.5997 | 0.5986 | 0.4356 |
| 20260324_0145_kyta | p2_merged_flap | fold_2 | 218 | 0.6239 | 0.5936 | 0.5930 | 0.4175 |
| 20260324_0146_jjsc | p4_merged_flap | fold_0 | 218 | 0.6606 | 0.4668 | 0.4832 | 0.3392 |
| 20260324_0146_obow | p4_merged_flap | fold_0 | 218 | 0.6606 | 0.4753 | 0.4771 | 0.3379 |
| 20260324_0148_hzpu | p4_merged_flap | fold_1 | 218 | 0.5688 | 0.3992 | 0.4426 | 0.2484 |
| 20260324_0202_hpjg | p4_merged_flap | fold_1 | 218 | 0.5688 | 0.3998 | 0.4543 | 0.2592 |
| 20260324_0202_ajks | p4_merged_flap | fold_2 | 218 | 0.6514 | 0.5842 | 0.5838 | 0.4386 |
| 20260324_0206_zbau | p4_merged_flap | fold_2 | 218 | 0.6330 | 0.5390 | 0.5444 | 0.4024 |
| 20260324_0206_nrai | p5_merged_flap | fold_0 | 218 | 0.6330 | 0.5059 | 0.5857 | 0.3980 |
| 20260324_0220_gfvi | p5_merged_flap | fold_0 | 218 | 0.6330 | 0.5068 | 0.5707 | 0.4016 |
| 20260324_0220_zfvu | p5_merged_flap | fold_1 | 218 | 0.5183 | 0.4057 | 0.4611 | 0.2626 |
| 20260324_0223_kvlf | p5_merged_flap | fold_1 | 218 | 0.5046 | 0.3958 | 0.4411 | 0.2326 |
| 20260324_0233_esnt | p5_merged_flap | fold_2 | 218 | 0.5688 | 0.5140 | 0.5843 | 0.3614 |
| 20260324_0244_jwwm | p5_merged_flap | fold_2 | 218 | 0.5505 | 0.4717 | 0.5180 | 0.3268 |

### Key Takeaways

- **P2 wins on macro F1 and kappa** — structured clinical context with merged flap description remains the best prompt style.
- **P4 matches P2 on accuracy** but macro F1 lags by 6pp — zero-shot recovers correct class roughly as often but distributes errors less evenly (hands flapping recall 0.916 but rocking recall collapses to 0.223).
- **P5 underperforms** — CoT instructions are ignored entirely; model outputs bare 1–2 word answers (avg 11 chars) regardless of the reasoning prompt. This is a known limitation of Qwen3-VL in video mode — `/think` token would be needed to trigger actual reasoning.
- **Fold 1 is consistently the hardest** (~0.05–0.09 lower κ vs folds 0/2). Fold 1 has more jumping (n=65 vs 45/60) and fewer hands flapping (n=79 vs 98/69) — harder class distribution.
- **Jumping recall remains the main bottleneck** across all prompts (0.299–0.389). Rocking is the second weakest (0.223–0.437). Both are candidate targets for description improvements.

---

## Temperature & Model Size Ablation (2026-03-26)

p2_merged_flap prompt held fixed; varying temperature and model size. All runs use Qwen3-VL native video input, no `max_pixels` cap, all 3 folds (n=218 each), merged GT evaluation.

### Summary (avg ± std across folds)

| Variant | n runs | Top-1 Acc | Macro F1 | Balanced Acc | κ |
|---------|--------|-----------|----------|--------------|---|
| temp=0.7, 30B (baseline) | 6 | 0.6216 ± 0.0514 | 0.5359 ± 0.0695 | 0.5588 ± 0.0501 | 0.3753 ± 0.0712 |
| **temp=0.3, 30B** | **6** | **0.6330 ± 0.0583** | **0.5590 ± 0.0729** | **0.5821 ± 0.0472** | **0.4068 ± 0.0769** |
| greedy, 30B | 3 | 0.6269 ± 0.0561 | 0.5504 ± 0.0692 | 0.5732 ± 0.0480 | 0.3949 ± 0.0743 |
| temp=1.0, 30B | 6 | 0.6055 ± 0.0563 | 0.5311 ± 0.0542 | 0.5500 ± 0.0358 | 0.3681 ± 0.0625 |
| temp=0.7, 8B | 3 | 0.5948 ± 0.0495 | 0.4241 ± 0.0277 | 0.4280 ± 0.0136 | 0.3201 ± 0.0488 |

### Individual Runs

| Run ID | Variant | Fold | n | Top-1 Acc | Macro F1 | Balanced Acc | κ |
|--------|---------|------|---|-----------|----------|--------------|---|
| 20260326_0857_yipg | temp_greedy | fold_0 | 218 | 0.6835 | 0.5776 | 0.5977 | 0.4404 |
| 20260326_0907_tqpj | temp_greedy | fold_1 | 218 | 0.5505 | 0.4555 | 0.5062 | 0.2901 |
| 20260326_1823_ieiz | temp_greedy | fold_2 | 218 | 0.6468 | 0.6182 | 0.6158 | 0.4543 |
| 20260326_1035_pskz | temp_03 | fold_0 | 218 | 0.6835 | 0.5777 | 0.5977 | 0.4447 |
| 20260326_1844_vsbb | temp_03 | fold_0 | 218 | 0.6927 | 0.5919 | 0.6078 | 0.4532 |
| 20260326_1155_bxmo | temp_03 | fold_1 | 218 | 0.5505 | 0.4637 | 0.5196 | 0.2952 |
| 20260326_1210_vnvh | temp_03 | fold_1 | 218 | 0.5550 | 0.4563 | 0.5149 | 0.3034 |
| 20260326_1225_dzaq | temp_03 | fold_2 | 218 | 0.6514 | 0.6241 | 0.6182 | 0.4612 |
| 20260326_1242_dfos | temp_03 | fold_2 | 218 | 0.6651 | 0.6404 | 0.6346 | 0.4832 |
| 20260326_1307_xevm | temp_10 | fold_0 | 218 | 0.6881 | 0.5924 | 0.6008 | 0.4508 |
| 20260326_1903_nmyg | temp_10 | fold_0 | 218 | 0.6560 | 0.5375 | 0.5357 | 0.3911 |
| 20260326_1307_lrct | temp_10 | fold_1 | 218 | 0.5459 | 0.4653 | 0.5171 | 0.3011 |
| 20260326_1322_hqqs | temp_10 | fold_1 | 218 | 0.5275 | 0.4511 | 0.4984 | 0.2692 |
| 20260326_1323_ndff | temp_10 | fold_2 | 218 | 0.6101 | 0.5670 | 0.5701 | 0.4010 |
| 20260326_1326_sril | temp_10 | fold_2 | 218 | 0.6055 | 0.5733 | 0.5777 | 0.3952 |
| 20260326_1341_qmht | p2_8b | fold_0 | 218 | 0.6606 | 0.4481 | 0.4337 | 0.3685 |
| 20260326_1350_vuwl | p2_8b | fold_1 | 218 | 0.5413 | 0.3853 | 0.4092 | 0.2533 |
| 20260326_1812_dkva | p2_8b | fold_2 | 218 | 0.5826 | 0.4389 | 0.4410 | 0.3386 |

### Key Takeaways

- **temp=0.3 is best** across all 4 metrics — lower temperature sharpens the distribution without collapsing to greedy; best single run was temp_03/fold_2 (κ=0.483).
- **Greedy is second** — deterministic and competitive, nearly matching temp=0.3 with zero variance within a fold.
- **temp=1.0 hurts** — higher entropy degrades all metrics; slight uptick in invalid responses (up to 2/218 per run).
- **8B is significantly weaker** — −11pp Macro F1 and −13pp Balanced Acc vs 30B. Top-1 Acc is similar but 8B fails on minority classes (rocking, spinning); 30B is much better calibrated.
- **Invalid rate is negligible** — max 5/218 (2.3%) for 8B fold_0, ≤1/218 for all 30B runs.

---

## Outstanding Issues

- **Top-2 accuracy broken** for native video mode (single output, no window voting). Need to fix.
- **max_new_tokens=64** too short for P5 CoT — rerun with 256+.
- **max_pixels=151200** was under-constraining resolution — resolved, use default `VIDEO_TOTAL_PIXELS` budget.
- **GT label mismatch**: val set contains 29 "one hand flap" clips; 4-class runs should use merged GT for canonical reporting. Consider fixing the val CSV to merge these upstream.
