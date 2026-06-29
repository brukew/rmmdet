## Fusion Deep Dive (Clip-level vs Window-level)

This document validates and refines the claims in **§6.1 Fusion: What It Actually Changes** using the saved Chapter 5 tables and the window/TAL prediction artifacts.

- **Repro script**: `actreg/insights/paper_plots/fusion_deep_dive.py`
- **Computed outputs**: `actreg/insights/tables/fusion_deep_dive.json`

### Executive Answer (Is the section “true”?)

**Mostly true**, with two important refinements:

- **The “background dominance” story is correct**, but the exact background prediction rates depend on the specific single-modality model variant (balanced vs. non-balanced). In this repo, V-JEPA is closer to **~79–81% background predictions**, PoseC3D (CE) is **~89%**, while the main fusion window model is **~44%**.
- **Fusion’s segment-level mAP gain is real but class-specific**: the **avg mAP gain is modest** (+0.41 points), and it is driven primarily by **large AP gains for Spinning** (and a smaller gain for Rocking), while **Hands Flapping and Jumping AP drop**.

---

## 6.1.1 Complementary Information & Complementary Errors (Clip-level)

### What the clip-level tables actually show

At clip level, fusion is consistently strong (and ranks #1 in both 4-class and 5-class clip-level comparisons). The clearest “complementarity” evidence is that fusion **dramatically improves recall** for the classes where single models struggle most.

#### 4-class per-class recall (Fusion vs single modalities)

From `table_5_3_4class_pr.json`:

- **Jumping recall**: **83.7%** (fusion) vs **68.4%** (V-JEPA) / **65.2%** (PoseC3D) / **72.1%** (STGCN++)  
  - Δ vs V-JEPA: **+15.3**
- **Rocking recall**: **76.7%** (fusion) vs **61.3%** (V-JEPA) / **47.1%** (PoseC3D) / **48.7%** (STGCN++)  
  - Δ vs V-JEPA: **+15.4**
- **Spinning recall**: **87.2%** (fusion) vs **78.3%** (V-JEPA) / **74.4%** (PoseC3D) / **74.0%** (STGCN++)  
  - Δ vs V-JEPA: **+8.9**
- **Hands Flapping recall**: **86.9%** (fusion) vs **84.0%** (V-JEPA) / **86.7%** (PoseC3D) / **89.1%** (STGCN++)
  - Fusion is **above V-JEPA**, roughly tied with PoseC3D, and slightly below STGCN++.

#### Precision impact (4-class)

Fusion **does not “catastrophically” hurt precision**. In fact, for Hands Flapping and Jumping it increases precision substantially relative to V-JEPA:

- **Hands Flapping precision**: **92.3%** (fusion) vs **80.5%** (V-JEPA)
- **Jumping precision**: **74.3%** (fusion) vs **64.8%** (V-JEPA)

Precision dips slightly for Spinning and Rocking relative to V-JEPA, but remains high (**~77–87%**).

### Deeper insight: complementarity is *not uniform across classes*

Even at clip level, fusion does **not** simply “improve everything equally.” The **largest gains** are concentrated in **Jumping** and **Rocking** recall, suggesting:

- Some classes require **scene/context cues** (RGB) *and* **kinematics** (pose/skeleton) to disambiguate hard cases.
- The “complementary errors” story is strongest where single models have **systematic blind spots**, not where performance is already saturated.

---

## 6.1.2 The Fusion Paradox Under Background Dominance (Window-level)

### Ground truth imbalance (validation windows)

Across folds 0–2, **90.42%** of validation windows are background:

- fold0: 90.81%  
- fold1: 88.43%  
- fold2: 91.53%  
- overall: **12790 / 14145 = 90.42%**

This makes **accuracy** a weak proxy for “RMM usefulness.”

### What fusion changes at window level: it “moves probability mass off background”

From `fusion_deep_dive.json` (computed from `tal_format_preds.csv` / `fused_tal_format_preds.csv` on fold val windows):

| Model | Predicted BG rate | Top-1 Acc | Macro-F1 (RMM-only) | Rocking (P/R/F1) | Spinning (P/R/F1) |
|---|---:|---:|---:|---:|---:|
| V-JEPA (Balanced) | **79.2%** | **80.9%** | **23.7%** | 5.6 / 8.2 / 6.7 | 11.3 / 22.9 / 13.4 |
| PoseC3D (CE) | **88.9%** | **85.9%** | **24.0%** | 0.0 / 0.0 / 0.0 | 18.8 / 26.4 / 21.0 |
| Fusion (V-JEPA + PoseC3D MLP) | **43.8%** | **48.1%** | **18.8%** | 1.8 / 41.9 / 3.4 | 4.8 / 54.6 / 8.7 |

**This directly supports the key mechanism in the claim**:

- Single-modality models are **background-dominant** (≈79–89% of their argmax predictions are background), which helps accuracy but suppresses RMM sensitivity.
- Fusion shifts sharply **away from background** (≈44% background predictions), which raises RMM recall (especially Rocking and Spinning) but causes a flood of background→RMM false positives.

### Why fusion looks “worse” at window level (despite higher RMM recall)

The critical detail is **which macro-F1 is being reported**:

- In `table_5_5_window_detection.json`, **macro F1 is computed over the 4 RMM classes only** (background excluded).  
  - This means background false alarms matter *a lot* because they destroy **RMM precision**.

Fusion is the extreme case: it becomes **recall-heavy but precision-poor** for the rare classes.

- **Rocking**: recall jumps to **41.9%**, but precision collapses to **1.8%** (F1 ≈ **3.4%**).
- **Spinning**: recall rises to **54.6%**, but precision is only **4.8%** (F1 ≈ **8.7%**).

So even though fusion “finds” many more true RMM windows, the metric is dominated by the enormous number of **false positives drawn from a 90% background pool**.

### A deeper (and important) refinement to the narrative

The window-level fusion result is not just “more false alarms” in the abstract:

- It is **highly class-skewed**: the biggest precision collapses are in **Rocking** and **Spinning**.
- This suggests a specific failure mode: at window scale, the fused representation makes it too easy to over-trigger these labels on non-RMM motion (camera motion, adult motion, partial child motion, etc.), especially when decision boundaries are not calibrated for background-heavy deployment.

---

## 6.1.2 → Why this trade-off can still help TAL (segment-level)

Window-level fusion can be thought of as a **proposal generator**: it “lights up” more candidate RMM regions. Segment-level TAL mAP can recover some utility because postprocessing can merge/suppress isolated errors.

Evidence from the saved best-eval postprocessing configs (`foldX/best_eval/eval_config.json`):

- `V-JEPA (Balanced)`: **thr=0.3**, smooth_k=1, merge_gap=0.5  
- `Fusion (V-JEPA + PoseC3D MLP)`: **thr=0.4**, smooth_k=1, merge_gap=1.0

So the fusion backbone is evaluated with a **stricter threshold** and **larger merge gap**, consistent with the idea that it needs stronger filtering to control proposal noise.

---

## 6.1.3 Segment-level mAP: the “paradox” is real, but the gains come from specific classes

From `table_5_7_segment_tal_map.json`:

- **avg mAP**: fusion **6.36** vs V-JEPA **5.95** (Δ **+0.41**)

But the per-class AP deltas show what is *actually* happening (tIoU = 0.3 shown; similar pattern at 0.5):

- **Spinning AP**: **+6.94** (12.63 vs 5.69)
- **Rocking AP**: **+1.00** (2.94 vs 1.94)
- **Hands Flapping AP**: **−2.19**
- **Jumping AP**: **−3.90**

### Deeper insight: fusion is acting like a “tail-class booster”

The segment-level improvement is driven mostly by **Spinning** (and secondarily Rocking), i.e. the classes that are hardest for single-modality window backbones. This is consistent with a complementary-information hypothesis, but it also means:

- The “fusion paradox” is not that fusion is universally better; it **redistributes** performance.
- If your downstream objective weights classes equally (mAP does), improving tail classes can outweigh losses on head classes.

---

## 6.1.4 What to revise in the draft claims (suggested edits)

### Correct the background prediction-rate numbers

Replace:

- “V-JEPA2 and PoseC3D predict background 84–87% of the time”

With a more accurate (and defensible) statement:

- “In our window-level experiments, single-modality models predict background for roughly **~79–89%** of windows (depending on the variant), while fusion predicts background for only **~44%** of windows.”

### Clarify the macro-F1 definition

Add one sentence to avoid confusion:

- “Macro-F1 for window detection is computed over the **RMM classes only (background excluded)**, so false positives drawn from the dominant background pool strongly depress the score.”

### Make the TAL implication more precise

Instead of “fusion improves TAL mAP in general,” use:

- “Fusion yields a small avg mAP gain that is **primarily driven by Spinning (and Rocking)** AP improvements, even though it reduces Hands Flapping and Jumping AP.”

---

## 6.1.5 Additional “meaningful” insights beyond the draft

1) **Window-level fusion is poorly calibrated for rare labels**  
   The fusion model’s Rocking and Spinning precisions (≈1.8% and ≈4.8%) are so low that it is effectively treating these classes as “default explanations” for many background windows. This is a calibration / decision-boundary problem, not just “noise.”

2) **Fusion helps TAL by increasing *coverage*, not by being cleaner**  
   The best-eval postprocessing uses a stricter threshold and merging, implying that a substantial portion of fusion’s raw window predictions are not directly usable without filtering.

3) **The clip→window generalization gap is the real story**  
   Clip-level fusion improves precision/recall cleanly; window-level fusion becomes recall-heavy and precision-poor. That gap strongly suggests the deployment setting (background-heavy streaming windows) is what turns “complementarity” into the paradox.

---

## 6.1.6 (Optional context) Proposal explosion / weak calibration is also real

The draft’s §6.1.3 claim about proposal explosion is consistent with the dedicated ActionFormer failure analysis:

- In `actreg/insights/writeup/tal_failure_analysis.md`, at **score ≥ 0.1** and **tIoU ≥ 0.3**, about **~79–80%** of predicted segments do **not** overlap any GT segment.
- Overlapping vs non-overlapping predictions have **very similar score distributions** (multiclass medians **0.035 vs 0.032**), which implies weak ranking / calibration.

This matters for the fusion story because it reinforces the same lesson: **downstream TAL quality depends heavily on ranking/calibration**, not just raw recall.


