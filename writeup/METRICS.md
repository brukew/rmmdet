# Metrics

Below are definitions of the evaluation metrics used across tasks. For each task, results are typically summarized as **mean ± std across CV folds**.

## Action Recognition

### Definitions

- **Top-1 Accuracy**: fraction of examples where the **highest-scoring** predicted class matches the ground-truth class.
- **Top-2 Accuracy**: fraction of examples where the ground-truth class is among the **two highest-scoring** predicted classes. This is useful when classes are visually similar and the model’s second choice is often reasonable.

- **Precision (per class)**: in a one-vs-rest view for class \(c\),
  - \( \text{Precision}_c = \frac{TP_c}{TP_c + FP_c} \)
  - Interprets: “when the model predicts class \(c\), how often is it correct?”

- **Recall (per class)**: in a one-vs-rest view for class \(c\),
  - \( \text{Recall}_c = \frac{TP_c}{TP_c + FN_c} \)
  - Interprets: “of the true class-\(c\) examples, how many did we recover?”

- **F1 (per class)**: harmonic mean of precision and recall:
  - \( \text{F1}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} \)

- **Macro Precision / Macro Recall / Macro F1**:
  - Compute each metric **per class**, then take the **unweighted mean** across classes.
  - Macro averaging is especially important when classes are imbalanced because it prevents dominant classes from overwhelming the score.

- **Cohen’s κ (kappa)**:
  - Agreement between predictions and labels **corrected for chance**.
  - \( \kappa = \frac{p_o - p_e}{1 - p_e} \), where \(p_o\) is observed agreement and \(p_e\) is expected agreement under chance.
  - Interprets: \(\kappa = 1\) perfect agreement, \(\kappa = 0\) chance-level agreement.

### What I will report

- **Clip-level classification (4-class and 5-class)**:
  - Top-1 / Top-2 Accuracy
  - Macro Precision / Macro Recall / Macro F1
  - Cohen’s κ
- **Video-level classification (when reported)**:
  - Same metrics, after aggregating clip predictions per video (video-level scores derived from clip scores for that video).
- **Window-level detection (includes background)**:
  - Top-1 / Top-2 Accuracy
  - Macro Precision / Macro Recall / Macro F1
  - Per-class F1 (to highlight rare-class behavior)
  - Cohen’s κ

## Temporal Action Localization

### Definitions

- **tIoU (temporal Intersection over Union):** Measures overlap between a predicted segment \([s_p, e_p]\) and a ground-truth segment \([s_g, e_g]\):
  - \( \text{tIoU} = \frac{\text{intersection length}}{\text{union length}} \)
  - Higher tIoU means better temporal boundary accuracy.

- **Precision / Recall (detection / localization):**
  - Precision asks: “of the predicted segments, how many match a ground-truth segment (at the chosen tIoU)?”
  - Recall asks: “of the ground-truth segments, how many were matched by at least one prediction (at the chosen tIoU)?”

- **Average Precision (AP):**
  - For a fixed tIoU threshold and a single class, AP summarizes the precision–recall tradeoff across score thresholds (area under the PR curve).

- **Mean Average Precision (mAP):**
  - Mean of AP across classes (for the multi-class task).
  - Reported at specific tIoU thresholds, e.g., **mAP@0.3**, **mAP@0.5**, **mAP@0.7**.

- **avg_mAP**:
  - A convenience summary defined as the mean of mAP values across a fixed set of tIoU thresholds (used to compare models with one number).

- **Recall@tIoU**:
  - Segment-level recall computed at a specific tIoU threshold (e.g., Recall@0.3 / Recall@0.5), typically reported for window-based TAL baselines.

- **Top-k recall (R@k, proposal-based)**:
  - For proposal-based TAL evaluation (e.g., ActionFormer/OpenTAD), **R@k** measures the fraction of ground-truth segments recovered when considering only the top \(k\) scored proposals (reported at multiple tIoU thresholds).

### What I will report

- **Window-based TAL (window scores → segments)**:
  - mAP@tIoU at standard thresholds used in the project (e.g., 0.3 / 0.5 / 0.7)
  - avg_mAP (average across those thresholds)
  - Recall@tIoU (when available) and avg_Recall
- **End-to-end segment-level TAL (transformer-based)**:
  - mAP@tIoU across the thresholds used by OpenTAD evaluation (e.g., 0.3–0.7)
  - avg_mAP
  - Top-k recall tables (R@k) at key tIoU thresholds
- **Binary TAL (RMM vs background)**:
  - Same reporting structure as above, but for a single positive class (broad localization)