# Table 5.6: Window-level Detection Metrics Comparison

Comparison of Top-1 Macro F1, Top-1 Accuracy, and Top-2 Accuracy for best window-level detection models.

| Model Family | Model | Top-1 Macro F1 | Top-1 Acc | Top-2 Acc | Acc Gain |
|--------------|-------|----------------|-----------|-----------|----------|
| RGB | V-JEPA Balanced | 23.7% ± 4.1% | 80.9% ± 1.7% | 95.1% ± 0.9% | +14.1% |
| Pose | PoseC3D (CE) | 24.0% ± 4.1% | 85.9% ± 0.6% | 96.0% ± 0.6% | +10.1% |
| Fusion | V-JEPA + PoseC3D (MLP) | 18.1% ± 1.4% | 52.3% ± 5.1% | 87.5% ± 3.4% | +35.2% |

**Notes:**
- Top-1 Macro F1: Standard macro-averaged F1 over 4 RMM classes
- Top-1/Top-2 Acc: Percentage of windows where GT class is in top-1/top-2 predictions
- Acc Gain: Improvement from considering top-2 predictions instead of top-1
- Values are mean ± std across CV folds