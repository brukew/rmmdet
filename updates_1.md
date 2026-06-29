## **RMM Classification**

### Setup

- **Tracking / COI**: SAM3 for tracking, COI manually labeled per video
- **Pose Estimation**: HRNet fine-tuned on COCO-WholeBody
- **Evaluation**: 3-fold CV with **Leave-Child-Timepoint-Out**
- **Classes**
    - Hands Flapping
    - One Hand Flap
    - Spinning
    - Jumping
    - Rocking
- **Class Distinctions**:
    - **4-class**: hands flapping + one-hand flap merged
    - **5-class**: hands flapping and one-hand flap treated separately

### **Methods**

- **Qwen2.5-VL** (VLM baseline): 16 consecutive frames × 4 evenly sampled windows, majority vote
- **V-JEPA 2** (Finetune) ****: cropped RGB COI, 64 evenly sampled frames
- **PoseC3D** (Finetune, Pretrained on Kinetics 400): keypoint heatmaps weighted by confidence, 48 evenly sampled frames

### Results

- **4-class**
    - **Qwen**
        - Top-1 Acc: 0.5682 ± 0.0335
        - Top-2 Acc: 0.7887 ± 0.0225
        - Macro-F1: 0.2777 ± 0.0065
    - **V-JEPA 2**
        - Top-1 Acc: **0.7596 ± 0.0326**
        - Top-2 Acc: **0.9372 ± 0.0132**
        - Macro-F1: **0.7514 ± 0.0187**
    - **PoseC3D**
        - Top-1 Acc: 0.7460 ± 0.0299
        - Top-2 Acc: 0.9125 ± 0.0039
        - Macro-F1: 0.7040 ± 0.0725
    
- **5-class**
    - **Qwen**
        - Top-1 Acc: 0.3769 ± 0.0599
        - Top-2 Acc: 0.6355 ± 0.0217
        - Macro-F1: 0.1226 ± 0.0192
    - **V-JEPA 2**
        - Top-1 Acc: 0.6079 ± 0.0718
        - Top-2 Acc: 0.8221 ± 0.0307
        - Macro-F1: 0.5959 ± 0.0721
    - **PoseC3D**
        - Top-1 Acc: **0.6675 ± 0.0156**
        - Top-2 Acc: **0.8459 ± 0.0161**
        - Macro-F1: **0.6514 ± 0.0282**

### Notes / Next Steps

- `rocking` presents a major challenge to both models with an average of 50% accuracy
- Next Steps
    - Finetune CTR-GCN
    - Dig deeper into failing clips for more info
    - Fusion model (RGB + Pose-based)

## Location Classification

### Setup

- Single Train/Val/Test split
- Classes
    - public / private, inside / outside

### Methods

- **DINOv3** (Finetune): RGB, 16 evenly sampled frames
- **V-JEPA 2** (Finetune): RGB, 16 consecutive frames × 2 windows
- **Qwen2.5-VL**: RGB, 16 consecutive frames × 4 windows

### **Results**

- **DINOv3**
    - **Top-1 Acc: 0.949**
    - **Top-2 Acc: 0.990**
    - **Macro-F1: 0.863**
- VJEPA-2
    - Top-1 Acc: 0.883
    - Top-2 Acc: 0.936
    - Macro-F1: 0.658
- Qwen2.5-VL
    - Top-1 Acc: 0.836