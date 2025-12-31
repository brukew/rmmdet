# RMM Classification Failure Review

**Reviewer:** _______________  
**Date:** _______________

This document lists videos and clips that all models failed to classify correctly. Use this to flag annotation issues, ambiguous cases, or data quality problems.

---

## Flag Legend

| Flag | Meaning |
|------|---------|
| ✅ | Annotation correct - models just failed |
| ⚠️ | Annotation ambiguous - needs review |
| ❌ | Annotation incorrect - should be relabeled |
| 🔄 | Mixed behaviors - multiple RMMs present |
| 📹 | Video quality issue |
| 👥 | Multi-person confusion |
| ❓ | Unclear - need second opinion |

---

## Section 1: Multi-Failure Videos (2+ clips failed)

These 10 videos have multiple clips where both V-JEPA and PoseC3D failed.

### 1. C5U1P3X9F0_14_month_310 (6 failed clips)
- **Label:** rocking
- **Video:** `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized/O.B._Home_Videos_AMES_C5U1P3X9F0/12-16 month videos/09-22-2019.mp4`
- [x] Reviewed
- **Flag:** ⚠️
- **Notes:** 
 Rocking/twisting mis-identified as jands flapping. understandable since hands are involved in the twisting and are moving around but this is moreso twisting. wonder how we can remedy this. we should def flag clips that are labeled as twisting and do firther analysis on those videos because they can be mistaken for hands flapping.

 pose estimation is perfect here
---

### 2. L0C6T6H2C6_36_month_102 (5 failed clips)
- **Label:** jumping
- **Video:** `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized/D.R._Home_Videos_AMES_L0C6T6H2C6/34-38 month videos/03-08-2021.mp4`
- [x] Reviewed
- **Flag:** 👥
- **Notes:** 
  Pose estimation completely fails here. Need to redo. although sam crops are correct, pose kpts are made for other individuals. 
---

### 3. J3J0V4T8C3_36_month_58 (4 failed clips)
- **Label:** hands flapping
- **Video:** `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized/C.B._Home_Videos_AMES_J3J0V4T8C3/34-38 month videos/06-18-2021 (2).mp4`
- [X] Reviewed
- **Flag:** ⚠️
- **Notes:**  
  Both jumping and hands flapping. Multi class clips.
  
---

### 4. C1G1X2G4C6_36_month_232 (2 failed clips)
- **Label:** jumping
- **Video:** `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized/J.V._Home_Videos_AMES_C1G1X2G4C6/34-38 month videos/IMG_7585.mp4`
- [ ] Reviewed
- **Flag:** ___
- **Notes:** 
  
---

### 5. C5L2D6N1W6_36_month_145 (2 failed clips)
- **Label:** jumping
- **Video:** `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized/G.L._Home_Videos_AMES_C5L2D6N1W6/34-38 month videos/3-13-20.mp4`
- [ ] Reviewed
- **Flag:** ___
- **Notes:** 
  
---

### 6. J3J0V4T8C3_14_month_50 (2 failed clips)
- **Label:** rocking
- **Video:** `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized/C.B._Home_Videos_AMES_J3J0V4T8C3/12-16 month videos/05-28-2019 (1).mp4`
- [ ] Reviewed
- **Flag:** ___
- **Notes:** 
  
---

### 7. L6F2P7K8P1_36_month_121 (2 failed clips)
- **Label:** jumping
- **Video:** `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized/E.F._Home_Videos_AMES_L6F2P7K8P1/34-38 month videos/10.17.2019.mp4`
- [ ] Reviewed
- **Flag:** ___
- **Notes:** 
  
---

### 8. S9B3J7I4R1_36_month_271 (2 failed clips)
- **Label:** jumping
- **Video:** `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized/M.J._Home_Videos_AMES_S9B3J7I4R1/34-38 month videos/20191019_154219.mp4`
- [ ] Reviewed
- **Flag:** ___
- **Notes:** 
  
---

### 9. T4R5V7C3L6_unknown_258 (2 failed clips)
- **Label:** hands flapping
- **Video:** `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized/L.P._Home_Videos_AMES_T4R5V7C3L6/34-38 month videos/344619313_6065239563588538_5835258672173916095_n.mp4`
- [ ] Reviewed
- **Flag:** ___
- **Notes:** 
  
---

### 10. W9K7Y8Y0V6_36_month_113 (2 failed clips)
- **Label:** spinning
- **Video:** `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized/D.W._Home_Videos_AMES_W9K7Y8Y0V6/34-38 month videos/35-month-daniel-hike.mp4`
- [ ] Reviewed
- **Flag:** ___
- **Notes:** 
  
---

## Section 2: All Individual Clips (48 clips where all 3 models failed)

### By True Label

#### Rocking (22 clips)
| Clip ID | V-JEPA | PoseC3D | Qwen | Quality | Flag | Notes |
|---------|--------|---------|------|---------|------|-------|
| S2C4T1Y7V7_36_month_6_1_0 | spinning | hands flapping | hands flapping | low | | |
| G4J8F6F4X0_36_month_109_0_0 | hands flapping | jumping | hands flapping | high | | |
| L4K0P3T4Q8_14_month_183_0_0 | hands flapping | jumping | hands flapping | high | | |
| J4J6X7C4D9_36_month_243_0_0 | jumping | spinning | hands flapping | medium | | |
| Z7I5F2E1L4_36_month_247_2_0 | jumping | jumping | hands flapping | high | | |
| L4M8P2T7S0_36_month_282_0_0 | spinning | spinning | jumping | high | | |
| J3J0V4T8C3_14_month_50_0_1 | jumping | hands flapping | hands flapping | low | | |
| J3J0V4T8C3_14_month_50_0_2 | jumping | hands flapping | hands flapping | low | | |
| N4U1A5H1W0_36_month_95_0_0 | hands flapping | hands flapping | hands flapping | low | | |
| X2V2P0U5X4_36_month_136_0_0 | hands flapping | jumping | hands flapping | low | | |
| S9B3J7I4R1_36_month_268_1_0 | hands flapping | hands flapping | hands flapping | low | | |
| L7Z6Q2U6Y6_14_month_356_0_2 | spinning | hands flapping | hands flapping | high | | |
| H7B1M3P5D9_36_month_74_0_3 | jumping | jumping | hands flapping | high | | |
| N4U1A5H1W0_14_month_94_0_0 | jumping | hands flapping | jumping | high | | |
| G4J8F6F4X0_14_month_104_0_0 | hands flapping | hands flapping | hands flapping | high | | |
| S7Y0I2P0S8_14_month_202_1_0 | jumping | jumping | jumping | high | | |
| C5U1P3X9F0_14_month_310_1_0 | hands flapping | hands flapping | hands flapping | high | | |
| C5U1P3X9F0_14_month_310_1_2 | hands flapping | hands flapping | hands flapping | high | | |
| C5U1P3X9F0_14_month_310_2_0 | hands flapping | hands flapping | hands flapping | low | | |
| C5U1P3X9F0_14_month_310_2_1 | hands flapping | hands flapping | hands flapping | low | | |
| C5U1P3X9F0_14_month_310_2_2 | hands flapping | hands flapping | hands flapping | high | | |
| C5U1P3X9F0_14_month_310_2_3 | hands flapping | hands flapping | hands flapping | high | | |

#### Jumping (17 clips)
| Clip ID | V-JEPA | PoseC3D | Qwen | Quality | Flag | Notes |
|---------|--------|---------|------|---------|------|-------|
| G4J8F6F4X0_36_month_111_0_0 | hands flapping | hands flapping | hands flapping | high | | |
| H9T5Y8D3A1_36_month_223_0_0 | hands flapping | hands flapping | hands flapping | high | | |
| T4R5V7C3L6_unknown_258_1_0 | hands flapping | hands flapping | hands flapping | high | | |
| A1H3H9Y3T1_36_month_24_0_0 | hands flapping | rocking | hands flapping | low | | |
| L0C6T6H2C6_36_month_102_0_0 | hands flapping | hands flapping | hands flapping | low | | |
| L0C6T6H2C6_36_month_102_0_1 | hands flapping | hands flapping | hands flapping | low | | |
| L0C6T6H2C6_36_month_102_0_2 | hands flapping | rocking | hands flapping | low | | |
| C5L2D6N1W6_36_month_145_0_0 | hands flapping | rocking | hands flapping | high | | |
| C5L2D6N1W6_36_month_145_0_4 | hands flapping | hands flapping | hands flapping | high | | |
| R6R8Q6A7K0_36_month_150_1_0 | hands flapping | hands flapping | hands flapping | high | | |
| C1G1X2G4C6_36_month_232_1_3 | hands flapping | hands flapping | hands flapping | high | | |
| C1G1X2G4C6_36_month_232_1_5 | hands flapping | hands flapping | hands flapping | high | | |
| L6F2P7K8P1_36_month_121_0_1 | hands flapping | hands flapping | hands flapping | high | | |
| L6F2P7K8P1_36_month_121_0_2 | hands flapping | hands flapping | hands flapping | high | | |
| E7Z4G3O5K2_unknown_140_0_0 | hands flapping | hands flapping | hands flapping | high | | |
| E7G3N4A1K5_36_month_207_0_2 | hands flapping | hands flapping | hands flapping | medium | | |
| W3O7N1N8U2_36_month_372_0_0 | hands flapping | hands flapping | hands flapping | high | | |

#### Spinning (6 clips)
| Clip ID | V-JEPA | PoseC3D | Qwen | Quality | Flag | Notes |
|---------|--------|---------|------|---------|------|-------|
| S2C4T1Y7V7_36_month_4_1_0 | hands flapping | hands flapping | hands flapping | low | | |
| N5I9D3R2M1_36_month_284_0_0 | rocking | rocking | hands flapping | high | | |
| L0C6T6H2C6_36_month_102_2_0 | hands flapping | hands flapping | hands flapping | low | | |
| L0C6T6H2C6_36_month_102_2_1 | hands flapping | rocking | hands flapping | low | | |
| W9K7Y8Y0V6_36_month_113_0_0 | hands flapping | hands flapping | hands flapping | high | | |
| W9K7Y8Y0V6_36_month_113_0_1 | hands flapping | hands flapping | hands flapping | high | | |

#### Hands Flapping (3 clips)
| Clip ID | V-JEPA | PoseC3D | Qwen | Quality | Flag | Notes |
|---------|--------|---------|------|---------|------|-------|
| Q9W9P7Z3J7_36_month_291_1_0 | jumping | jumping | jumping | high | | |
| J3J0V4T8C3_36_month_53_1_0 | jumping | jumping | jumping | low | | |
| D8G5K7I0D6_14_month_220_0_0 | rocking | jumping | jumping | high | | |

---

## Summary Statistics (fill in after review)

| Category | Count | Notes |
|----------|-------|-------|
| Total clips reviewed | /48 | |
| Annotation correct (✅) | | |
| Annotation ambiguous (⚠️) | | |
| Annotation incorrect (❌) | | |
| Mixed behaviors (🔄) | | |
| Video quality issues (📹) | | |
| Multi-person confusion (👥) | | |
| Unclear (❓) | | |

---

## Action Items

1. [ ] 
2. [ ] 
3. [ ] 

---

## Notes


