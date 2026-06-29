# Video Example Analysis

This document analyzes model predictions on 10 specific videos selected to illustrate
various challenges: rocking toys, multi-label overlap, False_in_Context annotations,
and brief events (1-second windows).

---

## rocking_toy: rocking toy

**Filename**: `12927071_1683482288587213_48606419_n.mp4`
**GT Video ID**: `S7V5F9U7N6_unknown_33`
**Validation Folds**: [2]

### Ground Truth

- **Segments**: 4
- **Classes**: rocking
- **Total Duration**: 18.0s

| Segment ID | Class | Start | End | Duration |
|------------|-------|-------|-----|----------|
| own_33_0_0... | rocking | 0.0s | 5.0s | 5.0s |
| own_33_0_1... | rocking | 7.0s | 11.0s | 4.0s |
| own_33_0_2... | rocking | 16.0s | 20.0s | 4.0s |
| own_33_0_3... | rocking | 23.0s | 28.0s | 5.0s |

### Clip-level Predictions

**fold_2**:

- **fusion_3way_mlp**: 4/4 correct (100.0%)
- **vjepa_sam3**: 4/4 correct (100.0%)
- **stgcnpp_4stream**: 1/4 correct (25.0%)
- **posec3d_nonweighted**: 2/4 correct (50.0%)

### Window-level Predictions

**fold_2**:

- **vjepa_balanced**: 27 windows, 14 predicted as RMM
- **posec3d_focal**: 27 windows, 0 predicted as RMM

### ActionFormer Predictions

**fold_2**:

- **multiclass**: 200 predicted segments
  - [0.8-5.0s] rocking (score=0.228)
  - [0.4-5.3s] hands flapping (score=0.200)
  - [22.6-27.3s] rocking (score=0.184)
  - [25.2-27.8s] hands flapping (score=0.178)
  - [22.5-25.0s] hands flapping (score=0.141)
- **binary**: 60 predicted segments
  - [0.2-5.9s] rmm (score=0.395)
  - [22.0-27.5s] rmm (score=0.258)
  - [24.9-27.8s] rmm (score=0.168)
  - [5.9-10.8s] rmm (score=0.131)
  - [15.9-20.6s] rmm (score=0.125)

---

## spinning_long: long term spinning

**Filename**: `IMG_4359.MOV`
**GT Video ID**: `S2C4T1Y7V7_36_month_4`
**Validation Folds**: [0]

### Ground Truth

- **Segments**: 5
- **Classes**: jumping, spinning
- **Total Duration**: 25.0s

| Segment ID | Class | Start | End | Duration |
|------------|-------|-------|-----|----------|
| onth_4_0_0... | jumping | 11.0s | 13.0s | 2.0s |
| onth_4_0_1... | jumping | 29.0s | 32.0s | 3.0s |
| onth_4_0_2... | jumping | 36.0s | 38.0s | 2.0s |
| onth_4_1_0... | spinning | 1.0s | 8.0s | 7.0s |
| onth_4_1_1... | spinning | 17.0s | 28.0s | 11.0s |

### Clip-level Predictions

**fold_0**:

- **fusion_3way_mlp**: 4/5 correct (80.0%)
- **vjepa_sam3**: 4/5 correct (80.0%)
- **stgcnpp_4stream**: 5/5 correct (100.0%)
- **posec3d_nonweighted**: 4/5 correct (80.0%)

### Window-level Predictions

**fold_0**:

- **vjepa_balanced**: 42 windows, 9 predicted as RMM
- **stgcnpp_ce**: 42 windows, 7 predicted as RMM
- **posec3d_focal**: 42 windows, 9 predicted as RMM
- **fusion_vjepa_posec3d**: 42 windows, 37 predicted as RMM

### ActionFormer Predictions

**fold_0**:

- **multiclass**: 1314 predicted segments
  - [5.6-15.6s] jumping (score=0.397)
  - [1.6-18.3s] spinning (score=0.391)
  - [26.2-30.8s] jumping (score=0.389)
  - [0.1-4.5s] jumping (score=0.385)
  - [30.4-32.8s] hands flapping (score=0.369)
- **binary**: 466 predicted segments
  - [6.2-15.8s] rmm (score=0.539)
  - [5.4-8.3s] rmm (score=0.471)
  - [-0.0-7.0s] rmm (score=0.466)
  - [34.9-39.3s] rmm (score=0.442)
  - [30.5-32.7s] rmm (score=0.442)

---

## spinning_fan: child spinning fan (NOT IN GT)

**Filename**: `06-29-2021 (7).mov`
**GT Video ID**: ``
**Validation Folds**: []

### Ground Truth

*No RMM segments annotated for this video.*

### Clip-level Predictions

*No clip-level predictions available.*
### Window-level Predictions

*No window-level predictions available.*
### ActionFormer Predictions

*No ActionFormer predictions available.*
---

## waving_flag: waving flag (False_in_Context)

**Filename**: `0722221810.mp4`
**GT Video ID**: `G7P6U6U2J5_36_month_200`
**Validation Folds**: [0]

### Ground Truth

- **Segments**: 4
- **Classes**: jumping, one hand flap, hands flapping
- **Total Duration**: 33.0s

| Segment ID | Class | Start | End | Duration |
|------------|-------|-------|-----|----------|
| th_200_0_0... | jumping | 1.0s | 3.0s | 2.0s |
| th_200_1_0... | hands flapping | 37.0s | 43.0s | 6.0s |
| th_200_1_1... | hands flapping | 49.0s | 51.0s | 2.0s |
| th_200_2_0... | one hand flap | 10.0s | 33.0s | 23.0s |

### Clip-level Predictions

**fold_0**:

- **fusion_3way_mlp**: 3/4 correct (75.0%)
- **vjepa_sam3**: 3/4 correct (75.0%)
- **stgcnpp_4stream**: 3/4 correct (75.0%)
- **posec3d_nonweighted**: 4/4 correct (100.0%)

### Window-level Predictions

**fold_0**:

- **vjepa_balanced**: 57 windows, 28 predicted as RMM
- **stgcnpp_ce**: 55 windows, 20 predicted as RMM
- **posec3d_focal**: 55 windows, 21 predicted as RMM
- **fusion_vjepa_posec3d**: 57 windows, 43 predicted as RMM

### ActionFormer Predictions

**fold_0**:

- **multiclass**: 200 predicted segments
  - [49.3-52.2s] hands flapping (score=0.405)
  - [36.6-42.7s] hands flapping (score=0.389)
  - [9.0-11.5s] hands flapping (score=0.326)
  - [1.1-3.6s] jumping (score=0.282)
  - [17.2-21.7s] hands flapping (score=0.270)
- **binary**: 139 predicted segments
  - [49.1-52.3s] rmm (score=0.540)
  - [27.2-30.2s] rmm (score=0.472)
  - [36.7-42.8s] rmm (score=0.461)
  - [31.5-34.2s] rmm (score=0.421)
  - [0.8-3.8s] rmm (score=0.362)

---

## lamp_cord: trying to grab lamp cord (False_in_Context)

**Filename**: `3-1-19.MOV`
**GT Video ID**: `M2N1C8G1C8_36_month_225`
**Validation Folds**: [0]

### Ground Truth

- **Segments**: 1
- **Classes**: hands flapping
- **Total Duration**: 2.0s

| Segment ID | Class | Start | End | Duration |
|------------|-------|-------|-----|----------|
| th_225_0_0... | hands flapping | 0.0s | 2.0s | 2.0s |

### Clip-level Predictions

**fold_0**:

- **fusion_3way_mlp**: 1/1 correct (100.0%)
- **vjepa_sam3**: 1/1 correct (100.0%)
- **stgcnpp_4stream**: 1/1 correct (100.0%)
- **posec3d_nonweighted**: 1/1 correct (100.0%)

### Window-level Predictions

**fold_0**:

- **vjepa_balanced**: 16 windows, 7 predicted as RMM
- **stgcnpp_ce**: 16 windows, 1 predicted as RMM
- **posec3d_focal**: 16 windows, 0 predicted as RMM
- **fusion_vjepa_posec3d**: 16 windows, 8 predicted as RMM

### ActionFormer Predictions

**fold_0**:

- **multiclass**: 140 predicted segments
  - [15.6-17.2s] hands flapping (score=0.241)
  - [0.3-2.6s] hands flapping (score=0.219)
  - [12.2-15.1s] hands flapping (score=0.186)
  - [8.0-10.1s] hands flapping (score=0.120)
  - [15.6-17.2s] rocking (score=0.088)
- **binary**: 41 predicted segments
  - [15.7-17.2s] rmm (score=0.203)
  - [0.4-2.6s] rmm (score=0.182)
  - [11.5-15.7s] rmm (score=0.123)
  - [0.2-3.3s] rmm (score=0.083)
  - [7.5-9.6s] rmm (score=0.073)

---

## splashing_water: splashing water

**Filename**: `5.4.19 2nd clip.MP4`
**GT Video ID**: `A2M5H0V6E3_14_month_238`
**Validation Folds**: [2]

### Ground Truth

- **Segments**: 2
- **Classes**: one hand flap, hands flapping
- **Total Duration**: 5.0s

| Segment ID | Class | Start | End | Duration |
|------------|-------|-------|-----|----------|
| th_238_0_0... | one hand flap | 1.0s | 5.0s | 4.0s |
| th_238_1_0... | hands flapping | 1.0s | 2.0s | 1.0s |

### Clip-level Predictions

**fold_2**:

- **fusion_3way_mlp**: 2/2 correct (100.0%)
- **vjepa_sam3**: 2/2 correct (100.0%)
- **stgcnpp_4stream**: 2/2 correct (100.0%)
- **posec3d_nonweighted**: 2/2 correct (100.0%)

### Window-level Predictions

**fold_2**:

- **vjepa_balanced**: 10 windows, 5 predicted as RMM
- **posec3d_focal**: 10 windows, 0 predicted as RMM

### ActionFormer Predictions

**fold_2**:

- **multiclass**: 97 predicted segments
  - [0.6-5.4s] hands flapping (score=0.419)
  - [0.8-5.2s] rocking (score=0.183)
  - [2.9-5.5s] hands flapping (score=0.162)
  - [0.5-3.2s] hands flapping (score=0.145)
  - [0.5-2.7s] rocking (score=0.121)
- **binary**: 29 predicted segments
  - [0.6-6.0s] rmm (score=0.476)
  - [0.5-3.0s] rmm (score=0.168)
  - [2.9-5.5s] rmm (score=0.159)
  - [2.2-5.0s] rmm (score=0.070)
  - [7.9-9.9s] rmm (score=0.069)

---

## trampoline: trampoline (1-sec windows)

**Filename**: `04 28 2017.mov`
**GT Video ID**: `L4M1H7J7G3_14_month_313`
**Validation Folds**: [0]

### Ground Truth

- **Segments**: 3
- **Classes**: jumping, hands flapping
- **Total Duration**: 3.0s

| Segment ID | Class | Start | End | Duration |
|------------|-------|-------|-----|----------|
| th_313_0_0... | jumping | 11.0s | 12.0s | 1.0s |
| th_313_0_1... | jumping | 42.0s | 43.0s | 1.0s |
| th_313_1_0... | hands flapping | 26.0s | 27.0s | 1.0s |

### Clip-level Predictions

**fold_0**:

- **fusion_3way_mlp**: 3/3 correct (100.0%)
- **vjepa_sam3**: 3/3 correct (100.0%)
- **stgcnpp_4stream**: 3/3 correct (100.0%)
- **posec3d_nonweighted**: 2/3 correct (66.7%)

### Window-level Predictions

**fold_0**:

- **vjepa_balanced**: 44 windows, 10 predicted as RMM
- **stgcnpp_ce**: 38 windows, 4 predicted as RMM
- **posec3d_focal**: 38 windows, 3 predicted as RMM
- **fusion_vjepa_posec3d**: 44 windows, 33 predicted as RMM

### ActionFormer Predictions

**fold_0**:

- **multiclass**: 200 predicted segments
  - [41.2-43.7s] hands flapping (score=0.375)
  - [25.2-27.2s] hands flapping (score=0.326)
  - [12.2-14.4s] hands flapping (score=0.303)
  - [8.7-13.6s] hands flapping (score=0.183)
  - [40.7-44.8s] hands flapping (score=0.176)
- **binary**: 109 predicted segments
  - [40.8-44.8s] rmm (score=0.374)
  - [0.3-3.1s] rmm (score=0.330)
  - [25.1-27.3s] rmm (score=0.320)
  - [12.3-14.5s] rmm (score=0.316)
  - [6.8-14.1s] rmm (score=0.283)

---

## game_jumping: game with jumping (False_in_Context)

**Filename**: `08.09.2022.MP4`
**GT Video ID**: `W2E0Z9B7Z8_36_month_331`
**Validation Folds**: [1]

### Ground Truth

- **Segments**: 1
- **Classes**: jumping
- **Total Duration**: 1.0s

| Segment ID | Class | Start | End | Duration |
|------------|-------|-------|-----|----------|
| th_331_0_0... | jumping | 14.0s | 15.0s | 1.0s |

### Clip-level Predictions

**fold_1**:

- **fusion_3way_mlp**: 1/1 correct (100.0%)
- **vjepa_sam3**: 1/1 correct (100.0%)
- **stgcnpp_4stream**: 1/1 correct (100.0%)
- **posec3d_nonweighted**: 1/1 correct (100.0%)

### Window-level Predictions

**fold_1**:

- **vjepa_balanced**: 61 windows, 34 predicted as RMM
- **stgcnpp_ce**: 32 windows, 7 predicted as RMM
- **posec3d_focal**: 32 windows, 0 predicted as RMM
- **fusion_vjepa_posec3d**: 61 windows, 49 predicted as RMM

### ActionFormer Predictions

**fold_1**:

- **multiclass**: 200 predicted segments
  - [18.2-23.2s] hands flapping (score=0.268)
  - [12.3-21.7s] jumping (score=0.214)
  - [30.7-33.1s] hands flapping (score=0.205)
  - [28.3-30.6s] hands flapping (score=0.185)
  - [10.9-13.2s] hands flapping (score=0.182)
- **binary**: 149 predicted segments
  - [9.5-19.2s] rmm (score=0.224)
  - [0.5-2.5s] rmm (score=0.214)
  - [19.1-24.1s] rmm (score=0.188)
  - [16.3-18.8s] rmm (score=0.168)
  - [27.4-32.5s] rmm (score=0.166)

---

## jumping_flapping_overlap: jumping and arm flapping overlap

**Filename**: `IMG_4620.MOV`
**GT Video ID**: `S2C4T1Y7V7_36_month_8`
**Validation Folds**: [0]

### Ground Truth

- **Segments**: 3
- **Classes**: jumping, one hand flap
- **Total Duration**: 11.0s

| Segment ID | Class | Start | End | Duration |
|------------|-------|-------|-----|----------|
| onth_8_0_0... | jumping | 0.0s | 1.0s | 1.0s |
| onth_8_0_1... | jumping | 6.0s | 14.0s | 8.0s |
| onth_8_1_0... | one hand flap | 12.0s | 14.0s | 2.0s |

### Clip-level Predictions

**fold_0**:

- **fusion_3way_mlp**: 1/3 correct (33.3%)
- **vjepa_sam3**: 2/3 correct (66.7%)
- **stgcnpp_4stream**: 2/3 correct (66.7%)
- **posec3d_nonweighted**: 2/3 correct (66.7%)

### Window-level Predictions

**fold_0**:

- **vjepa_balanced**: 17 windows, 12 predicted as RMM
- **stgcnpp_ce**: 17 windows, 5 predicted as RMM
- **posec3d_focal**: 17 windows, 4 predicted as RMM
- **fusion_vjepa_posec3d**: 17 windows, 12 predicted as RMM

### ActionFormer Predictions

**fold_0**:

- **multiclass**: 1314 predicted segments
  - [5.6-15.6s] jumping (score=0.397)
  - [1.6-18.3s] spinning (score=0.391)
  - [26.2-30.8s] jumping (score=0.389)
  - [0.1-4.5s] jumping (score=0.385)
  - [30.4-32.8s] hands flapping (score=0.369)
- **binary**: 466 predicted segments
  - [6.2-15.8s] rmm (score=0.539)
  - [5.4-8.3s] rmm (score=0.471)
  - [-0.0-7.0s] rmm (score=0.466)
  - [34.9-39.3s] rmm (score=0.442)
  - [30.5-32.7s] rmm (score=0.442)

---

## subtle_rocking: subtle rocking

**Filename**: `20181015_094544.mp4`
**GT Video ID**: `G4J8F6F4X0_14_month_103`
**Validation Folds**: [2]

### Ground Truth

- **Segments**: 1
- **Classes**: rocking
- **Total Duration**: 3.0s

| Segment ID | Class | Start | End | Duration |
|------------|-------|-------|-----|----------|
| th_103_0_0... | rocking | 8.0s | 11.0s | 3.0s |

### Clip-level Predictions

**fold_2**:

- **fusion_3way_mlp**: 1/1 correct (100.0%)
- **vjepa_sam3**: 1/1 correct (100.0%)
- **stgcnpp_4stream**: 0/1 correct (0.0%)
- **posec3d_nonweighted**: 0/1 correct (0.0%)

### Window-level Predictions

**fold_2**:

- **vjepa_balanced**: 26 windows, 11 predicted as RMM
- **posec3d_focal**: 26 windows, 1 predicted as RMM

### ActionFormer Predictions

**fold_2**:

- **multiclass**: 200 predicted segments
  - [7.1-9.8s] hands flapping (score=0.307)
  - [0.4-3.4s] hands flapping (score=0.305)
  - [15.0-19.6s] hands flapping (score=0.177)
  - [18.4-20.6s] hands flapping (score=0.162)
  - [3.5-6.1s] hands flapping (score=0.149)
- **binary**: 62 predicted segments
  - [0.5-3.8s] rmm (score=0.303)
  - [7.2-10.3s] rmm (score=0.293)
  - [1.6-10.8s] rmm (score=0.214)
  - [18.3-20.6s] rmm (score=0.182)
  - [15.4-20.4s] rmm (score=0.121)

---
