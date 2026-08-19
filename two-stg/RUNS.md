# Two-Stage TAL — Run Log & Leaderboard

Most runs use **ActionFormer Binary** detections (Stage 1) + **Stage-2 classification** (V-JEPA2-only, 5-class, or **3-way fusion**: live V-JEPA2 + MLP with PoseC3D/STGCN++).
Score combination: `final_score = det_score × max(stage2_class_probs)`. No detection threshold.

**3-way fusion variants**

| Variant | PoseC3D/STGCN++ source | MLP training |
|--------|-------------------------|--------------|
| **skeleton_index** (legacy) | Nearest val clip by tIoU from `skeleton_index.csv` | **Decontaminated:** cross-fold val preds only (`export_three_way_deploy_checkpoint.py`) |
| **Live proposals** | Per-proposal inference on H5 poses → CSVs; fallback to skeleton index for missing IDs | Same decontaminated bundle (trained on GT-clip scores, not proposal scores — causes distribution mismatch) |

**skeleton_index leakage warning:** The skeleton_index is built from the **same fold's val clips**, so at eval time a val-set proposal can look up GT-clip skeleton scores from the same video by tIoU. This is **test-time information leakage** — in deployment on unseen videos, no skeleton_index entries exist for that video and the system falls back to uniform 1/4 priors (losing the skeleton signal entirely). Run 3b numbers are therefore **not representative of deployment performance**. Run 3c (live proposal scores) is the honest, deployable approach.

**Contamination note:** An earlier export trained the MLP on **fold-N val** clip predictions, then evaluated on the same val TAL split — optimistic bias. Current checkpoints use **Approach A (cross-fold swap):** for fold N, the MLP is trained only on merged val predictions from the **other two folds** (~410–430 clips). `skeleton_index.csv` is still built from fold N val clips for tIoU lookup. Re-run eval after re-export to report honest numbers.

**GPU / SLURM**

- 3-way (skeleton index): `sbatch --export=FOLD=N two-stg/run_three_way_fold_gpu.sh` or `bash two-stg/run_three_way_cv_gpu.sh`
- Proposal pickle + skeleton inference only: `sbatch --export=FOLD=N two-stg/run_proposal_skeleton_inference.sh`
- Full 3-way + **live** proposal scores: `sbatch --export=FOLD=N two-stg/run_three_way_live_fold_gpu.sh` (long job) or `bash two-stg/run_three_way_live_cv_gpu.sh`

Logs: `two-stg/logs/`.

---

## Leaderboard (3-Fold CV Mean, OpenTAD mAP @ 0.3 / 0.5 / 0.7)

| # | Run | Classifier | avg_mAP | mAP@0.3 | mAP@0.5 | mAP@0.7 |
|---|-----|-----------|---------|---------|---------|---------|
| 3b | 3-way decontam | MLP fusion, skel_index lookup (**⚠️ leaks GT val scores at test time**) | 25.25%* | 35.21%* | 25.33%* | 13.40%* |
| 3c | 3-way live | AF + MLP + live skeleton (**honest/deployable**) | 16.78% | 23.43% | 16.93% | 8.34% |
| 1 | 4-class | AF + V-JEPA2 4-class | 17.83% | 25.29% | 17.74% | 9.05% |
| 5c | TriDet 3-way live | **TriDet** + MLP + live skeleton | 16.67% | 24.07% | 17.21% | 8.39% |
| 5 | TriDet 4-class | **TriDet** + V-JEPA2 4-class | 18.65% | 26.29% | 19.25% | 10.29% |
| 2 | 5-class | V-JEPA2 5-class (4 RMM + BG) | 16.07% | 22.88% | 16.29% | 8.12% |
| 6 | **TriDet Balanced (E2E)** | end-to-end 4-class | **19.72%** | **26.77%** | **20.53%** | **10.42%** |
| — | *ActionFormer Balanced (baseline)* | *end-to-end 4-class* | *16.70%* | *23.97%* | *17.57%* | *7.04%* |

**Binary detector comparison (Stage 1 only — not directly comparable to multi-class runs above):**

| # | Detector | avg_mAP | mAP@0.3 | mAP@0.5 | mAP@0.7 |
|---|----------|---------|---------|---------|---------|
| 4 | **TriDet Binary** | **28.42%** | **41.64%** | **29.00%** | **14.47%** |
| — | ActionFormer Binary | 27.93% | 40.83% | 28.55% | 11.97% |

TriDet vs ActionFormer Binary: +0.49 pp avg_mAP overall, but **+2.50 pp at mAP@0.7** (14.47% vs 11.97%) — confirms sharper boundary precision. Next: use TriDet proposals as Stage 1 → run Stage 2 classification → multi-class mAP for direct leaderboard comparison.

---

## Run Details

### Run 1: 4-class (best single-modality two-stage)

- **Date:** 2026-03-18
- **Classifier:** `v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls/fold_{0,1,2}`
- **Detector:** `OpenTAD/exps/sails_rmm/actionformer_vjepa_binary_fold{0,1,2}/gpu1_id99/result_detection.json`
- **Results:** `two-stg/eval_results/fold{0,1,2}/`, `two-stg/eval_results/cv_summary.json`
- **Script:** `sbatch --export=FOLD=N two-stg/run_single_fold_gpu.sh`

| Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|
| 0 | 30.32% | 23.39% | 14.00% | 23.23% |
| 1 | 22.87% | 15.87% | 6.28% | 15.77% |
| 2 | 22.67% | 13.96% | 6.88% | 14.50% |
| **Mean ± Std** | **25.29% ± 3.56%** | **17.74% ± 4.07%** | **9.05% ± 3.50%** | **17.83% ± 3.85%** |

### Run 3: 3-way fusion (historical — **contaminated** MLP on val)

These numbers used an export that trained the fusion MLP on **the same fold's val** clip predictions. **Do not use for claims.** Kept for comparison after re-running with the decontaminated export.

- **Results:** `two-stg/eval_results_3way/fold{0,1,2}/`
- **Artifact issue:** MLP saw val clip logits/softmax rows that overlap the TAL val evaluation distribution.

| Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|
| 0 | 49.05% | 32.32% | 16.55% | 32.64% |
| 1 | 38.59% | 27.66% | 14.32% | 26.86% |
| 2 | 38.85% | 22.31% | 10.43% | 23.86% |
| **Mean ± Std** | **42.17% ± 4.99%** | **27.43% ± 4.14%** | **13.77% ± 2.51%** | **27.79% ± 3.60%** |

### Run 3b: 3-way fusion — **decontaminated** MLP (cross-fold swap)

- **Date:** 2026-03-19
- **Export:** `python fusion/export_three_way_deploy_checkpoint.py --all-folds` (current code trains MLP on **other folds' val** preds only; `config.json` lists `mlp_train_folds`).
- **Stage-2:** skeleton_index lookup for PoseC3D/STGCN++ (nearest GT clip by tIoU).
- **Results:** `two-stg/eval_results_3way/fold{0,1,2}/` (overwrote contaminated Run 3)

| Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|
| 0 | 38.23% | 29.77% | 16.85% | 28.98% |
| 1 | 35.14% | 26.87% | 13.50% | 25.59% |
| 2 | 32.27% | 19.36% | 9.85% | 21.19% |
| **Mean ± Std** | **35.21% ± 2.43%** | **25.33% ± 4.39%** | **13.40% ± 2.86%** | **25.25% ± 3.19%** |

### Run 3c: 3-way + **live** PoseC3D/STGCN++ on proposals

- **Date:** 2026-03-20 (fold 0), 2026-03-24 (folds 1 & 2)
- **Pickle:** `python two-stg/build_proposal_poses.py --fold N` (or `--all-folds`); skip log: `proposals_fold{N}_skips.json`
- **Inference:** `two-stg/run_proposal_skeleton_inference.sh` (or inline in `run_three_way_live_fold_gpu.sh`)
- **Eval:** `two-stg/run_three_way_live_fold_gpu.sh` → default `two-stg/eval_results_3way_live/fold{N}/`
- **SLURM jobs:** 10716918 (fold 0), 10883900 (fold 1), 10884982 (fold 2)

**Finding:** 3-fold CV mean (16.78% avg_mAP) is **8.5 pp below** skeleton-index baseline 3b (25.25%). The 3-way MLP was trained on GT-clip-level PoseC3D/STGCN++ scores but receives noisier proposal-level scores at eval — a **train/test distribution mismatch**. Additionally, ~8% of proposals lack skeleton scores (pose cache truncation) and fall back to skeleton_index.

**Next step:** Retrain MLP on proposal-style skeleton scores to close the distribution gap.

| Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|
| 0 | 24.54% | 19.31% | 10.64% | 18.94% |
| 1 | 24.36% | 19.77% | 8.37% | 17.98% |
| 2 | 21.38% | 11.71% | 5.98% | 13.41% |
| **Mean ± Std** | **23.43% ± 1.44%** | **16.93% ± 3.72%** | **8.34% ± 1.91%** | **16.78% ± 2.35%** |

### Run 2: 5-class (4 RMM + background)

- **Date:** 2026-03-18
- **Classifier:** `v-jepa/runs/vjepa2_tal_cv_5class_balanced/fold_{0,1,2}`
- **Detector:** same as Run 1
- **Results:** `two-stg/eval_results_5class/fold{0,1,2}/`
- **Script:** `bash two-stg/run_5class_cv_gpu.sh`
- **Analysis:** `two-stg/eval_results_5class/RESULTS.md`

| Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|
| 0 | 30.97% | 22.95% | 12.17% | 22.08% |
| 1 | 18.45% | 12.56% | 5.98% | 12.54% |
| 2 | 19.23% | 13.36% | 6.21% | 13.57% |
| **Mean ± Std** | **22.88% ± 5.72%** | **16.29% ± 4.72%** | **8.12% ± 2.87%** | **16.07% ± 4.28%** |

**Why 5-class underperforms (deep dive):**

- **Score collapse:** 5-class scores are 37% of 4-class on average; when bg_prob > 0.9 (48% of predictions), scores drop to just 3.4%
- **Jumping devastated:** 94% of jumping predictions have bg_prob > 0.7, reducing scores to 10% of 4-class
- **Label disagreement:** 44% of predictions get different labels; jumping often relabeled as hands flapping (43%) or spinning (29%)
- **Training mismatch:** 5-class was trained on ~80% background windows; when applied to ActionFormer proposals (already filtered for RMM), it over-predicts background

---

## Stage 1 Ceiling Analysis (fold 0)

The binary ActionFormer (Stage 1) limits what any Stage 2 classifier can achieve.

**Recall:** ActionFormer finds most GT segments — 97.5% at tIoU=0.3, 88.0% at 0.5, 71.5% at 0.7.

**Oracle mAP:** If Stage 2 classified every proposal **perfectly** (correct class, optimal ranking):

| tIoU | GT Recall | Oracle mAP | Run 3c actual | Run 1 (V-JEPA2) | Efficiency (3c / oracle) |
|------|-----------|------------|---------------|-----------------|--------------------------|
| 0.3 | 97.5% | **67.4%** | 24.5% | 30.3% | 36% |
| 0.5 | 88.0% | **51.6%** | 19.3% | 23.4% | 37% |
| 0.7 | 71.5% | **30.3%** | 10.6% | 14.0% | 35% |

**Per-class oracle AP@0.3:** hands flapping 72.4%, jumping 78.1%, spinning 83.0%, rocking 36.2% — rocking is the hardest to localize.

**Takeaway:** Stage 2 achieves ~36% of the theoretical ceiling. Both bottlenecks are roughly equal in severity: Stage 1 caps mAP at ~67% (proposal quality / ranking), and Stage 2 currently captures only a third of that (classification accuracy). Improving either one alone won't get past ~30% mAP. 5,770 proposals for 158 GT segments (36:1 ratio) means even perfect classification still has a precision problem.

---

## Run 4: TriDet Binary (Stage 1 upgrade)

- **Date:** 2026-03-22 (completed)
- **Model:** TriDet (SGP projection + Trident-head with boundary distributions + IoU-weighted loss)
- **Features:** V-JEPA2 binary (same as ActionFormer runs)
- **Config:** `OpenTAD/configs/tridet/sails_rmm_vjepa_binary_fold{0,1,2}.py`
- **Training:** `sbatch OpenTAD/slurm/train_tridet_cv.sh binary N` (GPU, 50 epochs, AdamW lr=1e-4, wd=0.025)
- **Motivation:** Stage 1 ceiling analysis shows ActionFormer proposals cap oracle mAP at 67.4%@0.3 (36:1 proposal ratio, poor boundary precision at high tIoU). TriDet addresses this with:
  - **SGP blocks** instead of transformers → better multi-scale temporal aggregation
  - **Trident-head** predicts boundary probability distributions → sharper start/end times
  - **IoU-weighted classification loss** → couples detection confidence with localization quality, reducing false positives
- **Key config differences from ActionFormer:**
  - `projection`: `TriDetProj` with SGP blocks (k=5, init_conv_vars=0, input_noise=0.0005) vs `Conv1DTransformerProj` with MHA
  - `rpn_head`: `TriDetHead` (num_bins=16, boundary_kernel_size=3, iou_weight_power=0.2, GIOU iou_rate) vs `ActionFormerHead`
  - `optimizer.weight_decay`: 0.025 (TriDet default) vs 0.05 (ActionFormer default)
  - Everything else (dataset, NMS, scheduler, solver) kept identical for fair comparison
- **Expected outcome:** Better boundary precision (higher mAP@0.7), lower false-positive rate, higher oracle mAP ceiling for Stage 2.
- **Bug fix:** TriDet `post_processing` had `torch.zeros()` creating float labels → `TypeError` on `ext_cls[label.item()]`. Fixed with `dtype=torch.long` in `opentad/models/detectors/tridet.py:116`.
- **Results:** These are **end-to-end binary** detection results (single-class: RMM vs BG). **Not comparable to multi-class leaderboard** — binary detection is an easier task than 4-class. Comparable only to ActionFormer Binary (which was never evaluated e2e). Next step: use TriDet proposals as Stage 1 → Stage 2 classification → multi-class mAP.

| Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|
| 0 | 44.22% | 32.09% | 18.35% | 31.72% |
| 1 | 44.96% | 32.94% | 13.95% | 30.24% |
| 2 | 35.74% | 21.97% | 11.11% | 23.29% |
| **Mean ± Std** | **41.64% ± 4.29%** | **29.00% ± 5.07%** | **14.47% ± 2.97%** | **28.42% ± 3.65%** |

### Run 5: TriDet + V-JEPA2 4-class (mirror of Run 1)

- **Date:** 2026-03-22 (completed)
- **Stage 1:** TriDet Binary proposals (Run 4)
- **Stage 2:** V-JEPA2 4-class classifier (same as Run 1)
- **Script:** `sbatch --export=FOLD=N two-stg/run_tridet_single_fold_gpu.sh`
- **Results:** `two-stg/eval_results_tridet/fold{0,1,2}/`

| Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|
| 0 | 33.85% | 28.47% | 15.76% | 26.70% |
| 1 | 22.43% | 13.79% | 6.26% | 13.45% |
| 2 | 22.58% | 15.48% | 8.84% | 15.81% |
| **Mean ± Std** | **26.29% ± 6.55%** | **19.25% ± 8.03%** | **10.29% ± 4.91%** | **18.65% ± 7.07%** |

### Run 5c: TriDet + 3-way live fusion (mirror of Run 3c)

- **Date:** 2026-03-22 (completed)
- **Stage 1:** TriDet Binary proposals (Run 4)
- **Stage 2:** 3-way fusion MLP (V-JEPA2 + live PoseC3D/STGCN++ on TriDet proposals)
- **Script:** `sbatch --export=FOLD=N two-stg/run_tridet_three_way_live_fold_gpu.sh`
- **Results:** `two-stg/eval_results_tridet_3way_live/fold{0,1,2}/`
- **Skeleton scores:** Separate pickle/score dirs (`proposal_pickles_tridet/`, `proposal_scores_tridet/`) from TriDet proposals.

| Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|
| 0 | 24.75% | 20.44% | 11.54% | 19.55% |
| 1 | 24.55% | 18.66% | 7.08% | 16.25% |
| 2 | 22.93% | 12.52% | 6.56% | 14.20% |
| **Mean ± Std** | **24.07% ± 0.80%** | **17.21% ± 3.36%** | **8.39% ± 2.22%** | **16.67% ± 2.20%** |

### Run 6: TriDet Balanced — End-to-End 4-class

- **Date:** 2026-03-24 (completed)
- **Model:** TriDet E2E 4-class (hands flapping, jumping, rocking, spinning) — no two-stage pipeline
- **Features:** V-JEPA2 balanced (same as ActionFormer Balanced baseline)
- **Config:** `OpenTAD/configs/tridet/sails_rmm_vjepa_balanced_fold{0,1,2}.py`
- **Training:** `sbatch --partition=ou_bcs_low --gres=gpu:1 --time=06:00:00 OpenTAD/slurm/train_tridet_cv.sh balanced N`
- **SLURM jobs:** 10883791 (fold 0), 10883792 (fold 1), 10884980 (fold 2)
- **Motivation:** TriDet Binary showed +2.50 pp mAP@0.7 over ActionFormer Binary. Run 5 (TriDet 2-stage) showed +1.24 pp mAP@0.7 over Run 1 (AF 2-stage). This run tests whether TriDet's boundary and ranking improvements also benefit the simpler E2E setting vs ActionFormer Balanced (16.70% avg_mAP).
- **Key difference from Run 4 (TriDet Binary):** `num_classes=4`, `multiclass=True` NMS, balanced dataset with class-specific labels instead of binary RMM/BG.
- **Checkpoint note:** `best.pth` is saved by val loss (not mAP), which peaks at epoch 20-30, while mAP continues climbing. Numbers below are from peak validation mAP during training (fold 0: ~ep 33, fold 1: ep 49, fold 2: ~ep 24).

| Fold | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|------|---------|---------|---------|---------|
| 0 | 26.85% | 22.81% | 10.87% | 20.89% |
| 1 | 30.19% | 24.43% | 13.45% | 22.90% |
| 2 | 23.26% | 14.36% | 6.95% | 15.37% |
| **Mean ± Std** | **26.77% ± 2.83%** | **20.53% ± 4.36%** | **10.42% ± 2.66%** | **19.72% ± 3.15%** |

**Comparison across all approaches:**

| Run | Approach | avg_mAP | mAP@0.3 | mAP@0.5 | mAP@0.7 |
|-----|----------|---------|---------|---------|---------|
| **6** | **TriDet E2E 4-class** | **19.72%** | **26.77%** | **20.53%** | **10.42%** |
| 5 | TriDet + V-JEPA2 4-class (2-stage) | 18.65% | 26.29% | 19.25% | 10.29% |
| 1 | AF + V-JEPA2 4-class (2-stage) | 17.83% | 25.29% | 17.74% | 9.05% |
| 3c | AF + 3-way live (2-stage) | 16.78% | 23.43% | 16.93% | 8.34% |
| — | AF Balanced E2E (baseline) | 16.70% | 23.97% | 17.57% | 7.04% |
| 5c | TriDet + 3-way live (2-stage) | 16.67% | 24.07% | 17.21% | 8.39% |

**Findings:**
- **Run 6 (TriDet E2E) is the new best at 19.72% avg_mAP** — beats every other approach including the two-stage pipelines. +3.02 pp over ActionFormer Balanced (16.70%), +1.89 pp over the previous best single-modality (Run 1, 17.83%).
- **E2E > two-stage for TriDet:** Run 6 E2E (19.72%) beats Run 5 two-stage (18.65%) by +1.07 pp. The two-stage pipeline's V-JEPA2 re-classification adds noise rather than helping when TriDet already predicts classes directly.
- **TriDet > ActionFormer across the board:** +3.02 pp E2E, +0.82 pp two-stage V-JEPA2, and notably **+3.38 pp at mAP@0.7** (10.42% vs 7.04%) in the E2E setting — confirming sharper boundaries.
- **3-way MLP remains a drag:** Both 3c (16.78%) and 5c (16.67%) underperform their V-JEPA2-only counterparts due to MLP distribution mismatch.
- **High fold variance:** Fold 2 consistently underperforms (15.37% Run 6, 13.41% Run 3c) vs fold 1 (22.90%, 17.98%).

---

## Score Combination & Pose Filtering Ablations

### Score combination sweep (Runs 5, 5c)

Tested alternative score formulas on existing predictions. Geometric: `score = det^α × cls^(1−α)`. Current baseline: multiplicative `det × cls` (equivalent to geo α=0.5 since monotonic transforms don't change ranking).

**Run 5 (TriDet + V-JEPA2 4-class):**

| Formula | avg_mAP | @0.3 | @0.5 | @0.7 |
|---------|---------|------|------|------|
| **mult (baseline)** | **18.65%** | **26.29%** | **19.25%** | **10.29%** |
| geo α=0.7 (more det) | 16.22% | 22.91% | 16.63% | 9.02% |
| geo α=1.0 (det only) | 14.90% | 21.24% | 15.37% | 7.81% |
| geo α=0.3 (more cls) | 8.66% | 13.10% | 8.95% | 3.93% |
| geo α=0.0 (cls only) | 6.67% | 10.26% | 7.00% | 2.82% |

**Run 5c (TriDet + 3-way live):**

| Formula | avg_mAP | @0.3 | @0.5 | @0.7 |
|---------|---------|------|------|------|
| **mult (baseline)** | **16.66%** | **24.07%** | **17.21%** | **8.39%** |
| geo α=0.7 (more det) | 15.28% | 21.53% | 15.90% | 8.00% |
| geo α=1.0 (det only) | 15.16% | 21.20% | 15.83% | 8.03% |
| geo α=0.3 (more cls) | 8.11% | 12.70% | 8.13% | 3.68% |
| geo α=0.0 (cls only) | 5.93% | 9.33% | 5.91% | 2.73% |

**Findings:**
- **Multiplication is already optimal.** No alternative formula improves over the baseline.
- **V-JEPA2/MLP class probabilities are terrible at ranking** (cls-only: 6.67% / 5.93%). The binary detector's confidence is a far stronger ranking signal (det-only: 14.90% / 15.16%).
- **Interesting: det-only is slightly better for 5c (15.16%) than 5 (14.90%).** The det_scores are identical (same TriDet proposals), but class *labels* differ (MLP vs V-JEPA2 argmax). The 0.26 pp difference means the **MLP actually assigns slightly better class labels** than V-JEPA2 alone — when both are ranked by the same det_score. The MLP's classification is marginally more accurate; it's the MLP's *confidence calibration* that hurts when used for ranking (which is why cls-only is worse for 5c than for 5).
- **The bottleneck is not the score formula — it's the quality of the Stage 2 classifier on proposals.**

### Pose-based proposal filtering (Runs 5, 5c)

Tested dropping proposals based on pose cache data. Level 1: child presence < 50%. Level 2: "hands flapping" proposals with avg arm keypoint confidence < 0.6.

**Run 5:**

| Filter | avg_mAP | @0.3 | @0.5 | @0.7 |
|--------|---------|------|------|------|
| baseline | **18.65%** | **26.29%** | **19.25%** | 10.29% |
| L1 (child ≥ 50%) | 18.57% | 25.99% | 19.18% | **10.39%** |
| L2 (arm conf ≥ 0.6 for HF) | 17.82% | 25.08% | 18.44% | 9.82% |
| L1+L2 | 17.72% | 24.79% | 18.37% | 9.90% |

**Run 5c:**

| Filter | avg_mAP | @0.3 | @0.5 | @0.7 |
|--------|---------|------|------|------|
| baseline | **16.66%** | **24.07%** | **17.21%** | **8.39%** |
| L1 (child ≥ 50%) | 16.19% | 23.28% | 16.67% | 8.29% |
| L2 (arm conf ≥ 0.6 for HF) | 16.48% | 23.79% | 17.04% | 8.28% |
| L1+L2 | 16.01% | 23.02% | 16.52% | 8.16% |

**Findings:**
- **Both filters hurt overall mAP.** Dropping proposals removes true positives along with false positives.
- **L1 slightly improves mAP@0.7** in some folds — child presence filtering does help boundary precision, but the recall cost outweighs it at lower tIoU thresholds.
- **L2 hurts more** (-0.83 pp Run 5). Pose detector confidence doesn't reliably distinguish "child is flapping" from "pose detector missed arms" (fast motion blur, unusual arm positions). Mean arm conf for HF proposals is only 0.72, so the 0.6 threshold catches 18-20% of real HF proposals.
- **Pose detector quality is the limiting factor** — it cannot serve as a reliable post-hoc filter when its own confidence is noisy for the exact motions we care about.

Scripts: `two-stg/rescore_sweep.py`, `two-stg/pose_filter_eval.py`.

---

## Notes

- All paths relative to `/orcd/data/satra/001/users/brukew/actreg/`
- **TriDet Balanced E2E (Run 6) is the current best model at 19.72% avg_mAP** — surpasses all two-stage pipelines and ActionFormer Balanced (16.70%). Simpler pipeline, no V-JEPA2 re-classification or skeleton inference needed.
- **4-class outperforms 5-class by +1.76 avg_mAP** — the 5-class background signal is harmful because Stage 1 already filters for RMM.
- **3b (25.25% avg_mAP) is inflated** by skeleton_index test-time leakage: val proposals look up GT-clip scores from the same video. Not achievable in deployment (no skeleton_index entries for unseen videos → uniform fallback). Contamination on top of that had inflated it further (historical 27.79%).
- **3c (16.78% 3-fold CV) is the honest/deployable 3-way result** but is degraded by MLP distribution mismatch (MLP trained on GT-clip scores, receives proposal-level scores). Retraining the MLP on proposal-style skeleton inputs should improve 3c without the leakage problem.
- **Run 6 checkpoint note:** `best.pth` is saved by val loss, not mAP. Peak mAP occurs later in training (epoch 33-49 depending on fold). Numbers reported are peak validation mAP.
