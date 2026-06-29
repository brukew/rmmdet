# 3-Way Live Pipeline — Proposal Coverage Audit (Fold 0)

Date: 2026-03-20 | Job 10716918

## Pipeline flow

```
ActionFormer detections (result_detection.json)
  │  8154 proposals, 7248 unique segment_ids
  ▼
build_proposal_poses.py  (proposal → pose pickle)
  │  7085 kept, 1069 skipped  →  6650 unique segment_ids
  ▼
PoseC3D + STGCN++ (4-stream) inference  (evaluate_sails.py)
  │  7085 rows per CSV, 6650 unique segment_ids
  ▼
fuse_stgcn_4stream.py  (inner join on segment_id)
  │  14035 rows  ← cross-product of duplicates (see below)
  ▼
_load_proposal_score_dict  (CSV → dict, last-wins dedup)
  │  6650 entries for PoseC3D, 6650 for STGCN++
  ▼
eval_two_stage_tal.py  (classifies all 8154 proposals)
     • 6650 unique IDs use live skeleton scores
     • 598 unique IDs fall back to skeleton_index / uniform
```

## Issue 1 — Pose cache truncation (1069 skipped proposals)

**22 of 101** val videos have pose H5 caches shorter than the video.
Proposals beyond the cache get no keypoints and are skipped.

| Skip reason | Count |
|---|---:|
| `starts_after_pose_cache` | 795 |
| `insufficient_pose_coverage` (<10% frames) | 205 |
| `no_pose_cache` | 51 |
| `ends_before_pose_cache` | 18 |
| **Total skipped** | **1069** |

Details in `proposal_pickles/proposals_fold0_skips.json`.

**Impact on eval:** For the 598 unique IDs without live skeleton scores,
`_scores_for_proposal` (stage2_three_way.py:159) falls back to
`_lookup_skeleton_scores`, which uses the best tIoU-matching segment in
`skeleton_index.csv` (98 training-time videos) or uniform 1/4 priors.
V-JEPA2 still runs live on all 8154 proposals — only the skeleton
component degrades.

**Fix (future):** Extend pose H5 caches to cover full video duration, or
regenerate caches for the 22 affected videos.

## Issue 2 — Duplicate segment_ids from ActionFormer

ActionFormer's `result_detection.json` contains **437 duplicate proposal
IDs** — same `(video, start, end)` but different detection scores
(e.g. score=0.368 and score=0.033 for the same interval). These are
legitimate multi-score detections that NMS didn't collapse.

After pose-skip filtering, the pickle retains **420 duplicate IDs**:

| Multiplicity | Unique IDs | Rows |
|---:|---:|---:|
| 2× | 406 | 812 |
| 3× | 13 | 39 |
| 4× | 1 | 4 |
| **1× (unique)** | **6230** | **6230** |
| **Total** | **6650** | **7085** |

### Fused CSV cross-product blow-up

`fuse_stgcn_4stream.py` joins 4 stream CSVs on `segment_id` (inner join).
Duplicates create a **cross product**: 2^4=16 rows, 3^4=81, 4^4=256.

```
6230×1 + 406×16 + 13×81 + 1×256 = 14,035 rows  (matches actual)
```

The eval loader (`_load_proposal_score_dict`) iterates the CSV into a
Python dict keyed on `segment_id` — **last row wins**, collapsing back to
6650 entries. The specific cross-product row kept is arbitrary.

### Score consistency of duplicates

PoseC3D: 409 of 420 duplicate IDs have **inconsistent scores** across
their duplicate rows (test-time augmentation produces different outputs
for identical keypoints in separate annotation entries).

Fused STGCN++: all 420 duplicate IDs have inconsistent scores (inherited
from individual streams + cross-product mixing).

### Impact on mAP

Low. `eval_two_stage_tal.py` processes all 8154 ActionFormer proposals
independently. Duplicate-ID proposals get the same skeleton scores (last-wins)
but different ActionFormer detection scores, so their final score
(`det_score × cls_conf`) differs. Standard TAL mAP handles multiple
detections at the same interval: higher-scoring match first, others are FP.

### Fix (recommended for cleaner results)

Option A — Deduplicate skeleton CSVs before fusion:
```python
# In fuse_stgcn_4stream.py, after loading each stream:
df = df.groupby("segment_id", as_index=False)[score_cols].mean()
```

Option B — Deduplicate in build_proposal_poses.py:
Only create one annotation per unique `format_proposal_id` (keep first or
highest-scoring occurrence).

## Summary for fold-0 pilot

| Metric | Value |
|---|---:|
| Total ActionFormer proposals | 8154 |
| Unique proposal IDs | 7248 |
| With live skeleton scores | 6650 (91.7% of unique) |
| Falling back to skeleton_index | 598 (8.3% of unique) |
| Affected videos (pose cache gap) | 22 / 101 |
| Duplicate IDs in skeleton CSVs | 420 |
| Fused CSV row inflation | 7085 → 14,035 |

Both issues are acceptable for the **pilot fold-0 comparison** (live vs
skeleton-index baseline). The same skeleton_index fallback applies to
Run 3a/3b, so the comparison is fair. For a production run, extend pose
caches and deduplicate before fusion.
