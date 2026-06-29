# Fold 0 — live 3-way vs decontaminated skeleton-index

## Baseline (already on disk)

**Decontaminated 3-way, skeleton_index lookup** — `two-stg/eval_results_3way/fold0/metrics_opentad.json`

| Metric | Value |
|--------|------:|
| avg_mAP (OpenTAD mean) | **28.98%** |
| mAP@0.3 | 38.23% |
| mAP@0.5 | 29.77% |
| mAP@0.7 | 16.85% |

Custom avg@{0.3,0.5,0.7} (if you use `metrics_custom.json`): same story as in `RUNS.md` Run 3b fold 0.

## Submit fold-0 **live** job (one GPU)

From `actreg/` (uses existing `proposal_pickles/proposals_fold0.pkl`, skips rebuild):

```bash
cd /orcd/data/satra/001/users/brukew/actreg
sbatch --export=FOLD=0,SKIP_BUILD_ONLY=1 two-stg/run_three_way_live_fold_gpu.sh
```

- **Logs:** `two-stg/logs/two_stage_3way_live_<jobid>.out` / `.err`
- **Outputs:** `two-stg/eval_results_3way_live/fold0/` (`predictions.csv`, `metrics_opentad.json`, `info.json`)
- **Skeleton CSVs:** `two-stg/proposal_scores/posec3d/fold0/predictions_clip.csv`, `two-stg/proposal_scores/stgcnpp_fused_4stream/fold0/predictions_clip.csv`

If `sbatch` complains about **time limit**, override (example):

```bash
sbatch --time=8:00:00 --export=FOLD=0,SKIP_BUILD_ONLY=1 two-stg/run_three_way_live_fold_gpu.sh
```

Or use another partition your account allows (`#SBATCH --partition=...` in the script).

## After the job finishes

```bash
# Live fold 0
cat two-stg/eval_results_3way_live/fold0/metrics_opentad.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('live avg_mAP', d['avg_mAP']); print('mAP@0.3/0.5/0.7', d['mAP@0.3'], d['mAP@0.5'], d['mAP@0.7'])"

# Baseline (skeleton index)
cat two-stg/eval_results_3way/fold0/metrics_opentad.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('skel_index avg_mAP', d['avg_mAP']); print('mAP@0.3/0.5/0.7', d['mAP@0.3'], d['mAP@0.5'], d['mAP@0.7'])"
```

Then either:

- **If live looks good:** `bash two-stg/run_three_way_live_cv_gpu.sh` (or three `sbatch` for FOLD=1,2 with `SKIP_BUILD_ONLY=1` once pickles exist), and add a row to `RUNS.md` Run 3c.
- **If live is worse:** check `proposals_fold0_skips.json` coverage and missing `segment_id` rows in CSV vs eval (fallback to skeleton index only where CSVs lack IDs — already supported in `stage2_three_way.py`).

## Notes

- **~1k proposals skipped** on fold 0 for pose pickles; live scores exist only for proposals **in** the pickle. Missing IDs fall back to skeleton-index lookup in eval (if implemented).
- Job is long: PoseC3D + 4× STGCN++ on **~7k** clips + V-JEPA2 two-stage eval.
