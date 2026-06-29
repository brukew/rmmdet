#!/usr/bin/env python3
"""
Export a deployable 3-way fusion MLP checkpoint for two-stage TAL.

The original fusion CV (train_fusion_cv.py) uses nested inner CV and does not save
model weights. This script:

1. Merges V-JEPA2 + PoseC3D + STGCN++ clip predictions (same as three_way fusion).
2. Trains a single ThreeWayMLPFusion on cross-fold val predictions (decontaminated):
   for fold N, uses val predictions from folds (N+1)%3 and (N+2)%3 (equivalent to fold N's
   training clips), so the MLP never trains on fold N's validation segments.
3. Saves:
   - mlp_state.pt  (state_dict + training meta)
   - config.json
   - skeleton_index.csv  (per clip: video_key, time span, posec3d/stgcn score vectors)

Training hyperparameters match the best 3-way run (hidden_dim=24, dropout=0.1, etc.).

Usage (from actreg root):
  python fusion/export_three_way_deploy_checkpoint.py --fold 0
  python fusion/export_three_way_deploy_checkpoint.py --all-folds
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ACTREG_ROOT = Path(__file__).resolve().parent.parent
if str(ACTREG_ROOT) not in sys.path:
    sys.path.insert(0, str(ACTREG_ROOT))

from fusion.train_fusion_cv import (  # noqa: E402
    DEFAULT_ISSUE_CLIPS_CSV,
    DEFAULT_POSEC3D_ROOT,
    DEFAULT_STGCN_ROOT,
    DEFAULT_VJEPA_ROOT,
    ThreeWayFusionDataset,
    compute_class_weights,
    create_fusion_model,
    load_issue_clips,
    load_skeleton_predictions,
    load_vjepa_predictions,
    merge_three_predictions,
    train_three_way_fusion,
)

logger = logging.getLogger(__name__)


def _resolve_fold_dir(root: Path, fold: int) -> Path:
    """PoseC3D/STGCN++ work_dirs use ``fold0`` or ``fold_0`` depending on run."""
    d0 = root / f"fold{fold}"
    if d0.exists():
        return d0
    d1 = root / f"fold_{fold}"
    if d1.exists():
        return d1
    return d0


def normalize_video_file(video_file: str) -> str:
    from tal.tal_map_eval import normalize_video_file as nv

    return nv(video_file)


def build_skeleton_index(
    merged_df: pd.DataFrame,
    splits_root: Path,
    fold: int,
    num_classes: int,
) -> pd.DataFrame:
    """Attach video_key and half-open [start, end) interval per segment_id."""
    val_path = splits_root / "cv_splits_4class" / f"fold_{fold}_val.csv"
    train_path = splits_root / "cv_splits_4class" / f"fold_{fold}_train.csv"
    rows = []
    for path in (train_path, val_path):
        if not path.exists():
            continue
        rows.append(pd.read_csv(path))
    if not rows:
        raise FileNotFoundError(f"No train/val CSV for fold {fold}")
    seg_df = pd.concat(rows, ignore_index=True)
    seg_df = seg_df.drop_duplicates(subset=["segment_id"])
    seg_df["video_key"] = seg_df["video_file"].apply(normalize_video_file)

    pose_cols = [f"posec3d_score_class{i}" for i in range(num_classes)]
    stgcn_cols = [f"stgcn_score_class{i}" for i in range(num_classes)]
    seg_small = seg_df[["segment_id", "video_key", "start_sec", "end_sec"]]
    m = merged_df[["segment_id"] + pose_cols + stgcn_cols].merge(
        seg_small, on="segment_id", how="inner"
    )
    m["end_sec_exclusive"] = m["end_sec"].astype(float) + 1.0
    out_cols = (
        ["segment_id", "video_key", "start_sec", "end_sec_exclusive"]
        + pose_cols
        + stgcn_cols
    )
    return m[out_cols]


def export_one_fold(
    fold: int,
    output_dir: Path,
    vjepa_root: Path,
    posec3d_root: Path,
    stgcn_root: Path,
    splits_root: Path,
    num_classes: int,
    num_epochs: int,
    lr: float,
    batch_size: int,
    mlp_hidden_dim: int,
    mlp_dropout: float,
    issue_clips_csv: Path,
    device: torch.device,
) -> None:
    exclude = load_issue_clips(issue_clips_csv)

    # Same-fold merged preds: for skeleton_index.csv (eval-time tIoU lookup on val videos).
    vjepa_fold = vjepa_root / f"fold_{fold}"
    pose_fold = _resolve_fold_dir(posec3d_root, fold)
    st_fold = _resolve_fold_dir(stgcn_root, fold)
    vjepa_df = load_vjepa_predictions(vjepa_fold)
    pose_df = load_skeleton_predictions(pose_fold, model_name="PoseC3D")
    st_df = load_skeleton_predictions(st_fold, model_name="STGCN++")
    skel_merged = merge_three_predictions(
        vjepa_df, pose_df, st_df, num_classes, exclude_clips=exclude
    )

    # Cross-fold val preds for MLP training (out-of-sample for this fold's val set).
    other_folds = [f for f in (0, 1, 2) if f != fold]
    train_parts: list[pd.DataFrame] = []
    for of in other_folds:
        vj = load_vjepa_predictions(vjepa_root / f"fold_{of}")
        po = load_skeleton_predictions(_resolve_fold_dir(posec3d_root, of), model_name="PoseC3D")
        st = load_skeleton_predictions(_resolve_fold_dir(stgcn_root, of), model_name="STGCN++")
        train_parts.append(
            merge_three_predictions(vj, po, st, num_classes, exclude_clips=exclude)
        )
    train_merged = pd.concat(train_parts, ignore_index=True)
    logger.info(
        "Fold %d: MLP train rows (cross-fold)=%d, skeleton_index rows (fold val)=%d",
        fold,
        len(train_merged),
        len(skel_merged),
    )

    fold_out = output_dir / f"fold_{fold}"
    fold_out.mkdir(parents=True, exist_ok=True)

    skel_idx = build_skeleton_index(skel_merged, splits_root, fold, num_classes)
    skel_path = fold_out / "skeleton_index.csv"
    skel_idx.to_csv(skel_path, index=False)
    logger.info("Wrote skeleton index: %s (%d clips)", skel_path, len(skel_idx))

    dataset = ThreeWayFusionDataset(train_merged, num_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    class_weights = compute_class_weights(train_merged["label_id"].values, num_classes)

    model = create_fusion_model(
        fusion_type="three_way",
        num_classes=num_classes,
        hidden_dim=mlp_hidden_dim,
        dropout=mlp_dropout,
        init_alpha=0.0,
    ).to(device)

    train_three_way_fusion(
        model=model,
        train_loader=loader,
        class_weights=class_weights,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        log_interval=max(10, num_epochs // 10),
    )

    cfg = {
        "fold": fold,
        "num_classes": num_classes,
        "fusion_type": "three_way",
        "mlp_hidden_dim": mlp_hidden_dim,
        "mlp_dropout": mlp_dropout,
        "num_epochs_full_data_train": num_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "n_train_clips": len(train_merged),
        "mlp_train_folds": list(other_folds),
        "skeleton_index_fold": fold,
        "vjepa_root": str(vjepa_root),
        "posec3d_root": str(posec3d_root),
        "stgcn_root": str(stgcn_root),
        "note": (
            "Decontaminated: MLP trained on val predictions from other CV folds only "
            f"(folds {other_folds}), equivalent to fold {fold} training clips. "
            "skeleton_index.csv built from this fold's val predictions for TAL tIoU lookup. "
            "V-JEPA2 is live per proposal; optional live PoseC3D/STGCN++ proposal CSVs override lookup."
        ),
    }
    with open(fold_out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg,
        },
        fold_out / "mlp_state.pt",
    )
    logger.info("Saved %s", fold_out / "mlp_state.pt")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=None, choices=[0, 1, 2])
    p.add_argument("--all-folds", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ACTREG_ROOT / "two-stg" / "fusion_checkpoints" / "three_way",
    )
    p.add_argument("--vjepa-root", type=Path, default=DEFAULT_VJEPA_ROOT)
    p.add_argument("--posec3d-root", type=Path, default=DEFAULT_POSEC3D_ROOT)
    p.add_argument("--stgcn-root", type=Path, default=DEFAULT_STGCN_ROOT)
    p.add_argument("--splits-root", type=Path, default=ACTREG_ROOT / "dataprep" / "splits")
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--mlp-hidden-dim", type=int, default=24)
    p.add_argument("--mlp-dropout", type=float, default=0.1)
    p.add_argument("--issue-clips-csv", type=Path, default=DEFAULT_ISSUE_CLIPS_CSV)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    folds = [0, 1, 2] if args.all_folds else [args.fold]
    if folds == [None]:
        p.error("Provide --fold N or --all-folds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for f in folds:
        export_one_fold(
            fold=f,
            output_dir=args.output_dir,
            vjepa_root=args.vjepa_root,
            posec3d_root=args.posec3d_root,
            stgcn_root=args.stgcn_root,
            splits_root=args.splits_root,
            num_classes=args.num_classes,
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_dropout=args.mlp_dropout,
            issue_clips_csv=args.issue_clips_csv,
            device=device,
        )


if __name__ == "__main__":
    main()
