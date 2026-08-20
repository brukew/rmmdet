#!/usr/bin/env python
"""Fine-tune V-JEPA2 for SAILS RMM type classification with cross-validation.

Converted from `finetune_sails_vjepa2_cv.ipynb` for batch/Slurm runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional
    wandb = None

logger = logging.getLogger(__name__)

# Resolve filesystem locations from the repo's single source of truth (config.yaml).
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import PATHS  # noqa: E402

DEFAULT_ISSUE_CLIPS_CSV = PATHS.repo_root / "dataprep/issue_clips.csv"


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger for job output."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger.setLevel(numeric_level)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune V-JEPA2 for SAILS RMM type classification with cross-validation."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=PATHS.repo_root / "dataprep/cv_folds",
        help="Directory containing fold_*_train.csv and fold_*_val.csv files.",
    )
    parser.add_argument(
        "--clips-root",
        type=Path,
        default=PATHS.vjepa2_finetune_clips,
        help="Root directory containing fold subdirectories with MP4 clips.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/vjepa2_rmm_type_cv"),
        help="Where to store per-fold checkpoints and summaries.",
    )
    parser.add_argument("--model-id", default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=32,
        help="Frames per clip; set <=0 to use the processor default.",
    )
    parser.add_argument("--topk", type=int, default=2, help="k for top-k evaluation.")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional cap on folds to run.")
    parser.add_argument(
        "--start-fold",
        type=int,
        default=0,
        help="Skip folds with index < start-fold (useful for resuming).",
    )
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="If a fold output dir exists, load model/processor and only run evaluation/summary.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="Weights & Biases logging mode.",
    )
    parser.add_argument("--wandb-project", default="vjepa-rmm", help="W&B project name.")
    parser.add_argument(
        "--run-prefix",
        default="vjepa2-rmm-cv",
        help="Prefix for W&B run names (fold index is appended).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--issue-clips-csv",
        type=Path,
        default=DEFAULT_ISSUE_CLIPS_CSV,
        help="CSV containing segment_id's to exclude from training/evaluation.",
    )
    return parser.parse_args()


def quality_bucket(val: Optional[float]) -> str:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return "unknown"
    if x >= 4:
        return "high"
    if x >= 3:
        return "medium"
    return "low"


def load_issue_clips(csv_path: Optional[Path] = None) -> set:
    """Load set of segment IDs to exclude from training/evaluation."""
    if csv_path is None:
        csv_path = DEFAULT_ISSUE_CLIPS_CSV
    if not csv_path.exists():
        logger.warning("Issue clips CSV not found at %s; no clips will be excluded.", csv_path)
        return set()
    
    df = pd.read_csv(csv_path)
    issue_clips = set(df["segment_id"].astype(str).tolist())
    logger.info("Loaded %d issue clips to exclude from %s", len(issue_clips), csv_path)
    return issue_clips


def load_split(
    csv_paths: Sequence[Path],
    clips_root: Path,
    exclude_clips: Optional[set] = None,
) -> Tuple[List[Dict], List[Tuple[str, Path]]]:
    records: List[Dict] = []
    missing: List[Tuple[str, Path]] = []
    excluded_count = 0
    
    for csv_path in csv_paths:
        stem = csv_path.stem
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            seg = row.get("segment_id") or row.get("segment_global_id")
            label = row.get("rmm_type")
            if pd.isna(seg) or pd.isna(label):
                continue
            
            # Skip issue clips
            if exclude_clips and str(seg) in exclude_clips:
                excluded_count += 1
                continue
            
            clip_path = clips_root / stem / f"{seg}.mp4"
            if not clip_path.exists():
                missing.append((str(seg), clip_path))
                continue
            rec = {
                "segment_id": seg,
                "label": str(label),
                "clip": clip_path,
                "video_id": row.get("video_id"),
                "n_children": pd.to_numeric(row.get("n_children"), errors="coerce"),
                "n_adults": pd.to_numeric(row.get("n_adults"), errors="coerce"),
                "quality_rating": row.get("quality_rating"),
            }
            records.append(rec)
    
    if excluded_count > 0:
        logger.info("Excluded %d issue clips from split", excluded_count)
    
    return records, missing


def build_video_label_counts(records: Sequence[Dict]) -> Dict[str, int]:
    counts: Dict[str, set] = {}
    for rec in records:
        video_id = rec.get("video_id")
        if video_id is None:
            continue
        counts.setdefault(video_id, set()).add(rec["label"])
    return {k: len(v) for k, v in counts.items()}


def get_frames_per_clip(processor: VJEPA2VideoProcessor, override: int) -> int:
    if override and override > 0:
        return int(override)
    frames = (
        getattr(processor, "num_frames", None)
        or getattr(getattr(processor, "image_processor", processor), "num_frames", None)
        or getattr(getattr(processor, "feature_extractor", processor), "num_frames", None)
        or getattr(getattr(processor, "config", {}), "num_frames", None)
        or getattr(getattr(processor, "config", {}), "frames_per_clip", None)
        or 32
    )
    return int(frames)


class RMMDataset(Dataset):
    def __init__(self, records: List[Dict], label2id: Dict[str, int], frames_per_clip: int, video_label_counts: Dict):
        self.records = records
        self.label2id = label2id
        self.frames_per_clip = frames_per_clip
        self.video_label_counts = video_label_counts

    def __len__(self) -> int:
        return len(self.records)

    def _sample_indices(self, total: int) -> np.ndarray:
        if total <= 0:
            return np.zeros(self.frames_per_clip, dtype="int64")
        return np.round(np.linspace(0, total - 1, self.frames_per_clip)).astype("int64")

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        try:
            vr = VideoReader(str(rec["clip"]), ctx=cpu(0), fault_tol=1)
            indices = self._sample_indices(len(vr))
            frames = vr.get_batch(indices).asnumpy()
        except Exception as exc:  # pragma: no cover - I/O heavy
            logger.warning("Bad clip %s: %s", rec["clip"], exc)
            return None

        label_id = self.label2id[rec["label"]]
        video_id = rec.get("video_id")
        meta = {
            "segment_id": rec.get("segment_id"),
            "video_id": video_id,
            "label_name": rec["label"],
            "n_children": rec.get("n_children"),
            "n_adults": rec.get("n_adults"),
            "quality_rating": rec.get("quality_rating"),
            "quality_bucket": quality_bucket(rec.get("quality_rating")),
            "mixed_video": video_id in self.video_label_counts and self.video_label_counts[video_id] > 1,
        }
        return frames, label_id, meta


def collate_fn(samples, processor: VJEPA2VideoProcessor):
    samples = [s for s in samples if s is not None]
    if not samples:
        return None, None, None
    frame_batches, labels, metas = zip(*samples)
    inputs = processor(list(frame_batches), return_tensors="pt")
    labels_tensor = torch.tensor(labels)
    return inputs, labels_tensor, list(metas)


def aggregate_preds(labels, preds, probs, metas, id2label: Dict[int, str]) -> pd.DataFrame:
    df = pd.DataFrame(metas)
    df["label_id"] = labels
    df["pred_top1"] = preds
    df["label_name"] = df["label_id"].map(id2label)
    df["pred_name"] = df["pred_top1"].map(id2label)
    if probs is not None:
        df["pred_conf"] = probs.max(axis=1)
        # Add per-class probability scores for late fusion
        for i in range(probs.shape[1]):
            df[f"score_class{i}"] = probs[:, i]
    return df


def evaluate(loader: DataLoader, model: torch.nn.Module, device: torch.device, collect_probs: bool = False):
    model.eval()
    correct, total = 0, 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_metas: List[Dict] = []
    all_probs: List[torch.Tensor] = []
    with torch.no_grad():
        for inputs, labels, metas in loader:
            if inputs is None or labels is None:
                continue
            labels = labels.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_metas.extend(metas)
            if collect_probs:
                all_probs.append(torch.softmax(logits, dim=-1).cpu())
    acc = correct / max(total, 1)
    probs_tensor = torch.cat(all_probs, dim=0) if all_probs else None
    return acc, all_preds, all_labels, all_metas, probs_tensor


def evaluate_topk(loader: DataLoader, model: torch.nn.Module, device: torch.device, k: int = 2):
    model.eval()
    correct_top1 = 0
    correct_topk = 0
    total = 0

    all_preds_top1: List[int] = []
    all_preds_topk: List[torch.Tensor] = []
    all_labels: List[int] = []
    all_probs: List[torch.Tensor] = []
    all_metas: List[Dict] = []

    with torch.no_grad():
        for inputs, labels, metas in loader:
            if inputs is None or labels is None:
                continue

            labels = labels.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

            preds_top1 = logits.argmax(-1)
            correct_top1 += (preds_top1 == labels).sum().item()

            _, topk_indices = torch.topk(probs, k, dim=-1)
            correct_topk += torch.any(topk_indices == labels.unsqueeze(-1), dim=-1).sum().item()

            total += labels.size(0)

            all_preds_top1.extend(preds_top1.cpu().tolist())
            all_preds_topk.append(topk_indices.cpu())
            all_labels.extend(labels.cpu().tolist())
            all_probs.append(probs.cpu())
            all_metas.extend(metas)

    top1_acc = correct_top1 / max(total, 1)
    topk_acc = correct_topk / max(total, 1)

    all_preds_topk_tensor = torch.cat(all_preds_topk, dim=0) if all_preds_topk else torch.tensor([])
    all_probs_tensor = torch.cat(all_probs, dim=0) if all_probs else torch.tensor([])

    return {
        "top1_acc": top1_acc,
        f"top{k}_acc": topk_acc,
        "improvement": topk_acc - top1_acc,
        "preds_top1": all_preds_top1,
        "preds_topk": all_preds_topk_tensor,
        "labels": all_labels,
        "probs": all_probs_tensor,
        "metas": all_metas,
    }


def compute_classification_metrics(labels, preds, probs, id2label: Dict[int, str]):
    class_names = [id2label[i] for i in range(len(id2label))]
    label_ids = list(range(len(id2label)))
    report = classification_report(
        labels,
        preds,
        labels=label_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    prec_rec_f1 = precision_recall_fscore_support(
        labels, preds, labels=label_ids, average=None, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class": class_names,
            "precision": prec_rec_f1[0],
            "recall": prec_rec_f1[1],
            "f1": prec_rec_f1[2],
            "support": prec_rec_f1[3],
        }
    )

    # PR curves for top classes by support
    y_true = np.array(labels)
    pr_curves = {}
    if probs is not None and len(class_names) > 0:
        supports = per_class.set_index("class")["support"]
        top_classes = supports.sort_values(ascending=False).head(min(5, len(class_names))).index.tolist()
        for cls in top_classes:
            cls_id = class_names.index(cls)
            y_bin = (y_true == cls_id).astype(int)
            scores = probs[:, cls_id]
            prec, rec, _ = precision_recall_curve(y_bin, scores)
            ap = average_precision_score(y_bin, scores)
            pr_curves[cls] = {"precision": prec, "recall": rec, "ap": ap}

    def _avg(key: str) -> Dict[str, float]:
        return report.get(key, {"f1-score": 0.0, "precision": 0.0, "recall": 0.0})

    macro = _avg("macro avg")
    micro = _avg("micro avg")
    weighted = _avg("weighted avg")

    summary = {
        "macro_f1": macro["f1-score"],
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "micro_f1": micro["f1-score"],
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "weighted_f1": weighted["f1-score"],
    }
    return summary, per_class, pr_curves


def aggregate_video_predictions(
    metas_df: pd.DataFrame, labels: np.ndarray, preds: np.ndarray, probs: Optional[np.ndarray], id2label: Dict[int, str]
):
    """Aggregate clip-level predictions into video-level predictions using average probs or majority vote."""
    if metas_df is None or metas_df.empty or "video_id" not in metas_df:
        logger.warning("No video_id metadata available; skipping video-level aggregation.")
        return None

    valid = metas_df["video_id"].notna()
    if not valid.any():
        logger.warning("video_id missing for all samples; skipping video-level aggregation.")
        return None

    probs_available = probs is not None and len(probs) == len(labels)
    rows = []
    for vid, grp in metas_df[valid].reset_index().groupby("video_id"):
        clip_indices = grp["index"].to_numpy()
        label_ids = labels[clip_indices]
        pred_ids = preds[clip_indices]

        true_label = Counter(label_ids).most_common(1)[0][0]
        if probs_available:
            mean_probs = probs[clip_indices].mean(axis=0)
            pred_label = int(np.argmax(mean_probs))
            pred_conf = float(mean_probs.max())
        else:
            pred_label = Counter(pred_ids).most_common(1)[0][0]
            pred_conf = None
            mean_probs = None

        rows.append(
            {
                "video_id": vid,
                "true_id": int(true_label),
                "pred_id": int(pred_label),
                "true_name": id2label[int(true_label)],
                "pred_name": id2label[int(pred_label)],
                "n_clips": len(grp),
                "pred_conf": pred_conf,
                "mean_probs": mean_probs,
            }
        )

    if not rows:
        return None

    video_df = pd.DataFrame(rows)
    probs_video = None
    if probs_available:
        probs_video = np.stack(video_df["mean_probs"].to_numpy(), axis=0)

    return {
        "df": video_df,
        "labels": video_df["true_id"].to_numpy(),
        "preds": video_df["pred_id"].to_numpy(),
        "probs": probs_video,
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_ids: List[int],
    label_names: List[str],
    out_path: Path,
    normalize: Optional[str] = "true",
) -> None:
    if y_true is None or y_pred is None or len(y_true) == 0:
        logger.warning("Empty inputs; skipping confusion matrix.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=label_ids,
        display_labels=label_names,
        normalize=normalize,
        values_format=".2f" if normalize else "d",
        ax=ax,
        colorbar=False,
    )
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", out_path)


def metrics_for_subset(mask, labels, preds, probs, id2label: Dict[int, str]):
    if mask.sum() == 0:
        return None
    l = labels[mask]
    p = preds[mask]
    pr = probs[mask] if probs is not None else None
    summary, _, _ = compute_classification_metrics(l, p, pr, id2label)
    summary["size"] = len(l)
    return summary


class WandbAdapter:
    def __init__(self, mode: str, project: str, run_name: str, config: Dict):
        self.run = None
        self.enabled = False
        if mode == "disabled":
            logger.info("W&B logging disabled.")
            return
        if wandb is None:
            logger.warning("wandb not installed; disabling logging.")
            return
        self.run = wandb.init(project=project, name=run_name, config=config, mode=mode)
        self.enabled = True

    def log(self, data: Dict):
        if self.enabled:
            wandb.log(data)

    def log_table(self, name: str, dataframe: pd.DataFrame):
        if self.enabled and not dataframe.empty:
            wandb.log({name: wandb.Table(dataframe=dataframe)})

    def finish(self):
        if self.run is not None:
            self.run.finish()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if args.wandb_mode != "disabled" and wandb is None:
        logger.warning('wandb not installed; set --wandb-mode disabled or install wandb to log runs.')

    # Load issue clips to exclude
    issue_clips = load_issue_clips(args.issue_clips_csv)

    train_csvs = sorted(args.csv_dir.glob("fold_*_train.csv"))
    val_csvs = sorted(args.csv_dir.glob("fold_*_val.csv"))
    if not train_csvs or not val_csvs or len(train_csvs) != len(val_csvs):
        raise RuntimeError(f"Found {len(train_csvs)} train CSVs and {len(val_csvs)} val CSVs in {args.csv_dir}")
    logger.info("Found %d folds in %s", len(train_csvs), args.csv_dir)

    all_records, _ = load_split(train_csvs + val_csvs, args.clips_root, exclude_clips=issue_clips)
    video_label_counts = build_video_label_counts(all_records)
    all_labels = sorted({r["label"] for r in all_records})
    label2id = {lbl: i for i, lbl in enumerate(all_labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    logger.info("Labels: %s", label2id)

    base_processor = VJEPA2VideoProcessor.from_pretrained(args.model_id)
    base_frames_per_clip = get_frames_per_clip(base_processor, args.frames_per_clip)
    logger.info("Frames per clip: %d", base_frames_per_clip)

    args.output_root.mkdir(parents=True, exist_ok=True)

    fold_results = []
    fold_pairs = list(zip(train_csvs, val_csvs))
    if args.max_folds is not None:
        fold_pairs = fold_pairs[: args.max_folds]

    for fold_idx, (train_csv, val_csv) in enumerate(fold_pairs):
        if fold_idx < args.start_fold:
            logger.info("Skipping fold %d (start-fold=%d)", fold_idx, args.start_fold)
            continue
        logger.info("=" * 90)
        logger.info("Fold %d | train=%s | val=%s", fold_idx, train_csv.name, val_csv.name)

        fold_out = args.output_root / f"fold_{fold_idx}"
        reuse_existing = args.reuse_checkpoints and fold_out.exists()
        fold_out.mkdir(parents=True, exist_ok=True)
        if reuse_existing:
            fold_processor = VJEPA2VideoProcessor.from_pretrained(fold_out)
            frames_per_clip_fold = get_frames_per_clip(fold_processor, args.frames_per_clip)
            logger.info(
                "Reusing checkpoint for fold %d from %s (frames_per_clip=%d)",
                fold_idx,
                fold_out,
                frames_per_clip_fold,
            )
        else:
            fold_processor = base_processor
            frames_per_clip_fold = base_frames_per_clip

        train_records, miss_train = load_split([train_csv], args.clips_root, exclude_clips=issue_clips)
        val_records, miss_val = load_split([val_csv], args.clips_root, exclude_clips=issue_clips)
        logger.info("Missing clips -> train: %d | val: %d", len(miss_train), len(miss_val))
        if miss_train or miss_val:
            example_missing = (miss_train + miss_val)[:5]
            logger.warning("Example missing clips: %s", example_missing)

        train_ds = RMMDataset(train_records, label2id, frames_per_clip_fold, video_label_counts)
        val_ds = RMMDataset(val_records, label2id, frames_per_clip_fold, video_label_counts)

        collate = partial(collate_fn, processor=fold_processor)
        train_loader = None
        if not reuse_existing:
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
                prefetch_factor=2 if args.num_workers > 0 else None,
            )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        # Class weights to rebalance loss toward underrepresented labels
        cls_counts = Counter(rec["label"] for rec in train_records)
        weights = []
        total_train = sum(cls_counts.values())
        for i in range(len(label2id)):
            cls = id2label[i]
            count = max(cls_counts.get(cls, 0), 1)  # avoid divide-by-zero
            weights.append(total_train / (len(label2id) * count))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        logger.info("Fold %d class weights: %s", fold_idx, dict(zip(id2label.values(), weights)))

        if reuse_existing:
            model = VJEPA2ForVideoClassification.from_pretrained(fold_out).to(device)
        else:
            model = VJEPA2ForVideoClassification.from_pretrained(
                args.model_id,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True,
            ).to(device)
            for param in model.vjepa2.parameters():
                param.requires_grad = False

        if not reuse_existing:
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

        run_name = f"{args.run_prefix}-fold{fold_idx}-vjepa2-{frames_per_clip_fold}fr"
        wb_logger = WandbAdapter(
            mode=args.wandb_mode,
            project=args.wandb_project,
            run_name=run_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "frames_per_clip": frames_per_clip_fold,
                "fold": fold_idx,
                "accumulation_steps": args.accumulation_steps,
            },
        )

        if reuse_existing:
            logger.info("Skipping training for fold %d (reuse-checkpoints).", fold_idx)
        else:
            for epoch in range(1, args.num_epochs + 1):
                model.train()
                optimizer.zero_grad()
                running_loss = 0.0
                num_batches = 0

                for step, (inputs, labels, metas) in enumerate(train_loader, start=1):
                    if inputs is None or labels is None:
                        continue
                    labels = labels.to(device)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    loss = F.cross_entropy(outputs.logits, labels, weight=class_weights) / args.accumulation_steps
                    loss.backward()
                    running_loss += loss.item()
                    num_batches += 1

                    if step % args.accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    if step % args.log_interval == 0:
                        avg_loss = running_loss / max(num_batches, 1) * args.accumulation_steps
                        logger.info("  Epoch %d Step %d/%d | Loss: %.4f", epoch, step, len(train_loader), avg_loss)

                    wb_logger.log({"train/loss": loss.item(), "epoch": epoch, "step": step})

                if num_batches and num_batches % args.accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()

                val_acc, _, _, _, _ = evaluate(val_loader, model, device)
                wb_logger.log({"val/acc": val_acc, "epoch": epoch})
                logger.info("  Epoch %d complete | val_acc=%.3f", epoch, val_acc)

            # Save checkpoint per fold
            fold_out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(fold_out)
            fold_processor.save_pretrained(fold_out)
            logger.info("Saved fold %d model + processor to %s", fold_idx, fold_out)

        # Full eval with top-k and metrics
        topk_metrics = evaluate_topk(val_loader, model, device, k=args.topk)
        labels_arr = np.array(topk_metrics["labels"])
        preds_arr = np.array(topk_metrics["preds_top1"])
        probs_arr = (
            topk_metrics["probs"].cpu().numpy() if isinstance(topk_metrics["probs"], torch.Tensor) else None
        )
        metas_df = aggregate_preds(labels_arr, preds_arr, probs_arr, topk_metrics["metas"], id2label)
        clip_preds_path = fold_out / "clip_level_preds.csv"
        metas_df.to_csv(clip_preds_path, index=False)
        logger.info("Saved clip-level predictions to %s", clip_preds_path)

        summary_metrics, per_class_df, pr_curves = compute_classification_metrics(
            labels_arr, preds_arr, probs_arr, id2label
        )

        # Video-level aggregation
        video_metrics = aggregate_video_predictions(metas_df, labels_arr, preds_arr, probs_arr, id2label)
        video_summary = None
        video_per_class_df = None
        video_acc = None
        video_kappa = None
        if video_metrics:
            video_labels = video_metrics["labels"]
            video_preds = video_metrics["preds"]
            video_probs = video_metrics["probs"]
            video_summary, video_per_class_df, _ = compute_classification_metrics(
                video_labels, video_preds, video_probs, id2label
            )
            video_acc = float((video_labels == video_preds).mean()) if len(video_labels) else 0.0
            video_kappa = float(cohen_kappa_score(video_labels, video_preds)) if len(video_labels) else float("nan")

            video_conf_path = fold_out / "confusion_matrix_video.png"
            save_confusion_matrix(
                video_labels,
                video_preds,
                list(id2label.keys()),
                list(id2label.values()),
                video_conf_path,
                normalize="true",
            )

            video_pred_path = fold_out / "video_level_preds.csv"
            video_metrics["df"].to_csv(video_pred_path, index=False)
            logger.info("Saved video-level predictions to %s", video_pred_path)
        else:
            logger.warning("Skipping video-level metrics for fold %d; missing video_id metadata.", fold_idx)

        # Persist per-class tables
        per_class_path = fold_out / "per_class_clip.csv"
        per_class_df.to_csv(per_class_path, index=False)
        if video_per_class_df is not None:
            video_per_class_path = fold_out / "per_class_video.csv"
            video_per_class_df.to_csv(video_per_class_path, index=False)

        # Subset masks (clip-level)
        masks = {
            "n_children_1": metas_df["n_children"] == 1,
            "n_children_gt1": metas_df["n_children"] > 1,
            "mixed_video": metas_df["mixed_video"] == True,  # noqa: E712
        }
        for qb in metas_df["quality_bucket"].dropna().unique():
            masks[f"quality_{qb}"] = metas_df["quality_bucket"] == qb

        subset_metrics: Dict[str, Dict] = {}
        for name, mask in masks.items():
            res = metrics_for_subset(mask.values, labels_arr, preds_arr, probs_arr, id2label)
            if res:
                subset_metrics[name] = res

        fold_results.append(
            {
                "fold": fold_idx,
                "train_csv": train_csv.name,
                "val_csv": val_csv.name,
                "top1_acc": topk_metrics["top1_acc"],
                "topk_acc": topk_metrics[f"top{args.topk}_acc"],
                "improvement": topk_metrics["improvement"],
                **summary_metrics,
                "per_class": per_class_df,
                "pr_curves": pr_curves,
                "subset_metrics": subset_metrics,
                "video_acc": video_acc,
                "video_macro_f1": video_summary["macro_f1"] if video_summary else None,
                "video_kappa": video_kappa,
                "video_per_class": video_per_class_df,
                "metas": metas_df,
            }
        )

        wb_logger.log(
            {
                "eval/top1_acc": topk_metrics["top1_acc"],
                f"eval/top{args.topk}_acc": topk_metrics[f"top{args.topk}_acc"],
                "eval/topk_improvement": topk_metrics["improvement"],
                "eval/macro_f1": summary_metrics["macro_f1"],
                "eval/micro_f1": summary_metrics["micro_f1"],
                "eval/weighted_f1": summary_metrics["weighted_f1"],
            }
        )
        wb_logger.log_table("eval/per_class", per_class_df)
        if video_summary and video_per_class_df is not None:
            wb_logger.log(
                {
                    "eval/video_acc": video_acc,
                    "eval/video_macro_f1": video_summary["macro_f1"],
                    "eval/video_kappa": video_kappa,
                }
            )
            wb_logger.log_table("eval/per_class_video", video_per_class_df)
        if subset_metrics:
            subset_rows = []
            for name, metrics in subset_metrics.items():
                row = {"subset": name}
                row.update(metrics)
                subset_rows.append(row)
            wb_logger.log_table("eval/subsets", pd.DataFrame(subset_rows))
        wb_logger.finish()

        logger.info(
            "Fold %d | top1=%.3f | top%d=%.3f | macro F1=%.3f",
            fold_idx,
            topk_metrics["top1_acc"],
            args.topk,
            topk_metrics[f"top{args.topk}_acc"],
            summary_metrics["macro_f1"],
        )
        if video_summary:
            logger.info(
                "Fold %d (video-level) | acc=%.3f | macro F1=%.3f | kappa=%.3f",
                fold_idx,
                video_acc,
                video_summary["macro_f1"],
                video_kappa,
            )

    # Aggregate cross-fold results
    summary_rows = [
        {
            "fold": fr["fold"],
            "top1_acc": fr["top1_acc"],
            f"top{args.topk}_acc": fr["topk_acc"],
            "macro_f1": fr["macro_f1"],
            "micro_f1": fr["micro_f1"],
            "weighted_f1": fr["weighted_f1"],
            "video_acc": fr.get("video_acc"),
            "video_macro_f1": fr.get("video_macro_f1"),
            "video_kappa": fr.get("video_kappa"),
        }
        for fr in fold_results
    ]
    summary_df = pd.DataFrame(summary_rows)
    logger.info("Cross-fold summary:\n%s", summary_df)
    logger.info("Averages:\n%s", summary_df.mean(numeric_only=True))

    summary_path = args.output_root / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved cross-fold summary to %s", summary_path)

    # Save minimal JSON for programmatic use
    json_path = args.output_root / "cv_summary.json"
    json_path.write_text(json.dumps(summary_rows, indent=2))
    logger.info("Saved cross-fold summary JSON to %s", json_path)


if __name__ == "__main__":
    main()
