"""
3-way MLP stage-2 classifier for two-stage TAL (V-JEPA2 + PoseC3D + STGCN++).

V-JEPA2 probabilities are computed live on each detector proposal (RGB clip).

PoseC3D and STGCN++ vectors default to precomputed clip-level scores in
``skeleton_index.csv`` (nearest training clip by half-open tIoU; uniform fallback).

If ``pose_proposal_scores_csv`` and ``stgcn_proposal_scores_csv`` are provided
(aligned ``segment_id`` = :func:`format_proposal_id`), those per-proposal scores
are used first; missing IDs fall back to the skeleton index.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def format_proposal_id(af_key: str, start_sec: float, end_sec: float) -> str:
    """Deterministic ID shared with build_proposal_poses.py and eval_two_stage_tal.py."""
    return f"{af_key}__{start_sec:.3f}_{end_sec:.3f}"


def _tiou_half_open(a0: float, a1: float, b0: float, b1: float) -> float:
    """tIoU for half-open intervals [a0, a1) and [b0, b1)."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    len_a = max(0.0, a1 - a0)
    len_b = max(0.0, b1 - b0)
    union = len_a + len_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _load_proposal_score_dict(path: Path, num_classes: int) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    cols = [f"score_class{i}" for i in range(num_classes)]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"{path} missing column {c}")
    out: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        sid = str(row["segment_id"])
        out[sid] = np.array([float(row[c]) for c in cols], dtype=np.float32)
    return out


class ThreeWayStage2:
    def __init__(
        self,
        bundle_dir: Path,
        *,
        device: str | torch.device = "cuda",
        num_classes: int = 4,
        pose_proposal_scores_csv: Path | None = None,
        stgcn_proposal_scores_csv: Path | None = None,
    ):
        self.bundle_dir = Path(bundle_dir)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_classes = num_classes

        cfg_path = self.bundle_dir / "config.json"
        with open(cfg_path, encoding="utf-8") as f:
            self.config = json.load(f)

        mlp_path = self.bundle_dir / "mlp_state.pt"
        try:
            blob = torch.load(mlp_path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(mlp_path, map_location="cpu")
        actreg_root = self.bundle_dir.resolve().parents[3]
        import sys

        if str(actreg_root) not in sys.path:
            sys.path.insert(0, str(actreg_root))
        from fusion.train_fusion_cv import ThreeWayMLPFusion

        hidden = int(self.config.get("mlp_hidden_dim", 24))
        dropout = float(self.config.get("mlp_dropout", 0.1))
        self.model = ThreeWayMLPFusion(
            num_classes=num_classes, hidden_dim=hidden, dropout=dropout
        )
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        skel_path = self.bundle_dir / "skeleton_index.csv"
        skel_df = pd.read_csv(skel_path)
        self.pose_cols = [f"posec3d_score_class{i}" for i in range(num_classes)]
        self.stgcn_cols = [f"stgcn_score_class{i}" for i in range(num_classes)]

        self._by_video: dict[str, list[dict[str, Any]]] = {}
        for _, row in skel_df.iterrows():
            vk = str(row["video_key"])
            rec = {
                "start": float(row["start_sec"]),
                "end": float(row["end_sec_exclusive"]),
                "pose": np.array([float(row[c]) for c in self.pose_cols], dtype=np.float32),
                "stgcn": np.array([float(row[c]) for c in self.stgcn_cols], dtype=np.float32),
            }
            self._by_video.setdefault(vk, []).append(rec)

        uni = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)
        self._uniform_pose = uni
        self._uniform_stgcn = uni.copy()

        self._proposal_pose: dict[str, np.ndarray] | None = None
        self._proposal_stgcn: dict[str, np.ndarray] | None = None
        if pose_proposal_scores_csv is not None and stgcn_proposal_scores_csv is not None:
            pp = Path(pose_proposal_scores_csv)
            sp = Path(stgcn_proposal_scores_csv)
            self._proposal_pose = _load_proposal_score_dict(pp, num_classes)
            self._proposal_stgcn = _load_proposal_score_dict(sp, num_classes)
            logger.info(
                "Loaded proposal score CSVs: PoseC3D %d ids, STGCN++ %d ids",
                len(self._proposal_pose),
                len(self._proposal_stgcn),
            )
        elif pose_proposal_scores_csv is not None or stgcn_proposal_scores_csv is not None:
            raise ValueError(
                "Provide both pose_proposal_scores_csv and stgcn_proposal_scores_csv, or neither."
            )

        logger.info(
            "ThreeWayStage2 loaded from %s (%d videos in skeleton index)",
            self.bundle_dir,
            len(self._by_video),
        )

    def _lookup_skeleton_scores(self, video_key: str, p0: float, p1: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (pose_vec, stgcn_vec) for best matching clip; uniform if none."""
        clips = self._by_video.get(video_key)
        if not clips:
            return self._uniform_pose.copy(), self._uniform_stgcn.copy()

        best_tiou = 0.0
        best_pose = self._uniform_pose
        best_stgcn = self._uniform_stgcn
        for c in clips:
            t = _tiou_half_open(p0, p1, c["start"], c["end"])
            if t > best_tiou:
                best_tiou = t
                best_pose = c["pose"]
                best_stgcn = c["stgcn"]
        if best_tiou <= 0:
            return self._uniform_pose.copy(), self._uniform_stgcn.copy()
        return best_pose.copy(), best_stgcn.copy()

    def _scores_for_proposal(
        self,
        video_key: str,
        p0: float,
        p1: float,
        proposal_id: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            proposal_id
            and self._proposal_pose is not None
            and self._proposal_stgcn is not None
            and proposal_id in self._proposal_pose
            and proposal_id in self._proposal_stgcn
        ):
            return (
                self._proposal_pose[proposal_id].copy(),
                self._proposal_stgcn[proposal_id].copy(),
            )
        return self._lookup_skeleton_scores(video_key, p0, p1)

    @torch.no_grad()
    def fuse_batch(
        self,
        vjepa_probs: torch.Tensor,
        video_keys: list[str],
        segments: list[tuple[float, float]],
        proposal_ids: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vjepa_probs: (B, C) softmax probabilities from live V-JEPA2
            video_keys: length B (GT normalized video path)
            segments: (start, end) half-open proposal interval per row
            proposal_ids: optional per-row IDs matching precomputed skeleton CSV rows

        Returns:
            fused_probs: (B, C)
            fused_conf: (B,) max prob per row
        """
        bsz = vjepa_probs.shape[0]
        zp = []
        zs = []
        for i in range(bsz):
            pid = proposal_ids[i] if proposal_ids else None
            pv, sv = self._scores_for_proposal(
                video_keys[i], segments[i][0], segments[i][1], pid
            )
            zp.append(torch.from_numpy(pv))
            zs.append(torch.from_numpy(sv))
        z_posec3d = torch.stack(zp).to(self.device)
        z_stgcn = torch.stack(zs).to(self.device)
        z_v = vjepa_probs.to(self.device)

        logits = self.model(z_v, z_posec3d, z_stgcn)
        probs = F.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values
        return probs, conf


def default_bundle_dir(actreg_root: Path, fold: int) -> Path:
    return actreg_root / "two-stg" / "fusion_checkpoints" / "three_way" / f"fold_{fold}"
