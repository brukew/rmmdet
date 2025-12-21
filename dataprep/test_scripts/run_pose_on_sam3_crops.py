from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from sam3_crops_utils import (
    load_parsed_csv,
    load_rotation_report,
    prepare_frame_and_boxes,
)

try:
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover - runtime dependency
    raise ImportError("Install ultralytics to run YOLO pose (pip install ultralytics).") from e


def load_yolo_pose(model_name: str, device: Optional[str] = None):
    """Load a YOLO pose model; model_name examples: yolov8x-pose.pt, yolo11x-pose.pt."""
    model = YOLO(model_name)
    if device:
        model.to(device)
    return model


def crop_with_padding(frame: np.ndarray, box: np.ndarray, pad: int = 20) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop frame around a box with padding and clamp to frame bounds."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def run_pose_on_crops(
    row_idx: int,
    frame_idx: int,
    model,
    videos_data: List[dict],
    rotation_data: dict,
    target_ids: Optional[Iterable[int]] = None,
    pad: int = 20,
    conf: float = 0.25,
    device: Optional[str] = None,
) -> List[dict]:
    """
    Run YOLO pose on crops defined by SAM3 boxes for a specific CSV row/frame.

    Returns a list of dicts containing crop coordinates, object id, and YOLO results.
    """
    data = prepare_frame_and_boxes(
        row_idx=row_idx,
        frame_idx=frame_idx,
        videos_data=videos_data,
        rotation_data=rotation_data,
    )
    if data is None:
        raise RuntimeError(f"Could not prepare frame/boxes for row {row_idx}, frame {frame_idx}")

    frame = data["frame"]
    boxes = data["boxes"]
    obj_ids = data["obj_ids"]
    scores = data["scores"]
    matched = data["matched_indices"]

    if boxes is None or obj_ids is None or scores is None:
        raise RuntimeError("No boxes available for this frame.")

    # Filter which detections to run
    if target_ids:
        keep = [i for i, oid in enumerate(obj_ids) if int(oid) in set(target_ids)]
    elif matched:
        keep = matched
    else:
        keep = list(range(len(obj_ids)))

    results = []
    for i in keep:
        crop_rgb, (cx1, cy1, cx2, cy2) = crop_with_padding(frame, boxes[i], pad=pad)
        if crop_rgb.size == 0:
            continue
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        res = model.predict(crop_bgr, conf=conf, verbose=False, device=device)[0]
        annotated_bgr = res.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        results.append(
            {
                "obj_id": int(obj_ids[i]),
                "score": float(scores[i]),
                "crop_coords": (cx1, cy1, cx2, cy2),
                "crop_rgb": crop_rgb,
                "annotated_rgb": annotated_rgb,
                "yolo_result": res,
            }
        )
    return results


def save_results(results: List[dict], output_dir: Path, prefix: str) -> None:
    """Save annotated crops to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for res in results:
        obj_id = res["obj_id"]
        out_path = output_dir / f"{prefix}_id{obj_id}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(res["annotated_rgb"], cv2.COLOR_RGB2BGR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO pose on SAM3 crops for a specific CSV row/frame.")
    parser.add_argument("--row-idx", type=int, required=True, help="1-based row index in the parsed CSV.")
    parser.add_argument("--frame-idx", type=int, required=True, help="Frame number to process.")
    parser.add_argument("--model", type=str, default="yolov8x-pose.pt", help="YOLO pose checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Device for YOLO (e.g., cuda, cpu).")
    parser.add_argument("--target-ids", type=int, nargs="*", default=None, help="Optional list of SAM3 object IDs to process.")
    parser.add_argument("--pad", type=int, default=20, help="Padding around box when cropping.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--output-dir", type=Path, default=Path("sam3_pose_crops"), help="Directory to save annotated crops.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos_data = load_parsed_csv()
    rotation_data = load_rotation_report()
    model = load_yolo_pose(args.model, device=args.device)

    results = run_pose_on_crops(
        row_idx=args.row_idx,
        frame_idx=args.frame_idx,
        model=model,
        videos_data=videos_data,
        rotation_data=rotation_data,
        target_ids=args.target_ids,
        pad=args.pad,
        conf=args.conf,
        device=args.device,
    )

    prefix = f"row{args.row_idx}_frame{args.frame_idx}"
    save_results(results, args.output_dir, prefix)
    print(f"Saved {len(results)} crops to {args.output_dir}")


if __name__ == "__main__":
    main()
