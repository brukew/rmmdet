#!/usr/bin/env python3
"""
Quick OpenCV VideoCapture probe.

Opens a video, prints basic metadata, then attempts to read a limited number
of frames to exercise the decoder (useful for spotting HEVC issues).
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2


def fourcc_to_str(fourcc: float) -> str:
    """Convert numeric fourcc to string."""
    i = int(fourcc)
    return "".join([chr((i >> 8 * j) & 0xFF) for j in range(4)])


def scan_paths(root: Path, suffixes: Tuple[str, ...]) -> List[Path]:
    """Yield video files under root matching suffixes."""
    if root.is_file():
        return [root]
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes])


def probe_video(video_path: Path, max_frames: int, loglevel: int) -> Tuple[bool, int, str]:
    """Try to read frames from a single video. Returns (ok, successes, message)."""
    try:
        cv2.setLogLevel(loglevel)
    except Exception:
        pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, 0, "Failed to open"

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    fps_str = f"{fps:.3f}" if fps else "0"

    print(
        f"Opened: {video_path} | {width}x{height} fps={fps_str} "
        f"frames={total_frames} fourcc='{fourcc_to_str(fourcc)}'"
    )

    successes = 0
    failures = 0
    for idx in range(max_frames):
        try:
            ret, frame = cap.read()
        except Exception as e:
            failures += 1
            return False, successes, f"Exception at frame {idx}: {e}"

        if not ret or frame is None:
            # Treat clean EOF at or past reported frame count as success
            if total_frames and successes >= total_frames - 1:
                return True, successes, "EOF (at reported frame count)"
            failures += 1
            return False, successes, f"Read failed at frame {idx}"

        successes += 1
    return failures == 0, successes, "OK"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test cv2.VideoCapture on one or more videos (recursive allowed)."
    )
    parser.add_argument("path", help="Video file or directory to scan.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum number of frames to attempt to read per video (default: 100).",
    )
    parser.add_argument(
        "--loglevel",
        type=int,
        default=0,
        help="OpenCV log level (0=silence). Default: 0.",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=[".mp4", ".mov", ".mkv"],
        help="File extensions to include when scanning a directory (default: .mp4 .mov .mkv).",
    )
    args = parser.parse_args()

    root = Path(args.path).expanduser()
    if not root.exists():
        print(f"Path does not exist: {root}")
        return 1

    videos = scan_paths(root, tuple(ext.lower() for ext in args.ext))
    if not videos:
        print("No videos found.")
        return 1

    print(f"Found {len(videos)} video(s). Probing up to {args.max_frames} frames each.")
    corrupted: List[Path] = []

    for vid in videos:
        ok, successes, msg = probe_video(vid, args.max_frames, args.loglevel)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {vid} ({successes} frames) -> {msg}")
        if not ok:
            corrupted.append(vid)

    if corrupted:
        print("\nCorrupted/failed videos:")
        for c in corrupted:
            print(f"- {c}")
        return 2

    print("\nAll videos passed the probe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
