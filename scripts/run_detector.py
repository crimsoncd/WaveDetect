#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.wave_detector import load_config
from src.wave_detector.pipeline import process_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hot strip wave detector")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--config", default=None, help="YAML config path")
    p.add_argument("--max-frames", type=int, default=None, help="Process at most N frames")
    p.add_argument("--no-video", action="store_true", help="Do not save annotated video")
    p.add_argument("--no-ng", action="store_true", help="Do not save NG frame images")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    summary = process_video(
        video_path=Path(args.video),
        output_dir=Path(args.output),
        cfg=cfg,
        max_frames=args.max_frames,
        save_video=not args.no_video,
        save_ng_images=not args.no_ng,
    )
    print("Run finished")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
