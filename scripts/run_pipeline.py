#!/usr/bin/env python3
"""
Entry point for running the full traffic violation detection pipeline.

Usage:
    python scripts/run_pipeline.py --config config/config.yaml
    python scripts/run_pipeline.py --config config/config.yaml --camera cam_01
    python scripts/run_pipeline.py --config config/config.yaml --max-frames 500

Every camera listed in config.yaml's `sources` is run sequentially in this
single-process version; for true multi-camera concurrency, run one process
per camera (or per cluster) — see the deployment notes in README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_system.utils.config import load_config, ConfigError
from traffic_system.utils.logging_utils import configure_logging, get_logger
from traffic_system.utils.video_source import VideoSource
from traffic_system.detection.detector import VehicleDetector, ModelLoadError
from traffic_system.storage.db import Database
from traffic_system.storage.repository import ViolationRepository
from traffic_system.pipeline import CameraPipeline, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the traffic violation detection pipeline.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--camera", default=None, help="Only run this camera_id (default: all)")
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after this many processed frames per camera (default: run until source ends)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        app_config = load_config(args.config)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    configure_logging(app_config.system.get("log_level", "INFO"))
    logger = get_logger("run_pipeline")

    device = resolve_device(app_config.system.get("device", "auto"))
    logger.info("Using device: %s", device)

    try:
        detector = VehicleDetector(app_config)
    except ModelLoadError as e:
        logger.error(str(e))
        return 1

    db = Database(app_config)
    db.create_all()
    repository = ViolationRepository(db)

    sources = app_config.sources
    if args.camera:
        sources = [s for s in sources if s.camera_id == args.camera]
        if not sources:
            logger.error("No source with camera_id='%s' found in config.", args.camera)
            return 1

    target_fps = app_config.system["target_fps"]

    for source_config in sources:
        logger.info("Starting camera '%s' (%s)", source_config.camera_id, source_config.uri)
        camera_pipeline = CameraPipeline(
            app_config=app_config,
            source_config=source_config,
            detector=detector,
            repository=repository,
            device=device,
        )

        processed = 0
        with VideoSource(source_config, target_fps) as video_source:
            for raw_frame in video_source.frames():
                camera_pipeline.process_one_frame(raw_frame)
                processed += 1
                if args.max_frames is not None and processed >= args.max_frames:
                    logger.info(
                        "Reached --max-frames=%d for camera '%s', stopping.",
                        args.max_frames, source_config.camera_id,
                    )
                    break

        logger.info("Finished camera '%s' — processed %d frames", source_config.camera_id, processed)
        camera_pipeline.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
