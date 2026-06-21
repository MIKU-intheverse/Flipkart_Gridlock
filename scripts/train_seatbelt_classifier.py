#!/usr/bin/env python3
"""
Trains the seatbelt compliance classifier.

Usage:
    python scripts/train_seatbelt_classifier.py --config config/config.yaml

Mirrors train_helmet_classifier.py exactly, pointed at the
training.seatbelt_classifier config section — kept as a separate script
(rather than one script with a --target flag) so each classifier has its
own clear, greppable entry point and log output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_system.utils.config import load_config, ConfigError
from traffic_system.utils.logging_utils import configure_logging, get_logger
from traffic_system.pipeline import resolve_device
from traffic_system.training.train_loop import train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the seatbelt compliance classifier.")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        app_config = load_config(args.config)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    configure_logging(app_config.system.get("log_level", "INFO"))
    logger = get_logger("train_seatbelt_classifier")

    cfg = app_config.training["seatbelt_classifier"]
    device = resolve_device(app_config.system.get("device", "auto"))
    logger.info("Training seatbelt classifier on device: %s", device)

    data_dir = str(app_config.resolve_path(cfg["data_dir"]))
    output_path = str(app_config.resolve_path(cfg["output_path"]))

    try:
        metrics = train_classifier(
            data_dir=data_dir,
            output_path=output_path,
            image_size=cfg["image_size"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
            val_split=cfg["val_split"],
            device=device,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    logger.info("Training complete: %s", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
