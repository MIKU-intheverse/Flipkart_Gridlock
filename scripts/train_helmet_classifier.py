#!/usr/bin/env python3
"""
Trains the helmet compliance classifier.

Usage:
    python scripts/train_helmet_classifier.py --config config/config.yaml

Reads all hyperparameters from config.yaml's training.helmet_classifier
section — nothing is hardcoded here, so re-running with different settings
only requires editing the YAML, not this script.
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
    parser = argparse.ArgumentParser(description="Train the helmet compliance classifier.")
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
    logger = get_logger("train_helmet_classifier")

    cfg = app_config.training["helmet_classifier"]
    device = resolve_device(app_config.system.get("device", "auto"))
    logger.info("Training helmet classifier on device: %s", device)

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
