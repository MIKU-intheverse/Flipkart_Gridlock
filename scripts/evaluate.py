#!/usr/bin/env python3
"""
Performance evaluation (project plan §4.8 / §7).

Computes:
  - Detection: mAP@0.5 via the Ultralytics validator, against a labeled
    YOLO-format validation set.
  - Violation classification: Precision/Recall/F1 against a small manually
    labeled set of (frame, expected_violations) ground truth.
  - OCR: plate-level exact-match accuracy against a manually verified
    plate-text ground truth CSV.

Usage:
    python scripts/evaluate.py --config config/config.yaml --task detection
    python scripts/evaluate.py --config config/config.yaml --task violations --ground-truth data/eval/violations_gt.json
    python scripts/evaluate.py --config config/config.yaml --task ocr --ground-truth data/eval/plates_gt.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_system.utils.config import load_config, ConfigError
from traffic_system.utils.logging_utils import configure_logging, get_logger
from traffic_system.detection.detector import VehicleDetector, ModelLoadError

logger = get_logger("evaluate")


def evaluate_detection(app_config, val_data_yaml: str) -> dict:
    """
    Delegates to Ultralytics' own validator, which computes mAP@0.5 and
    mAP@0.5:0.95 plus per-class AP against a YOLO-format labeled val split.
    `val_data_yaml` follows the standard Ultralytics dataset YAML format
    (train/val paths + class names) and is NOT the same file as config.yaml.
    """
    from ultralytics import YOLO

    weights_path = str(app_config.resolve_path(app_config.detection["weights_path"]))
    model = YOLO(weights_path)
    metrics = model.val(data=val_data_yaml, split="val")

    per_class_ap = {
        name: float(ap) for name, ap in zip(model.names.values(), metrics.box.maps)
    }
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class_AP": per_class_ap,
    }


def evaluate_violations(ground_truth_path: str) -> dict:
    """
    Expects a JSON file shaped as:
        [
          {"violation_type": "helmet_noncompliance", "predicted": true, "actual": true},
          {"violation_type": "helmet_noncompliance", "predicted": true, "actual": false},
          ...
        ]
    i.e. one row per flagged candidate plus a ground-truth label (from
    manual review of a sample of frames), produced separately by running
    the pipeline against a labeled validation clip and recording outcomes.
    This function only computes the metrics from that file — it does not
    run the pipeline itself.
    """
    with open(ground_truth_path, "r") as f:
        records = json.load(f)

    by_type: dict[str, list[dict]] = {}
    for r in records:
        by_type.setdefault(r["violation_type"], []).append(r)

    results = {}
    for vtype, rows in by_type.items():
        tp = sum(1 for r in rows if r["predicted"] and r["actual"])
        fp = sum(1 for r in rows if r["predicted"] and not r["actual"])
        fn = sum(1 for r in rows if not r["predicted"] and r["actual"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results[vtype] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "sample_size": len(rows),
        }
    return results


def evaluate_ocr(ground_truth_path: str) -> dict:
    """
    Expects a CSV with columns: predicted_text, ground_truth_text
    (produced by running the pipeline and pairing its OCR output against
    manually verified plate strings for the same vehicles).
    """
    rows = []
    with open(ground_truth_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {ground_truth_path}")

    exact_matches = 0
    char_accuracy_sum = 0.0

    for row in rows:
        predicted = (row["predicted_text"] or "").strip().upper()
        actual = (row["ground_truth_text"] or "").strip().upper()

        if predicted == actual:
            exact_matches += 1

        max_len = max(len(predicted), len(actual), 1)
        p_padded = predicted.ljust(max_len)
        a_padded = actual.ljust(max_len)
        matches = sum(1 for a, b in zip(p_padded, a_padded) if a == b)
        char_accuracy_sum += matches / max_len

    return {
        "exact_match_accuracy": round(exact_matches / len(rows), 4),
        "char_level_accuracy": round(char_accuracy_sum / len(rows), 4),
        "sample_size": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pipeline components.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--task", required=True, choices=["detection", "violations", "ocr"])
    parser.add_argument(
        "--val-data-yaml", default=None,
        help="Ultralytics dataset YAML, required for --task detection",
    )
    parser.add_argument(
        "--ground-truth", default=None,
        help="Path to ground-truth JSON/CSV, required for --task violations / ocr",
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

    if args.task == "detection":
        if not args.val_data_yaml:
            logger.error("--val-data-yaml is required for --task detection")
            return 1
        results = evaluate_detection(app_config, args.val_data_yaml)

    elif args.task == "violations":
        if not args.ground_truth:
            logger.error("--ground-truth is required for --task violations")
            return 1
        results = evaluate_violations(args.ground_truth)

    else:  # ocr
        if not args.ground_truth:
            logger.error("--ground-truth is required for --task ocr")
            return 1
        results = evaluate_ocr(args.ground_truth)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
