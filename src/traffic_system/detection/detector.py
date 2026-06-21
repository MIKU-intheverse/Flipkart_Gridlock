"""
Stage 2: Vehicle & Road User Detection.

Thin, config-driven wrapper around a YOLOv10 model (Ultralytics API). All
thresholds, weight paths, and class-id-to-name mapping come from
config.yaml's `detection` section — nothing here hardcodes a class name or
a confidence number, so swapping the trained weights or retraining with a
different class list never requires touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import BBox, Detection

logger = get_logger(__name__)


class ModelLoadError(Exception):
    pass


class VehicleDetector:
    """
    Wraps a YOLOv10 (.pt) model. Built against the Ultralytics `YOLO` API,
    which YOLOv10 checkpoints are compatible with directly.
    """

    def __init__(self, app_config: AppConfig):
        cfg = app_config.detection
        self._weights_path = str(app_config.resolve_path(cfg["weights_path"]))
        self._conf_threshold = cfg["confidence_threshold"]
        self._iou_threshold = cfg["iou_threshold"]
        self._image_size = cfg["image_size"]
        self._class_map: dict[int, str] = cfg["classes"]
        self._device = app_config.system.get("device", "auto")

        self._model = self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ModelLoadError(
                "ultralytics package is required for detection. "
                "Install with: pip install ultralytics"
            ) from e

        try:
            model = YOLO(self._weights_path)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load detection weights from '{self._weights_path}'. "
                f"Check detection.weights_path in config.yaml. Original error: {e}"
            ) from e

        logger.info("Loaded detection model from %s", self._weights_path)
        return model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Runs one forward pass and returns Detection objects with class names
        resolved via the config's class map rather than the model's own
        internal names, so a mismatch between the weights file and the
        config is caught early instead of silently using the wrong label.
        """
        device_arg = None if self._device == "auto" else self._device

        results = self._model.predict(
            frame,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            imgsz=self._image_size,
            device=device_arg,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        detections: list[Detection] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self._class_map.get(class_id)
            if class_name is None:
                logger.warning(
                    "Detected class_id=%d has no entry in config.detection.classes; skipping. "
                    "Update config.yaml if this class should be tracked.",
                    class_id,
                )
                continue

            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append(Detection(
                class_name=class_name,
                confidence=float(box.conf[0]),
                bbox=BBox(x1, y1, x2, y2),
            ))

        return detections

    @property
    def vehicle_class_names(self) -> set[str]:
        return set(self._class_map.values())
