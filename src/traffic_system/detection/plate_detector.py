"""
License plate localization model — a separate, smaller YOLOv10 model
trained only to find plate rectangles, run on a vehicle crop rather than
the full frame. Kept in `detection/` since it's structurally the same kind
of model as the main vehicle detector, just with a different weights file
and a single class.
"""

from __future__ import annotations

import numpy as np

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import BBox
from traffic_system.detection.detector import ModelLoadError

logger = get_logger(__name__)


class PlateDetector:
    def __init__(self, app_config: AppConfig):
        cfg = app_config.lpr
        self._weights_path = str(app_config.resolve_path(cfg["plate_detector_weights"]))
        self._conf_threshold = cfg["plate_confidence_threshold"]
        self._device = app_config.system.get("device", "auto")
        self._model = self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ModelLoadError(
                "ultralytics package is required for plate detection."
            ) from e
        try:
            model = YOLO(self._weights_path)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load plate detector weights from '{self._weights_path}'. "
                f"Check lpr.plate_detector_weights in config.yaml. Original error: {e}"
            ) from e
        logger.info("Loaded plate detector from %s", self._weights_path)
        return model

    def locate_plate(self, vehicle_crop: np.ndarray) -> BBox | None:
        """
        Returns the highest-confidence plate bbox found in the crop
        (in the crop's own coordinate space), or None if no plate found.
        """
        if vehicle_crop.size == 0:
            return None

        device_arg = None if self._device == "auto" else self._device
        results = self._model.predict(
            vehicle_crop, conf=self._conf_threshold, device=device_arg, verbose=False
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        best_idx = int(np.argmax([float(c) for c in boxes.conf]))
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[best_idx]]
        return BBox(x1, y1, x2, y2)
