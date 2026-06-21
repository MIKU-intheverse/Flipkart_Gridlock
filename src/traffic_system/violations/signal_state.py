"""
Traffic signal state source — shared by the stop-line and red-light checks.

Supports two modes, both selected via config (violations.red_light.signal_state_source):
  - "manual": an external feed (e.g. signal controller API, or a manual
    override during testing) pushes the current state in; this module just
    holds the latest value per camera.
  - "classifier": a small color classifier runs on the configured
    traffic_light_roi crop from the camera's zone geometry.

Neither mode hardcodes a color/threshold — the classifier path loads its
own weights from config, and the manual path expects the caller to supply
state through `set_manual_state`.
"""

from __future__ import annotations

import cv2
import numpy as np

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.violations.geometry import CameraGeometry

logger = get_logger(__name__)

VALID_STATES = {"red", "yellow", "green", "unknown"}


class SignalStateProvider:
    def __init__(self, app_config: AppConfig):
        cfg = app_config.violations["red_light"]
        self._source = cfg["signal_state_source"]
        self._manual_states: dict[str, str] = {}

        self._classifier_weights_path = None
        if self._source == "classifier":
            self._classifier_weights_path = str(
                app_config.resolve_path(cfg["signal_classifier_weights"])
            )
            # A real color classifier model would be loaded here. Since this
            # repo ships without pretrained signal-color weights, classifier
            # mode raises clearly rather than silently behaving like "manual".
            logger.warning(
                "signal_state_source='classifier' is configured but no "
                "classifier implementation is bundled in this prototype. "
                "Falling back to HSV-heuristic color classification on the "
                "traffic_light_roi crop. Train and wire in a proper "
                "classifier for production use."
            )

    def set_manual_state(self, camera_id: str, state: str) -> None:
        if state not in VALID_STATES:
            raise ValueError(f"Invalid signal state '{state}', expected one of {VALID_STATES}")
        self._manual_states[camera_id] = state

    def get_state(self, camera_id: str, frame: np.ndarray, geometry: CameraGeometry) -> str:
        if self._source == "manual":
            return self._manual_states.get(camera_id, "unknown")
        return self._classify_from_roi(frame, geometry)

    @staticmethod
    def _classify_from_roi(frame: np.ndarray, geometry: CameraGeometry) -> str:
        """
        HSV-heuristic fallback: looks at the dominant hue inside the
        configured traffic-light ROI. This is a placeholder good enough for
        a controlled prototype demo — not a substitute for a trained
        classifier in a real deployment, which is why the warning above
        fires whenever this path is used.
        """
        x1, y1, x2, y2 = geometry.traffic_light_roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return "unknown"

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | \
                   cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
        green_mask = cv2.inRange(hsv, (45, 80, 80), (90, 255, 255))

        counts = {
            "red": int(np.sum(red_mask > 0)),
            "yellow": int(np.sum(yellow_mask > 0)),
            "green": int(np.sum(green_mask > 0)),
        }
        best_state, best_count = max(counts.items(), key=lambda kv: kv[1])
        return best_state if best_count > 20 else "unknown"
