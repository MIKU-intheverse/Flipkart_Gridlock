"""
Wrong-side driving check (project plan §4.3 / problem statement requirement).

Compares a tracked vehicle's recent motion direction (from its track
history, not a single frame) against the camera's configured lane
direction vector using a cosine-angle test. No model involved.
"""

from __future__ import annotations

import math
from datetime import datetime

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import TrackedObject, ViolationRecord
from traffic_system.violations.geometry import CameraGeometry

logger = get_logger(__name__)


def _normalize(v: tuple[float, float]) -> tuple[float, float]:
    mag = math.hypot(v[0], v[1])
    if mag < 1e-6:
        return (0.0, 0.0)
    return (v[0] / mag, v[1] / mag)


def _angle_between_degrees(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    n1, n2 = _normalize(v1), _normalize(v2)
    dot = n1[0] * n2[0] + n1[1] * n2[1]
    dot = max(-1.0, min(1.0, dot))   # clamp for numerical safety before acos
    return math.degrees(math.acos(dot))


class WrongSideChecker:
    def __init__(self, app_config: AppConfig):
        cfg = app_config.violations["wrong_side"]
        self._enabled = cfg["enabled"]
        self._angle_threshold = cfg["angle_threshold_degrees"]
        self._min_history = cfg["min_track_history_for_direction"]

    def check(
        self,
        vehicles: list[TrackedObject],
        geometry: CameraGeometry,
        camera_id: str,
        frame_id: int,
        timestamp: datetime,
    ) -> list[ViolationRecord]:
        if not self._enabled:
            return []

        results: list[ViolationRecord] = []
        lane_vector = geometry.lane_direction_vector

        for vehicle in vehicles:
            if len(vehicle.history) < self._min_history:
                continue   # not enough motion history yet to judge direction reliably

            travel_vector = self._compute_travel_direction(vehicle)
            if travel_vector == (0.0, 0.0):
                continue   # effectively stationary, direction undefined

            angle = _angle_between_degrees(travel_vector, lane_vector)
            if angle >= self._angle_threshold:
                results.append(ViolationRecord(
                    violation_type="wrong_side_driving",
                    camera_id=camera_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    track_id=vehicle.track_id,
                    vehicle_class=vehicle.class_name,
                    bbox=vehicle.bbox,
                    detector_confidence=vehicle.detection_confidence,
                    classifier_confidence=1.0,
                    rule_certainty=0.9,
                    extra={"angle_degrees": angle},
                ))

        return results

    @staticmethod
    def _compute_travel_direction(vehicle: TrackedObject) -> tuple[float, float]:
        """
        Uses the displacement between the earliest and latest recorded
        centroid in the track's history rather than a single-frame velocity,
        which is far less noisy for a slow-moving or briefly-jittering track.
        """
        first = vehicle.history[0]
        last = vehicle.history[-1]
        return (last[0] - first[0], last[1] - first[1])
