"""
Stop-line and red-light violation checks.

Both depend on the same underlying event: did a tracked vehicle's reference
point cross the configured stop line THIS frame (not just "is past it"),
while the signal was red. The distinction between the two violation types
is which zone the vehicle ends up in after crossing:
  - crossed the stop line but stayed before the intersection -> stop_line_violation
  - crossed the stop line AND entered the deeper intersection zone -> red_light_violation
"""

from __future__ import annotations

from datetime import datetime

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import TrackedObject, ViolationRecord
from traffic_system.violations.geometry import CameraGeometry

logger = get_logger(__name__)


class StopLineRedLightChecker:
    def __init__(self, app_config: AppConfig):
        stop_cfg = app_config.violations["stop_line"]
        red_cfg = app_config.violations["red_light"]
        self._stop_line_enabled = stop_cfg["enabled"]
        self._red_light_enabled = red_cfg["enabled"]
        self._crossing_margin = stop_cfg["crossing_margin_px"]

        # Per-track previous "side of line" sign, needed to detect a
        # crossing event rather than a static "is past the line" state.
        self._previous_side: dict[int, float] = {}

    def check(
        self,
        vehicles: list[TrackedObject],
        geometry: CameraGeometry,
        signal_state: str,
        camera_id: str,
        frame_id: int,
        timestamp: datetime,
    ) -> list[ViolationRecord]:
        if not self._stop_line_enabled and not self._red_light_enabled:
            return []

        results: list[ViolationRecord] = []

        for vehicle in vehicles:
            reference_point = self._reference_point(vehicle)
            current_side = geometry.stop_line.signed_side(reference_point)
            previous_side = self._previous_side.get(vehicle.track_id)
            self._previous_side[vehicle.track_id] = current_side

            if previous_side is None:
                continue   # first time seeing this track — no crossing event possible yet

            crossed = self._has_crossed(previous_side, current_side)
            self._previous_side[vehicle.track_id] = current_side

            if not crossed:
                continue
            if signal_state != "red":
                continue   # crossing on green/yellow/unknown is not a violation here

            entered_intersection = geometry.red_light_zone.contains(reference_point)

            if entered_intersection and self._red_light_enabled:
                results.append(self._make_record(
                    "red_light_violation", vehicle, camera_id, frame_id, timestamp,
                    rule_certainty=0.95,
                ))
            elif self._stop_line_enabled:
                results.append(self._make_record(
                    "stop_line_violation", vehicle, camera_id, frame_id, timestamp,
                    rule_certainty=0.85,
                ))

        return results

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _reference_point(vehicle: TrackedObject) -> tuple[float, float]:
        # Bottom-center of the bbox approximates where the vehicle's front
        # contact point with the road is, which is the geometrically
        # meaningful point for a stop-line crossing check.
        x1, y1, x2, y2 = vehicle.bbox.as_tuple()
        return ((x1 + x2) / 2.0, y2)

    def _has_crossed(self, previous_side: float, current_side: float) -> bool:
        if abs(current_side) < self._crossing_margin:
            return False   # right on the line — ambiguous, wait for next frame
        return (previous_side > 0) != (current_side > 0)

    @staticmethod
    def _make_record(
        violation_type: str,
        vehicle: TrackedObject,
        camera_id: str,
        frame_id: int,
        timestamp: datetime,
        rule_certainty: float,
    ) -> ViolationRecord:
        return ViolationRecord(
            violation_type=violation_type,
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp=timestamp,
            track_id=vehicle.track_id,
            vehicle_class=vehicle.class_name,
            bbox=vehicle.bbox,
            detector_confidence=vehicle.detection_confidence,
            classifier_confidence=1.0,
            rule_certainty=rule_certainty,
            extra={},
        )
