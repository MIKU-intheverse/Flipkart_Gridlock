"""
Illegal parking check (rescoped per project plan §4.3): polygon containment
+ dwell-time timer using existing tracker IDs, no camera calibration or
perspective correction.
"""

from __future__ import annotations

import math
from datetime import datetime

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import TrackedObject, ViolationRecord
from traffic_system.violations.geometry import CameraGeometry

logger = get_logger(__name__)


class _DwellState:
    __slots__ = ("zone_entry_time", "reference_centroid", "already_flagged")

    def __init__(self, entry_time: datetime, centroid: tuple[float, float]):
        self.zone_entry_time = entry_time
        self.reference_centroid = centroid
        self.already_flagged = False


class IllegalParkingChecker:
    def __init__(self, app_config: AppConfig):
        cfg = app_config.violations["illegal_parking"]
        self._enabled = cfg["enabled"]
        self._dwell_threshold_seconds = cfg["dwell_threshold_seconds"]
        self._drift_tolerance_px = cfg["centroid_drift_tolerance_px"]

        # Per-track dwell state, keyed by track_id. Reset whenever a track
        # leaves the zone or drifts more than the tolerance (i.e. it's
        # actually moving, not parked).
        self._dwell_states: dict[int, _DwellState] = {}

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
        active_track_ids = {v.track_id for v in vehicles}

        # Forget dwell state for tracks that disappeared (left frame / track lost).
        for stale_id in list(self._dwell_states.keys()):
            if stale_id not in active_track_ids:
                del self._dwell_states[stale_id]

        for vehicle in vehicles:
            centroid = vehicle.bbox.centroid
            in_any_zone = any(zone.contains(centroid) for zone in geometry.no_parking_zones)

            if not in_any_zone:
                self._dwell_states.pop(vehicle.track_id, None)
                continue

            state = self._dwell_states.get(vehicle.track_id)
            if state is None:
                self._dwell_states[vehicle.track_id] = _DwellState(timestamp, centroid)
                continue

            drift = math.hypot(
                centroid[0] - state.reference_centroid[0],
                centroid[1] - state.reference_centroid[1],
            )
            if drift > self._drift_tolerance_px:
                # Vehicle actually moved within the zone — restart the timer,
                # it isn't parked, it's just passing through or repositioning.
                self._dwell_states[vehicle.track_id] = _DwellState(timestamp, centroid)
                continue

            dwell_seconds = (timestamp - state.zone_entry_time).total_seconds()
            if dwell_seconds >= self._dwell_threshold_seconds and not state.already_flagged:
                state.already_flagged = True
                results.append(ViolationRecord(
                    violation_type="illegal_parking",
                    camera_id=camera_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    track_id=vehicle.track_id,
                    vehicle_class=vehicle.class_name,
                    bbox=vehicle.bbox,
                    detector_confidence=vehicle.detection_confidence,
                    classifier_confidence=1.0,
                    rule_certainty=0.9,
                    extra={"dwell_seconds": dwell_seconds},
                ))

        return results
