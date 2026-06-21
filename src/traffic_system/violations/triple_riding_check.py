"""
Triple riding check — pure geometry, no model (project plan §4.3).

Counts how many `person` tracks have IoU overlap above threshold with a
given `motorcycle` track's bbox. Flags if the count exceeds the configured
max allowed riders.
"""

from __future__ import annotations

from datetime import datetime

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import TrackedObject, ViolationRecord

logger = get_logger(__name__)


class TripleRidingChecker:
    def __init__(self, app_config: AppConfig):
        cfg = app_config.violations["triple_riding"]
        self._enabled = cfg["enabled"]
        self._iou_threshold = cfg["iou_threshold"]
        self._max_allowed = cfg["max_allowed_riders"]

    def check(
        self,
        motorcycles: list[TrackedObject],
        persons: list[TrackedObject],
        camera_id: str,
        frame_id: int,
        timestamp: datetime,
    ) -> list[ViolationRecord]:
        if not self._enabled:
            return []

        results: list[ViolationRecord] = []

        for moto in motorcycles:
            rider_count = sum(
                1 for p in persons if p.bbox.iou(moto.bbox) > self._iou_threshold
            )
            if rider_count > self._max_allowed:
                results.append(ViolationRecord(
                    violation_type="triple_riding",
                    camera_id=camera_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    track_id=moto.track_id,
                    vehicle_class=moto.class_name,
                    bbox=moto.bbox,
                    detector_confidence=moto.detection_confidence,
                    classifier_confidence=1.0,   # no classifier involved
                    rule_certainty=0.95,          # geometry is near-certain but not absolute
                    extra={"rider_count": rider_count},
                ))

        return results
