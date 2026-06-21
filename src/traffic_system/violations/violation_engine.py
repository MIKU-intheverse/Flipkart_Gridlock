"""
Violation Engine — orchestrates all seven checks (Stage 4 in full).

Splits tracked objects by class once per frame, then calls each enabled
checker with exactly the subset of objects relevant to it. This is the
single place that knows how all seven checks fit together; individual
checker modules know nothing about each other.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import torch

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import TrackedObject, ViolationRecord
from traffic_system.violations.geometry import CameraGeometry
from traffic_system.violations.helmet_check import HelmetChecker
from traffic_system.violations.seatbelt_check import SeatbeltChecker
from traffic_system.violations.triple_riding_check import TripleRidingChecker
from traffic_system.violations.wrong_side_check import WrongSideChecker
from traffic_system.violations.stop_line_red_light_check import StopLineRedLightChecker
from traffic_system.violations.illegal_parking_check import IllegalParkingChecker
from traffic_system.violations.signal_state import SignalStateProvider
from traffic_system.violations.confidence_router import ConfidenceRouter, RoutedViolations

logger = get_logger(__name__)

# Class names treated as "two-wheeler" for the rider-association checks.
# Sourced from config so a retrained model with different class names
# doesn't require touching this file.
_TWO_WHEELER_CLASS = "motorcycle"
_PERSON_CLASS = "person"


class ViolationEngine:
    def __init__(self, app_config: AppConfig, device: torch.device):
        self._person_class = app_config.detection.get("person_class", _PERSON_CLASS)
        self._vehicle_classes = set(app_config.detection["vehicle_classes"])

        self._helmet_checker = HelmetChecker(app_config, device)
        self._seatbelt_checker = SeatbeltChecker(app_config, device)
        self._triple_riding_checker = TripleRidingChecker(app_config)
        self._wrong_side_checker = WrongSideChecker(app_config)
        self._stop_red_checker = StopLineRedLightChecker(app_config)
        self._illegal_parking_checker = IllegalParkingChecker(app_config)
        self._signal_provider = SignalStateProvider(app_config)
        self._router = ConfidenceRouter(app_config)

    def set_manual_signal_state(self, camera_id: str, state: str) -> None:
        """Pass-through so callers (e.g. a test harness or a signal-controller
        integration) can push live signal state without reaching into internals."""
        self._signal_provider.set_manual_state(camera_id, state)

    def process_frame(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        geometry: CameraGeometry,
        camera_id: str,
        frame_id: int,
        timestamp: datetime,
    ) -> RoutedViolations:
        motorcycles = [t for t in tracked_objects if t.class_name == _TWO_WHEELER_CLASS]
        persons = [t for t in tracked_objects if t.class_name == self._person_class]
        vehicles = [t for t in tracked_objects if t.class_name in self._vehicle_classes]

        all_records: list[ViolationRecord] = []

        all_records += self._helmet_checker.check(
            frame, persons, motorcycles, camera_id, frame_id, timestamp
        )
        all_records += self._seatbelt_checker.check(
            frame, vehicles, camera_id, frame_id, timestamp
        )
        all_records += self._triple_riding_checker.check(
            motorcycles, persons, camera_id, frame_id, timestamp
        )
        all_records += self._wrong_side_checker.check(
            vehicles, geometry, camera_id, frame_id, timestamp
        )

        signal_state = self._signal_provider.get_state(camera_id, frame, geometry)
        all_records += self._stop_red_checker.check(
            vehicles, geometry, signal_state, camera_id, frame_id, timestamp
        )
        all_records += self._illegal_parking_checker.check(
            vehicles, geometry, camera_id, frame_id, timestamp
        )

        return self._router.route(all_records)
