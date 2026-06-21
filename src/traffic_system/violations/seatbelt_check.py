"""
Seatbelt non-compliance check — rescoped per project plan §4.3.

Only runs on `car`-class tracks (windshield visibility from typical roadside
CCTV angles makes this unreliable for other vehicle types). If the
windshield crop is too small/low-resolution to trust, the result is marked
`indeterminate` rather than guessed, and routed to human review instead of
silently being dropped or silently being treated as compliant.
"""

from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np
import torch
from torchvision import transforms

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import BBox, TrackedObject, ViolationRecord
from traffic_system.violations.crop_classifier_model import CropBinaryClassifier

logger = get_logger(__name__)


def _load_classifier(weights_path: str, image_size: int, device: torch.device) -> CropBinaryClassifier:
    model = CropBinaryClassifier(input_size=image_size)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class SeatbeltChecker:
    def __init__(self, app_config: AppConfig, device: torch.device):
        cfg = app_config.violations["seatbelt"]
        self._enabled = cfg["enabled"]
        self._applies_to = set(cfg["applies_to_classes"])
        self._top_frac = cfg["windshield_crop_top_fraction"]
        self._bottom_frac = cfg["windshield_crop_bottom_fraction"]
        self._min_width = cfg["min_crop_width"]
        self._min_height = cfg["min_crop_height"]
        self._confidence_threshold = cfg["confidence_threshold"]
        self._device = device

        self._classifier = None
        if self._enabled:
            weights_path = str(app_config.resolve_path(cfg["classifier_weights"]))
            image_size = app_config.training["seatbelt_classifier"]["image_size"]
            self._classifier = _load_classifier(weights_path, image_size, device)
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def check(
        self,
        frame: np.ndarray,
        vehicles: list[TrackedObject],
        camera_id: str,
        frame_id: int,
        timestamp: datetime,
    ) -> list[ViolationRecord]:
        if not self._enabled or self._classifier is None:
            return []

        results: list[ViolationRecord] = []

        for vehicle in vehicles:
            if vehicle.class_name not in self._applies_to:
                continue  # rescoping rule: only run on classes this check is valid for

            crop = self._extract_windshield_crop(frame, vehicle.bbox)
            if crop is None:
                # Crop unusable (too small / bad angle) -> indeterminate,
                # not silently skipped and not silently guessed.
                results.append(self._indeterminate_record(
                    vehicle, camera_id, frame_id, timestamp,
                    reason="crop_below_minimum_resolution",
                ))
                continue

            prob_noncompliant = self._classify(crop)
            if prob_noncompliant >= self._confidence_threshold:
                results.append(ViolationRecord(
                    violation_type="seatbelt_noncompliance",
                    camera_id=camera_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    track_id=vehicle.track_id,
                    vehicle_class=vehicle.class_name,
                    bbox=vehicle.bbox,
                    detector_confidence=vehicle.detection_confidence,
                    classifier_confidence=float(prob_noncompliant),
                    rule_certainty=1.0,
                    extra={},
                ))

        return results

    # ------------------------------------------------------------------ internals

    def _extract_windshield_crop(self, frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox.as_int_tuple()
        height = y2 - y1
        crop_y1 = y1 + int(height * self._top_frac)
        crop_y2 = y1 + int(height * self._bottom_frac)

        x1, crop_y1 = max(0, x1), max(0, crop_y1)
        x2 = min(frame.shape[1], x2)
        crop_y2 = min(frame.shape[0], crop_y2)

        if (x2 - x1) < self._min_width or (crop_y2 - crop_y1) < self._min_height:
            return None

        return frame[crop_y1:crop_y2, x1:x2]

    def _classify(self, crop: np.ndarray) -> float:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        prob = self._classifier.predict_proba(tensor)
        return float(prob.item())

    @staticmethod
    def _indeterminate_record(
        vehicle: TrackedObject, camera_id: str, frame_id: int, timestamp: datetime, reason: str
    ) -> ViolationRecord:
        return ViolationRecord(
            violation_type="seatbelt_indeterminate",
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp=timestamp,
            track_id=vehicle.track_id,
            vehicle_class=vehicle.class_name,
            bbox=vehicle.bbox,
            detector_confidence=vehicle.detection_confidence,
            classifier_confidence=0.0,
            rule_certainty=0.0,   # forces this into human review via the confidence gate
            extra={"reason": reason},
        )
