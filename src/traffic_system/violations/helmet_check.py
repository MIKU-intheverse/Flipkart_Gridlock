"""
Helmet non-compliance check.

Approach (per project plan §4.3): associate each `person` track with the
`motorcycle` track it's riding via IoU overlap, crop the head region of the
rider's bbox, and run it through a small trained classifier. No violation
is raised for persons not associated with a motorcycle (e.g. pedestrians).
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


class HelmetChecker:
    def __init__(self, app_config: AppConfig, device: torch.device):
        cfg = app_config.violations["helmet"]
        self._enabled = cfg["enabled"]
        self._head_fraction = cfg["head_crop_fraction"]
        self._confidence_threshold = cfg["confidence_threshold"]
        self._min_crop_size = cfg["min_crop_size"]
        self._device = device

        self._classifier = None
        if self._enabled:
            weights_path = str(app_config.resolve_path(cfg["classifier_weights"]))
            # image_size must match what the classifier was trained with —
            # read from training config so the two never drift apart.
            image_size = app_config.training["helmet_classifier"]["image_size"]
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
        riders: list[TrackedObject],
        motorcycles: list[TrackedObject],
        camera_id: str,
        frame_id: int,
        timestamp: datetime,
    ) -> list[ViolationRecord]:
        if not self._enabled or self._classifier is None:
            return []

        results: list[ViolationRecord] = []

        for rider in riders:
            associated_moto = self._find_associated_motorcycle(rider, motorcycles)
            if associated_moto is None:
                continue  # not a motorcycle rider — e.g. a pedestrian, skip

            head_crop = self._extract_head_crop(frame, rider.bbox)
            if head_crop is None:
                continue

            prob_noncompliant = self._classify(head_crop)
            if prob_noncompliant >= self._confidence_threshold:
                results.append(ViolationRecord(
                    violation_type="helmet_noncompliance",
                    camera_id=camera_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    track_id=rider.track_id,
                    vehicle_class=associated_moto.class_name,
                    bbox=rider.bbox,
                    detector_confidence=rider.detection_confidence,
                    classifier_confidence=float(prob_noncompliant),
                    rule_certainty=1.0,
                    extra={"associated_motorcycle_track_id": associated_moto.track_id},
                ))

        return results

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _find_associated_motorcycle(
        rider: TrackedObject, motorcycles: list[TrackedObject]
    ) -> TrackedObject | None:
        best_moto, best_iou = None, 0.0
        for moto in motorcycles:
            iou = rider.bbox.iou(moto.bbox)
            if iou > best_iou:
                best_iou, best_moto = iou, moto
        # Require *some* overlap to count as "riding" this motorcycle —
        # a small nonzero threshold filters out coincidental nearby boxes.
        return best_moto if best_iou > 0.05 else None

    def _extract_head_crop(self, frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox.as_int_tuple()
        head_height = int((y2 - y1) * self._head_fraction)
        head_y2 = y1 + max(head_height, 1)

        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        head_y2 = min(frame.shape[0], head_y2)

        if x2 - x1 < self._min_crop_size or head_y2 - y1 < self._min_crop_size:
            return None

        return frame[y1:head_y2, x1:x2]

    def _classify(self, crop: np.ndarray) -> float:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        prob = self._classifier.predict_proba(tensor)
        return float(prob.item())
