"""
Stage 6: Evidence Generation (project plan §4.6).

For every routed violation, draws an annotated frame and builds the
structured metadata record (JSON-serializable dict) ready for Stage 7
storage. Violation-type -> color mapping is config-driven via a small
in-module default that can be overridden, not hardcoded inline in the
drawing calls.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import EvidencePackage, PlateResult, ViolationRecord

logger = get_logger(__name__)

_VIOLATION_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "helmet_noncompliance": (0, 0, 255),
    "seatbelt_noncompliance": (0, 140, 255),
    "seatbelt_indeterminate": (128, 128, 128),
    "triple_riding": (0, 220, 220),
    "wrong_side_driving": (130, 0, 130),
    "red_light_violation": (0, 0, 200),
    "stop_line_violation": (0, 100, 255),
    "illegal_parking": (0, 180, 90),
}
_DEFAULT_COLOR = (255, 255, 255)


class EvidenceGenerator:
    def __init__(self, app_config: AppConfig):
        cfg = app_config.evidence
        self._output_dir = app_config.resolve_path(cfg["output_dir"])
        self._jpeg_quality = cfg["jpeg_quality"]
        self._draw_confidence = cfg["draw_confidence_on_frame"]
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        frame: np.ndarray,
        violation: ViolationRecord,
        plate: PlateResult | None,
        gps_lat: float | None,
        gps_lon: float | None,
    ) -> EvidencePackage:
        violation_id = str(uuid.uuid4())
        annotated = self._annotate_frame(frame, violation, plate)
        image_path = self._save_image(annotated, violation_id)

        metadata = self._build_metadata(violation_id, violation, plate, gps_lat, gps_lon, str(image_path))

        return EvidencePackage(
            violation=violation,
            plate=plate,
            annotated_image_path=str(image_path),
            metadata_record=metadata,
        )

    # ------------------------------------------------------------------ internals

    def _annotate_frame(
        self, frame: np.ndarray, violation: ViolationRecord, plate: PlateResult | None
    ) -> np.ndarray:
        annotated = frame.copy()
        color = _VIOLATION_COLORS_BGR.get(violation.violation_type, _DEFAULT_COLOR)
        x1, y1, x2, y2 = violation.bbox.as_int_tuple()

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label_parts = [violation.violation_type.replace("_", " ").title()]
        if self._draw_confidence:
            label_parts.append(f"{violation.composite_confidence:.0%}")
        if plate is not None and plate.plate_text:
            label_parts.append(plate.plate_text)
        label = " | ".join(label_parts)

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y1 = max(0, y1 - text_h - 10)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        timestamp_text = violation.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            annotated, timestamp_text, (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )
        return annotated

    def _save_image(self, image: np.ndarray, violation_id: str) -> Path:
        filename = f"violation_{violation_id}.jpg"
        path = self._output_dir / filename
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        return path

    @staticmethod
    def _build_metadata(
        violation_id: str,
        violation: ViolationRecord,
        plate: PlateResult | None,
        gps_lat: float | None,
        gps_lon: float | None,
        image_path: str,
    ) -> dict:
        return {
            "violation_id": violation_id,
            "timestamp": violation.timestamp.isoformat(),
            "camera_id": violation.camera_id,
            "camera_gps_lat": gps_lat,
            "camera_gps_lon": gps_lon,
            "violation_type": violation.violation_type,
            "confidence": violation.composite_confidence,
            "detector_confidence": violation.detector_confidence,
            "classifier_confidence": violation.classifier_confidence,
            "rule_certainty": violation.rule_certainty,
            "track_id": violation.track_id,
            "vehicle_class": violation.vehicle_class,
            "bbox": list(violation.bbox.as_tuple()),
            "plate_text": plate.plate_text if plate else None,
            "plate_ocr_confidence": plate.ocr_confidence if plate else None,
            "plate_format_valid": plate.format_valid if plate else False,
            "plate_used_super_resolution": plate.used_super_resolution if plate else False,
            "plate_needs_review": plate.needs_manual_review if plate else True,
            "evidence_image_path": image_path,
            "extra": json.dumps(violation.extra),
            "reviewed": False,
            "review_decision": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
