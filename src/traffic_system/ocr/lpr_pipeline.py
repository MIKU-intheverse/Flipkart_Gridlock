"""
Stage 5: License Plate Recognition — full pipeline.

Ties together plate localization (on a vehicle crop), conditional
super-resolution, OCR, and format validation/correction into the single
`recognize()` call the main pipeline uses. A failed read at any step never
raises — it always returns a PlateResult with needs_manual_review=True so
the violation evidence still proceeds to Stage 6 (project plan §4.5/§4.6).

If the OCR engine itself fails to initialize (e.g. model weights can't be
downloaded because of a network restriction), LPR degrades to
"always needs manual review" rather than taking down the entire pipeline —
losing plate recognition should never mean losing violation detection.
"""

from __future__ import annotations

import numpy as np
import torch

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import BBox, PlateResult
from traffic_system.detection.plate_detector import PlateDetector
from traffic_system.detection.detector import ModelLoadError
from traffic_system.ocr.super_resolution import PlateSuperResolver
from traffic_system.ocr.ocr_engine import PlateOcrEngine, OcrEngineLoadError
from traffic_system.ocr.plate_format import get_validator

logger = get_logger(__name__)


class LicensePlateRecognizer:
    def __init__(self, app_config: AppConfig, device: torch.device):
        cfg = app_config.lpr
        self._enabled = cfg["enabled"]
        self._min_plate_width = cfg["min_plate_width_px"]
        self._confidence_threshold_for_accept = 0.75  # exact-match acceptance bar
        self._degraded = False   # True if OCR/plate models failed to load

        if self._enabled:
            try:
                self._plate_detector = PlateDetector(app_config)
                self._super_resolver = PlateSuperResolver(app_config, device)
                self._ocr_engine = PlateOcrEngine(app_config)
                self._validator = get_validator(cfg["format_region"])
            except (OcrEngineLoadError, ModelLoadError) as e:
                logger.error(
                    "LPR failed to initialize (%s) — plate recognition will be "
                    "disabled for this run; violations will still be detected "
                    "and stored, just without an OCR-read plate number. Fix the "
                    "underlying issue (e.g. model download/network access, or a "
                    "missing/incorrect plate_detector_weights path) and restart "
                    "to re-enable plate recognition.", e,
                )
                self._degraded = True

    def recognize(self, frame: np.ndarray, vehicle_bbox: BBox) -> PlateResult:
        if not self._enabled or self._degraded:
            return self._undetected_result()

        vehicle_crop = self._safe_crop(frame, vehicle_bbox)
        if vehicle_crop is None:
            return self._undetected_result()

        plate_bbox_local = self._plate_detector.locate_plate(vehicle_crop)
        if plate_bbox_local is None:
            return self._undetected_result()

        plate_crop = self._safe_crop(vehicle_crop, plate_bbox_local)
        if plate_crop is None:
            return self._undetected_result()

        used_sr = False
        if plate_crop.shape[1] < self._min_plate_width and self._super_resolver.enabled:
            plate_crop = self._super_resolver.upscale(plate_crop)
            used_sr = True

        raw_text, ocr_confidence = self._ocr_engine.read_text(plate_crop)
        if raw_text is None:
            return PlateResult(
                plate_text=None, raw_ocr_text=None, ocr_confidence=0.0,
                format_valid=False, used_super_resolution=used_sr, needs_manual_review=True,
            )

        validation = self._validator.validate(raw_text)
        accepted = validation.format_valid and ocr_confidence >= self._confidence_threshold_for_accept

        return PlateResult(
            plate_text=validation.corrected_text if validation.format_valid else raw_text,
            raw_ocr_text=raw_text,
            ocr_confidence=ocr_confidence,
            format_valid=validation.format_valid,
            used_super_resolution=used_sr,
            needs_manual_review=not accepted,
        )

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _safe_crop(image: np.ndarray, bbox: BBox) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox.as_int_tuple()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    @staticmethod
    def _undetected_result() -> PlateResult:
        return PlateResult(
            plate_text=None, raw_ocr_text=None, ocr_confidence=0.0,
            format_valid=False, used_super_resolution=False, needs_manual_review=True,
        )

    def shutdown(self) -> None:
        """Releases the OCR worker process, if one was successfully started.
        Safe to call even when LPR is disabled or failed to initialize."""
        if self._enabled and not self._degraded:
            self._ocr_engine.shutdown()
