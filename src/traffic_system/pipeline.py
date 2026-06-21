"""
The main pipeline orchestrator — wires Stages 1 through 7 together for a
single camera. One CameraPipeline instance is created per configured
source; `run_pipeline.py` drives one or more of these.

This is intentionally the only file that knows the full stage order. Every
individual stage module only knows its own input/output contract (defined
in utils/types.py), which is what keeps the system "flowing smoothly"
without any stage needing to know about any other stage's internals.
"""

from __future__ import annotations

import torch

from traffic_system.utils.config import AppConfig, SourceConfig, load_zone_config
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import FrameContext
from traffic_system.utils.video_source import VideoSource, RawFrame
from traffic_system.preprocessing.preprocessor import Preprocessor
from traffic_system.detection.detector import VehicleDetector
from traffic_system.tracking.tracker import ByteTrackTracker
from traffic_system.violations.violation_engine import ViolationEngine
from traffic_system.violations.geometry import CameraGeometry
from traffic_system.violations.confidence_router import RoutedViolations
from traffic_system.ocr.lpr_pipeline import LicensePlateRecognizer
from traffic_system.evidence.evidence_generator import EvidenceGenerator
from traffic_system.storage.repository import ViolationRepository

logger = get_logger(__name__)


def resolve_device(preference: str) -> torch.device:
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


class CameraPipeline:
    """
    Holds every per-camera stateful component (tracker, geometry, dwell
    timers inside the violation engine) and exposes a single
    `process_one_frame()` call that a runner script invokes per frame.
    """

    def __init__(
        self,
        app_config: AppConfig,
        source_config: SourceConfig,
        detector: VehicleDetector,
        repository: ViolationRepository,
        device: torch.device,
    ):
        self._app_config = app_config
        self._source_config = source_config
        self._camera_id = source_config.camera_id

        self._preprocessor = Preprocessor(app_config)
        self._detector = detector              # shared across cameras — stateless, safe to share
        self._tracker = ByteTrackTracker(app_config)   # NOT shared — per-camera track state
        self._violation_engine = ViolationEngine(app_config, device)
        self._lpr = LicensePlateRecognizer(app_config, device)
        self._evidence_generator = EvidenceGenerator(app_config)
        self._repository = repository

        zone_dict = load_zone_config(app_config, source_config.zone_config_path)
        self._geometry = CameraGeometry.from_dict(zone_dict)

    def process_one_frame(self, raw_frame: RawFrame) -> RoutedViolations:
        enhanced_image, quality_flags = self._preprocessor.process(
            raw_frame.image, self._camera_id
        )

        frame_ctx = FrameContext(
            camera_id=self._camera_id,
            frame_id=raw_frame.frame_id,
            timestamp=raw_frame.timestamp,
            image=enhanced_image,
            raw_image=raw_frame.image,
            quality_flags=quality_flags,
        )

        detections = self._detector.detect(frame_ctx.image)
        tracked_objects = self._tracker.update(detections)

        routed = self._violation_engine.process_frame(
            frame=frame_ctx.image,
            tracked_objects=tracked_objects,
            geometry=self._geometry,
            camera_id=self._camera_id,
            frame_id=frame_ctx.frame_id,
            timestamp=frame_ctx.timestamp,
        )

        # Only auto-approved and human-review violations get OCR + evidence;
        # rejected_low_confidence violations are dropped here, matching the
        # confidence-gate design — see violations/confidence_router.py.
        for violation in (routed.auto_approved + routed.human_review):
            plate_result = self._lpr.recognize(frame_ctx.raw_image, violation.bbox)
            evidence = self._evidence_generator.generate(
                frame=frame_ctx.raw_image,
                violation=violation,
                plate=plate_result,
                gps_lat=self._source_config.gps_lat,
                gps_lon=self._source_config.gps_lon,
            )
            self._repository.insert(evidence.metadata_record)

        if routed.auto_approved or routed.human_review:
            logger.info(
                "[%s] frame=%d auto_approved=%d human_review=%d rejected=%d",
                self._camera_id, frame_ctx.frame_id,
                len(routed.auto_approved), len(routed.human_review),
                len(routed.rejected_low_confidence),
            )

        return routed

    def set_manual_signal_state(self, state: str) -> None:
        self._violation_engine.set_manual_signal_state(self._camera_id, state)

    def shutdown(self) -> None:
        """Releases the LPR worker process cleanly. Optional to call —
        the worker is a daemon process and is reaped automatically if the
        parent exits without calling this — but calling it avoids leaving
        a lingering process around during long-running multi-camera use."""
        self._lpr.shutdown()
