"""
Shared data types passed between pipeline stages.

Every stage (preprocessing -> detection -> tracking -> violations -> ocr ->
evidence -> storage) consumes and/or produces these types. Keeping them in
one place means a module never has to guess the shape of another module's
output — it imports the type and gets autocomplete + type-checking for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np


@dataclass
class BBox:
    """Axis-aligned bounding box in absolute pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def iou(self, other: "BBox") -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        union = self.area + other.area - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union


@dataclass
class Detection:
    """A single object detection from the perception model, pre-tracking."""
    class_name: str
    confidence: float
    bbox: BBox


@dataclass
class FrameContext:
    """
    Everything about one processed frame that downstream stages need.
    Produced by the preprocessing stage, enriched by every later stage.
    """
    camera_id: str
    frame_id: int
    timestamp: datetime
    image: np.ndarray                      # the (possibly enhanced) BGR frame
    raw_image: np.ndarray                  # untouched original, kept for evidence frames
    quality_flags: dict = field(default_factory=dict)   # e.g. {"was_blurry": True}


@dataclass
class TrackedObject:
    """A detection with an identity that persists across frames."""
    track_id: int
    class_name: str
    bbox: BBox
    detection_confidence: float
    velocity: tuple[float, float] = (0.0, 0.0)   # px/frame, set by the tracker
    frames_since_seen: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)  # centroid history


@dataclass
class ViolationRecord:
    """
    The uniform output of every violation-detection module (Stage 4). All
    seven violation checks emit this same shape so Stage 5/6/7 never need
    to know which specific rule or classifier produced it.
    """
    violation_type: str
    camera_id: str
    frame_id: int
    timestamp: datetime
    track_id: int
    vehicle_class: str
    bbox: BBox
    detector_confidence: float                 # confidence from Stage 2 detection
    classifier_confidence: float = 1.0         # from a learned classifier, if used
    rule_certainty: float = 1.0                # from geometric/temporal logic, if used
    extra: dict = field(default_factory=dict)  # violation-specific details (e.g. rider_count)

    @property
    def composite_confidence(self) -> float:
        return self.detector_confidence * self.classifier_confidence * self.rule_certainty


@dataclass
class PlateResult:
    plate_text: Optional[str]
    raw_ocr_text: Optional[str]
    ocr_confidence: float
    format_valid: bool
    used_super_resolution: bool
    needs_manual_review: bool


@dataclass
class EvidencePackage:
    violation: ViolationRecord
    plate: Optional[PlateResult]
    annotated_image_path: str
    metadata_record: dict
