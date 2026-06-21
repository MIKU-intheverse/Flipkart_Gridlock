"""
Violation Classification & Confidence Scoring (project plan §4.4).

This is the module that applies ONE consistent threshold rule across all
seven violation types, rather than the ad-hoc single-example version that
motivated this rewrite. Every ViolationRecord, regardless of which checker
produced it, passes through here and comes out routed into exactly one
bucket.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import ViolationRecord

logger = get_logger(__name__)


@dataclass
class RoutedViolations:
    auto_approved: list[ViolationRecord] = field(default_factory=list)
    human_review: list[ViolationRecord] = field(default_factory=list)
    rejected_low_confidence: list[ViolationRecord] = field(default_factory=list)


class ConfidenceRouter:
    def __init__(self, app_config: AppConfig):
        cfg = app_config.violations["confidence_routing"]
        self._auto_approve_threshold = cfg["auto_approve_threshold"]
        self._review_queue_threshold = cfg["review_queue_threshold"]

    def route(self, records: list[ViolationRecord]) -> RoutedViolations:
        routed = RoutedViolations()
        for record in records:
            score = record.composite_confidence
            if score >= self._auto_approve_threshold:
                routed.auto_approved.append(record)
            elif score >= self._review_queue_threshold:
                routed.human_review.append(record)
            else:
                routed.rejected_low_confidence.append(record)

        if records:
            logger.debug(
                "Routed %d violations -> auto=%d, review=%d, rejected=%d",
                len(records), len(routed.auto_approved),
                len(routed.human_review), len(routed.rejected_low_confidence),
            )
        return routed
