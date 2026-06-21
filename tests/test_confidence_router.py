import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_system.utils.types import BBox, ViolationRecord
from traffic_system.violations.confidence_router import ConfidenceRouter


class _FakeAppConfig:
    """Minimal stand-in exposing only what ConfidenceRouter reads."""
    def __init__(self, auto_approve=0.9, review_queue=0.5):
        self.violations = {
            "confidence_routing": {
                "auto_approve_threshold": auto_approve,
                "review_queue_threshold": review_queue,
            }
        }


def _make_record(composite_confidence_components: tuple[float, float, float]) -> ViolationRecord:
    det, cls_conf, rule = composite_confidence_components
    return ViolationRecord(
        violation_type="test_violation",
        camera_id="cam_test",
        frame_id=0,
        timestamp=datetime.now(timezone.utc),
        track_id=1,
        vehicle_class="car",
        bbox=BBox(0, 0, 10, 10),
        detector_confidence=det,
        classifier_confidence=cls_conf,
        rule_certainty=rule,
    )


class TestConfidenceRouter:
    def setup_method(self):
        self.router = ConfidenceRouter(_FakeAppConfig())

    def test_high_confidence_auto_approved(self):
        record = _make_record((0.95, 1.0, 1.0))
        routed = self.router.route([record])
        assert len(routed.auto_approved) == 1
        assert len(routed.human_review) == 0
        assert len(routed.rejected_low_confidence) == 0

    def test_mid_confidence_goes_to_review(self):
        record = _make_record((0.6, 1.0, 1.0))
        routed = self.router.route([record])
        assert len(routed.auto_approved) == 0
        assert len(routed.human_review) == 1

    def test_low_confidence_rejected(self):
        record = _make_record((0.2, 1.0, 1.0))
        routed = self.router.route([record])
        assert len(routed.rejected_low_confidence) == 1

    def test_composite_confidence_is_multiplicative(self):
        # A single weak component should drag down the composite score
        # even if the others are perfect — this is what the multiply
        # (rather than average) design is meant to guarantee.
        record = _make_record((1.0, 1.0, 0.3))
        assert abs(record.composite_confidence - 0.3) < 1e-9
        routed = self.router.route([record])
        # 0.3 is below review_queue_threshold (0.5) -> rejected, not reviewed.
        # This demonstrates the multiplicative penalty actually has teeth:
        # a weak rule_certainty alone is enough to drop a record out of the
        # review band entirely, not just out of auto-approval.
        assert len(routed.rejected_low_confidence) == 1
        assert len(routed.human_review) == 0
        assert len(routed.auto_approved) == 0

    def test_empty_input_returns_empty_buckets(self):
        routed = self.router.route([])
        assert routed.auto_approved == []
        assert routed.human_review == []
        assert routed.rejected_low_confidence == []
