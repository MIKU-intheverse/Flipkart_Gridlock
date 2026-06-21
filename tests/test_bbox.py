import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_system.utils.types import BBox


class TestBBox:
    def test_iou_identical_boxes_is_one(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(0, 0, 10, 10)
        assert abs(a.iou(b) - 1.0) < 1e-9

    def test_iou_disjoint_boxes_is_zero(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(100, 100, 110, 110)
        assert a.iou(b) == 0.0

    def test_iou_half_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 0, 15, 10)
        # intersection area = 5*10=50, union = 100+100-50=150 -> iou = 1/3
        assert abs(a.iou(b) - (1 / 3)) < 1e-6

    def test_centroid(self):
        box = BBox(0, 0, 10, 20)
        assert box.centroid == (5.0, 10.0)

    def test_area(self):
        box = BBox(0, 0, 10, 20)
        assert box.area == 200.0

    def test_negative_coords_clip_to_zero_dims(self):
        # x2 < x1 should not produce a negative width/height/area
        box = BBox(10, 10, 5, 5)
        assert box.width == 0.0
        assert box.height == 0.0
        assert box.area == 0.0
