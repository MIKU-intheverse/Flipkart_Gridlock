import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from traffic_system.violations.geometry import StopLine, Polygon, CameraGeometry


class TestStopLine:
    def test_signed_side_opposite_signs_across_line(self):
        line = StopLine(point_a=(0, 100), point_b=(200, 100))
        above = line.signed_side((100, 50))
        below = line.signed_side((100, 150))
        assert (above > 0) != (below > 0)

    def test_point_exactly_on_line_is_near_zero(self):
        line = StopLine(point_a=(0, 100), point_b=(200, 100))
        on_line = line.signed_side((100, 100))
        assert abs(on_line) < 1e-6


class TestPolygon:
    def test_point_inside_square(self):
        square = Polygon(points=[(0, 0), (10, 0), (10, 10), (0, 10)])
        assert square.contains((5, 5)) is True

    def test_point_outside_square(self):
        square = Polygon(points=[(0, 0), (10, 0), (10, 10), (0, 10)])
        assert square.contains((50, 50)) is False

    def test_point_outside_near_edge(self):
        square = Polygon(points=[(0, 0), (10, 0), (10, 10), (0, 10)])
        assert square.contains((10.5, 5)) is False


class TestCameraGeometryFromDict:
    def test_round_trip(self):
        d = {
            "camera_id": "cam_test",
            "frame_width": 1920,
            "frame_height": 1080,
            "lane_direction_vector": [0.0, -1.0],
            "stop_line": {"point_a": [0, 500], "point_b": [1000, 500]},
            "red_light_zone_polygon": [[0, 400], [1000, 400], [1000, 500], [0, 500]],
            "no_parking_zones": [
                {"zone_id": "z1", "polygon": [[0, 0], [50, 0], [50, 50], [0, 50]]}
            ],
            "traffic_light_roi": {"x1": 10, "y1": 10, "x2": 30, "y2": 60},
        }
        geometry = CameraGeometry.from_dict(d)
        assert geometry.camera_id == "cam_test"
        assert geometry.lane_direction_vector == (0.0, -1.0)
        assert len(geometry.no_parking_zones) == 1
        assert geometry.traffic_light_roi == (10, 10, 30, 60)
