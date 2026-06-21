"""
Per-camera geometry loaded from a zone config JSON file (see
config/zones/cam_01_zones.json for the schema). This is the data that gets
"drawn once per camera" during site setup — never hardcoded in Python.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StopLine:
    point_a: tuple[float, float]
    point_b: tuple[float, float]

    def signed_side(self, point: tuple[float, float]) -> float:
        """
        Returns a signed value: which side of the line `point` is on.
        Sign flips exactly when a point crosses from one side to the other,
        which is what the stop-line/red-light checks use to detect crossing.
        """
        ax, ay = self.point_a
        bx, by = self.point_b
        px, py = point
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


@dataclass
class Polygon:
    points: list[tuple[float, float]]

    def contains(self, point: tuple[float, float]) -> bool:
        """Standard ray-casting point-in-polygon test."""
        x, y = point
        n = len(self.points)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.points[i]
            xj, yj = self.points[j]
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            ):
                inside = not inside
            j = i
        return inside


@dataclass
class CameraGeometry:
    camera_id: str
    frame_width: int
    frame_height: int
    lane_direction_vector: tuple[float, float]
    stop_line: StopLine
    red_light_zone: Polygon
    no_parking_zones: list[Polygon]
    traffic_light_roi: tuple[int, int, int, int]   # x1, y1, x2, y2

    @classmethod
    def from_dict(cls, d: dict) -> "CameraGeometry":
        tl_roi = d["traffic_light_roi"]
        return cls(
            camera_id=d["camera_id"],
            frame_width=d["frame_width"],
            frame_height=d["frame_height"],
            lane_direction_vector=tuple(d["lane_direction_vector"]),
            stop_line=StopLine(
                point_a=tuple(d["stop_line"]["point_a"]),
                point_b=tuple(d["stop_line"]["point_b"]),
            ),
            red_light_zone=Polygon(points=[tuple(p) for p in d["red_light_zone_polygon"]]),
            no_parking_zones=[
                Polygon(points=[tuple(p) for p in z["polygon"]])
                for z in d.get("no_parking_zones", [])
            ],
            traffic_light_roi=(
                tl_roi["x1"], tl_roi["y1"], tl_roi["x2"], tl_roi["y2"]
            ),
        )
