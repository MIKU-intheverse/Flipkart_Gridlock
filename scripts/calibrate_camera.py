#!/usr/bin/env python3
"""
Interactive camera calibration tool — produces a zone config JSON file
(the per-camera geometry consumed by violations/geometry.py) by letting an
operator click points on a representative frame from the camera.

Usage:
    python scripts/calibrate_camera.py --source data/sample_videos/intersection_01.mp4 \
        --camera-id cam_01 --output config/zones/cam_01_zones.json

Controls:
    1) Click 2 points to define the stop line, press 's'
    2) Click 4+ points to define the red-light/intersection zone, press 'r'
    3) Click 4+ points to define a no-parking zone, press 'p' (repeatable)
    4) Click 2 points on the traffic light bounding box (top-left, bottom-right), press 't'
    5) Click 2 points to define the lane direction (from, to), press 'l'
    6) Press 'q' to save and quit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class CalibrationSession:
    def __init__(self, frame: np.ndarray, camera_id: str):
        self.frame = frame
        self.camera_id = camera_id
        self.display = frame.copy()
        self.clicked_points: list[tuple[int, int]] = []

        self.stop_line: list[list[int]] | None = None
        self.red_light_zone: list[list[int]] | None = None
        self.no_parking_zones: list[list[list[int]]] = []
        self.traffic_light_roi: dict | None = None
        self.lane_direction_vector: list[float] | None = None

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
            cv2.circle(self.display, (x, y), 4, (0, 0, 255), -1)

    def reset_clicks(self) -> list[tuple[int, int]]:
        points, self.clicked_points = self.clicked_points, []
        return points

    def run(self) -> None:
        window = "Calibration - see script docstring for controls"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self.on_mouse)

        print(__doc__)

        while True:
            cv2.imshow(window, self.display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("s"):
                pts = self.reset_clicks()
                if len(pts) >= 2:
                    self.stop_line = [list(pts[0]), list(pts[1])]
                    cv2.line(self.display, pts[0], pts[1], (255, 0, 0), 2)
                    print(f"Stop line set: {self.stop_line}")
                else:
                    print("Need at least 2 points for the stop line.")

            elif key == ord("r"):
                pts = self.reset_clicks()
                if len(pts) >= 3:
                    self.red_light_zone = [list(p) for p in pts]
                    cv2.polylines(self.display, [np.array(pts)], True, (0, 255, 0), 2)
                    print(f"Red-light zone set: {self.red_light_zone}")
                else:
                    print("Need at least 3 points for a polygon.")

            elif key == ord("p"):
                pts = self.reset_clicks()
                if len(pts) >= 3:
                    zone = [list(p) for p in pts]
                    self.no_parking_zones.append(zone)
                    cv2.polylines(self.display, [np.array(pts)], True, (0, 165, 255), 2)
                    print(f"No-parking zone #{len(self.no_parking_zones)} added: {zone}")
                else:
                    print("Need at least 3 points for a polygon.")

            elif key == ord("t"):
                pts = self.reset_clicks()
                if len(pts) >= 2:
                    x1, y1 = pts[0]
                    x2, y2 = pts[1]
                    self.traffic_light_roi = {
                        "x1": min(x1, x2), "y1": min(y1, y2),
                        "x2": max(x1, x2), "y2": max(y1, y2),
                    }
                    cv2.rectangle(self.display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    print(f"Traffic light ROI set: {self.traffic_light_roi}")
                else:
                    print("Need exactly 2 points (top-left, bottom-right).")

            elif key == ord("l"):
                pts = self.reset_clicks()
                if len(pts) >= 2:
                    (fx, fy), (tx, ty) = pts[0], pts[1]
                    vec = np.array([tx - fx, ty - fy], dtype=np.float64)
                    norm = np.linalg.norm(vec)
                    vec = vec / norm if norm > 0 else vec
                    self.lane_direction_vector = [round(float(v), 4) for v in vec]
                    cv2.arrowedLine(self.display, (fx, fy), (tx, ty), (255, 0, 255), 2)
                    print(f"Lane direction vector set: {self.lane_direction_vector}")
                else:
                    print("Need exactly 2 points (from, to).")

            elif key == ord("q"):
                break

        cv2.destroyAllWindows()

    def to_zone_dict(self) -> dict:
        missing = []
        if self.stop_line is None:
            missing.append("stop_line (key 's')")
        if self.red_light_zone is None:
            missing.append("red_light_zone (key 'r')")
        if self.traffic_light_roi is None:
            missing.append("traffic_light_roi (key 't')")
        if self.lane_direction_vector is None:
            missing.append("lane_direction_vector (key 'l')")
        if missing:
            raise ValueError(
                "Calibration incomplete — missing: " + ", ".join(missing) +
                ". Re-run and define all required zones before quitting."
            )

        h, w = self.frame.shape[:2]
        return {
            "camera_id": self.camera_id,
            "frame_width": w,
            "frame_height": h,
            "lane_direction_vector": self.lane_direction_vector,
            "stop_line": {"point_a": self.stop_line[0], "point_b": self.stop_line[1]},
            "red_light_zone_polygon": self.red_light_zone,
            "no_parking_zones": [
                {"zone_id": f"np_zone_{i+1}", "polygon": zone}
                for i, zone in enumerate(self.no_parking_zones)
            ],
            "traffic_light_roi": self.traffic_light_roi,
        }


def grab_representative_frame(source: str) -> np.ndarray:
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source '{source}'")
    # Skip a few frames in so we're not calibrating against a black/transition frame.
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read a frame from source '{source}'")
    cap.release()
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive per-camera zone calibration.")
    parser.add_argument("--source", required=True, help="Video file, RTSP URL, or webcam index")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output", required=True, help="Path to write the zone config JSON")
    args = parser.parse_args()

    frame = grab_representative_frame(args.source)
    session = CalibrationSession(frame, args.camera_id)
    session.run()

    try:
        zone_dict = session.to_zone_dict()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(zone_dict, f, indent=2)

    print(f"Saved zone configuration to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
