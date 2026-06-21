"""
Kalman-filter-based single-object track state, used internally by the
ByteTrack-style tracker. Models each tracked object's bounding box as
(cx, cy, aspect_ratio, height) plus their velocities — the standard
formulation used in SORT/DeepSORT/ByteTrack-style trackers.
"""

from __future__ import annotations

import numpy as np

from traffic_system.utils.types import BBox


def _bbox_to_state(bbox: BBox) -> np.ndarray:
    cx, cy = bbox.centroid
    h = bbox.height
    a = bbox.width / h if h > 0 else 1.0
    return np.array([cx, cy, a, h], dtype=np.float64)


def _state_to_bbox(state: np.ndarray) -> BBox:
    cx, cy, a, h = state[:4]
    h = max(h, 1e-3)
    w = a * h
    return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class KalmanBoxTracker:
    """
    A constant-velocity Kalman filter over an 8-dim state:
    [cx, cy, aspect_ratio, height, vcx, vcy, va, vh]
    """

    _next_id = 1

    def __init__(self, bbox: BBox, class_name: str, detection_confidence: float):
        self.track_id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self.class_name = class_name
        self.detection_confidence = detection_confidence
        self.frames_since_seen = 0
        self.hit_streak = 0
        self.age = 0
        self.history: list[tuple[float, float]] = []

        dt = 1.0
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = dt

        self._H = np.zeros((4, 8))
        self._H[:4, :4] = np.eye(4)

        self._Q = np.eye(8) * 1.0
        self._Q[4:, 4:] *= 0.01    # velocity components assumed to change slowly
        self._R = np.eye(4) * 10.0
        self._P = np.eye(8) * 10.0

        self.state = np.zeros(8)
        self.state[:4] = _bbox_to_state(bbox)
        self.history.append(bbox.centroid)

    def predict(self) -> BBox:
        self.state = self._F @ self.state
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.age += 1
        self.frames_since_seen += 1
        return _state_to_bbox(self.state)

    def update(self, bbox: BBox, detection_confidence: float) -> None:
        z = _bbox_to_state(bbox)
        y = z - self._H @ self.state
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self._P = (np.eye(8) - K @ self._H) @ self._P

        self.detection_confidence = detection_confidence
        self.frames_since_seen = 0
        self.hit_streak += 1
        self.history.append(bbox.centroid)
        if len(self.history) > 60:
            self.history.pop(0)

    @property
    def bbox(self) -> BBox:
        return _state_to_bbox(self.state)

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.state[4]), float(self.state[5]))
