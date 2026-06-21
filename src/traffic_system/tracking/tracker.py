"""
Stage 3: Multi-Object Tracking — a ByteTrack-style tracker.

ByteTrack's defining idea, implemented here: associate using BOTH high and
low confidence detections rather than discarding low-confidence boxes
outright. High-confidence detections are matched first against predicted
track positions; whatever tracks remain unmatched get a second matching
pass against the low-confidence detections, which recovers tracks through
brief occlusion/motion-blur dips that would otherwise be lost.

Built from scratch on top of a Kalman filter + Hungarian (linear_sum_assignment)
matching, rather than importing an external ByteTrack package, to keep the
project self-contained with only declared dependencies (numpy/scipy).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.utils.types import BBox, Detection, TrackedObject
from traffic_system.tracking.kalman_track import KalmanBoxTracker

logger = get_logger(__name__)


def _iou_cost_matrix(tracks: list[BBox], detections: list[BBox]) -> np.ndarray:
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))
    cost = np.zeros((len(tracks), len(detections)))
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            cost[i, j] = 1.0 - t.iou(d)
    return cost


def _match(tracks: list[BBox], detections: list[BBox], iou_threshold: float):
    """
    Returns (matched_pairs, unmatched_track_idx, unmatched_det_idx) using
    Hungarian assignment on an IoU-distance cost matrix, rejecting matches
    below the configured IoU threshold.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost = _iou_cost_matrix(tracks, detections)
    row_idx, col_idx = linear_sum_assignment(cost)

    matched_pairs = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(detections)))

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= (1.0 - iou_threshold):
            matched_pairs.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

    return matched_pairs, sorted(unmatched_tracks), sorted(unmatched_dets)


class ByteTrackTracker:
    """
    One instance per camera stream — tracks must not be shared across
    cameras, since track IDs and motion history are scene-specific.
    """

    def __init__(self, app_config: AppConfig):
        cfg = app_config.tracking
        self._track_thresh = cfg["track_thresh"]
        self._match_thresh = cfg["match_thresh"]
        self._buffer_frames = cfg["track_buffer_frames"]
        self._tracks: list[KalmanBoxTracker] = []

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        # Split into high/low confidence pools — the core ByteTrack idea.
        high_conf = [d for d in detections if d.confidence >= self._track_thresh]
        low_conf = [d for d in detections if d.confidence < self._track_thresh]

        # Predict every existing track forward one step before matching.
        predicted_boxes = [t.predict() for t in self._tracks]

        # ---- Pass 1: match existing tracks against HIGH-confidence detections.
        high_boxes = [d.bbox for d in high_conf]
        matched, unmatched_track_idx, unmatched_high_idx = _match(
            predicted_boxes, high_boxes, self._match_thresh
        )
        for track_idx, det_idx in matched:
            self._tracks[track_idx].update(
                high_conf[det_idx].bbox, high_conf[det_idx].confidence
            )

        # ---- Pass 2: remaining unmatched tracks get a second chance against
        # LOW-confidence detections — this is what recovers tracks through
        # brief occlusion or motion blur instead of dropping them.
        remaining_predicted = [predicted_boxes[i] for i in unmatched_track_idx]
        low_boxes = [d.bbox for d in low_conf]
        matched_low, still_unmatched_local, unmatched_low_idx = _match(
            remaining_predicted, low_boxes, self._match_thresh
        )
        for local_track_idx, det_idx in matched_low:
            real_track_idx = unmatched_track_idx[local_track_idx]
            self._tracks[real_track_idx].update(
                low_conf[det_idx].bbox, low_conf[det_idx].confidence
            )

        still_unmatched_track_idx = {unmatched_track_idx[i] for i in still_unmatched_local}

        # ---- New tracks: any high-confidence detection that matched nothing
        # starts a brand-new track. (Low-confidence detections never spawn
        # new tracks — only confirm existing ones — to avoid creating
        # spurious tracks from noisy single-frame false positives.)
        for det_idx in unmatched_high_idx:
            det = high_conf[det_idx]
            self._tracks.append(KalmanBoxTracker(det.bbox, det.class_name, det.confidence))

        # ---- Prune tracks that have been unmatched too long.
        self._tracks = [
            t for t in self._tracks
            if t.frames_since_seen <= self._buffer_frames
        ]

        return self._to_tracked_objects()

    def _to_tracked_objects(self) -> list[TrackedObject]:
        out = []
        for t in self._tracks:
            if t.frames_since_seen > 0:
                continue  # only report tracks confirmed this frame
            out.append(TrackedObject(
                track_id=t.track_id,
                class_name=t.class_name,
                bbox=t.bbox,
                detection_confidence=t.detection_confidence,
                velocity=t.velocity,
                frames_since_seen=t.frames_since_seen,
                history=list(t.history),
            ))
        return out
