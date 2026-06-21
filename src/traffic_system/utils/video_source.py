"""
Video source reader — wraps cv2.VideoCapture with frame sampling baked in,
so the pipeline downstream only ever sees frames at the configured
target_fps, regardless of the source's native frame rate. Works identically
for a video file, an RTSP URL, or a webcam index, since OpenCV abstracts
that already; this class just adds the sampling logic on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

import cv2
import numpy as np

from traffic_system.utils.config import SourceConfig
from traffic_system.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RawFrame:
    camera_id: str
    frame_id: int
    timestamp: datetime
    image: np.ndarray


class VideoSource:
    def __init__(self, source_config: SourceConfig, target_fps: int):
        self._camera_id = source_config.camera_id
        self._uri = source_config.uri
        self._target_fps = target_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_counter = 0

    def open(self) -> None:
        # An integer-looking URI is treated as a webcam index, matching
        # OpenCV's own convention, rather than guessing from string content.
        source = int(self._uri) if self._uri.isdigit() else self._uri
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open video source '{self._uri}' for camera '{self._camera_id}'. "
                f"Check sources[].uri in config.yaml."
            )
        logger.info("Opened video source '%s' for camera '%s'", self._uri, self._camera_id)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def frames(self) -> Iterator[RawFrame]:
        """
        Yields frames sampled down to target_fps. Computes the source's
        native FPS at open time and only yields every Nth frame, falling
        back to "yield every frame" if the source doesn't report a valid
        FPS (common for some RTSP streams), rather than crashing on a
        divide-by-zero.
        """
        if self._cap is None:
            raise RuntimeError("VideoSource.open() must be called before frames()")

        source_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            logger.warning(
                "Source for camera '%s' did not report a valid FPS; "
                "sampling will pass through every frame.", self._camera_id,
            )
            frame_interval = 1
        else:
            frame_interval = max(1, round(source_fps / self._target_fps))

        raw_index = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            if raw_index % frame_interval == 0:
                yield RawFrame(
                    camera_id=self._camera_id,
                    frame_id=self._frame_counter,
                    timestamp=datetime.now(timezone.utc),
                    image=frame,
                )
                self._frame_counter += 1

            raw_index += 1

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
