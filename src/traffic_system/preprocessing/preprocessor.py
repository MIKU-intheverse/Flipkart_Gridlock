"""
Stage 1: Image Preprocessing.

Implements the conditional-correction design from the project plan: a cheap
quality check runs on every frame, and only frames that actually need it get
the relevant fix (CLAHE for low light, a mitigation pass for rain, a blur
flag for motion blur). Clean frames pass through untouched.

Nothing here is model-based — this stage is intentionally classical/OpenCV
only, matching the "deliberately simple" scope decision in the project plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    brightness: float
    blur_score: float
    is_dark: bool
    is_blurry: bool


class Preprocessor:
    """
    Stateful per-camera preprocessor. Stateful because the "skip blurry frame
    if a recent sharp frame of the same scene exists" rule needs a small
    rolling memory of recent blur scores per camera.
    """

    def __init__(self, app_config: AppConfig):
        cfg = app_config.preprocessing
        self._enabled = cfg["enabled"]
        self._clahe_clip_limit = cfg["clahe"]["clip_limit"]
        self._clahe_tile_grid = tuple(cfg["clahe"]["tile_grid_size"])
        self._brightness_threshold = cfg["brightness_threshold_low"]
        self._blur_threshold = cfg["blur"]["laplacian_var_threshold"]
        self._skip_blurry = cfg["blur"]["skip_blurry_if_recent_sharp_exists"]
        self._recent_window = cfg["blur"]["recent_sharp_window_frames"]
        self._rain_enabled = cfg["rain_mitigation"]["enabled"]
        self._rain_kernel = cfg["rain_mitigation"]["median_blur_kernel"]
        self._rain_sharpen_strength = cfg["rain_mitigation"]["sharpen_strength"]
        self._shadow_gamma = cfg["shadow_gamma_correction"]

        self._clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip_limit, tileGridSize=self._clahe_tile_grid
        )
        # Per-camera rolling history of recent blur scores, used by the
        # "skip if a recent sharp frame exists" rule.
        self._recent_blur_history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ public

    def assess_quality(self, frame: np.ndarray) -> QualityMetrics:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return QualityMetrics(
            brightness=brightness,
            blur_score=blur_score,
            is_dark=brightness < self._brightness_threshold,
            is_blurry=blur_score < self._blur_threshold,
        )

    def process(self, frame: np.ndarray, camera_id: str) -> tuple[np.ndarray, dict]:
        """
        Returns (possibly enhanced frame, quality_flags dict). quality_flags
        is attached to the FrameContext so downstream stages (in particular
        the confidence-scoring step in violations) can see that a frame was
        degraded, even after it's been corrected.
        """
        if not self._enabled:
            return frame, {}

        metrics = self.assess_quality(frame)
        flags: dict = {
            "brightness": metrics.brightness,
            "blur_score": metrics.blur_score,
            "was_dark": metrics.is_dark,
            "was_blurry": metrics.is_blurry,
        }

        processed = frame

        if metrics.is_dark:
            processed = self._apply_clahe(processed)
            processed = self._apply_shadow_gamma(processed)

        if metrics.is_blurry:
            skip = self._should_skip_due_to_recent_sharp(camera_id, metrics.blur_score)
            flags["blurry_frame_skipped"] = skip
            if not skip and self._rain_enabled:
                # Token mitigation only — see project plan's stated limitation
                # that full deraining/deblurring is out of scope for this prototype.
                processed = self._apply_mitigation_filter(processed)

        self._update_blur_history(camera_id, metrics.blur_score)
        return processed, flags

    # ------------------------------------------------------------------ internals

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_channel)
        merged = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _apply_shadow_gamma(self, frame: np.ndarray) -> np.ndarray:
        gamma = self._shadow_gamma
        if gamma == 1.0:
            return frame
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype("uint8")
        return cv2.LUT(frame, table)

    def _apply_mitigation_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Lightweight classical filter used as a token mitigation for rain/blur
        artifacts: a median blur to suppress high-frequency streak noise,
        followed by an unsharp-mask pass to recover edge definition. This is
        explicitly not a learned deraining/deblurring model — see the
        project plan's documented limitation on this point.
        """
        k = self._rain_kernel if self._rain_kernel % 2 == 1 else self._rain_kernel + 1
        denoised = cv2.medianBlur(frame, k)
        blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
        sharpened = cv2.addWeighted(
            denoised, 1 + self._rain_sharpen_strength,
            blurred, -self._rain_sharpen_strength, 0,
        )
        return sharpened

    def _should_skip_due_to_recent_sharp(self, camera_id: str, current_blur_score: float) -> bool:
        if not self._skip_blurry:
            return False
        history = self._recent_blur_history.get(camera_id, [])
        return any(score >= self._blur_threshold for score in history)

    def _update_blur_history(self, camera_id: str, blur_score: float) -> None:
        history = self._recent_blur_history.setdefault(camera_id, [])
        history.append(blur_score)
        if len(history) > self._recent_window:
            history.pop(0)
