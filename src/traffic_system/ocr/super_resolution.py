"""
Plate crop super-resolution, used only when a localized plate crop is too
small/low-resolution for reliable OCR. Two backends, selected via config:
  - "bicubic": classical upscaling, zero extra dependencies, always available.
  - "esrgan": learned super-resolution for sharper character edges, requires
    a weights file.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PlateSuperResolver:
    def __init__(self, app_config: AppConfig, device: torch.device):
        cfg = app_config.lpr["super_resolution"]
        self._enabled = cfg["enabled"]
        self._scale = cfg["upscale_factor"]
        self._backend = cfg["backend"]
        self._device = device
        self._esrgan_model = None

        if self._enabled and self._backend == "esrgan":
            weights_path = app_config.resolve_path(cfg["esrgan_weights_path"])
            if weights_path.exists():
                self._esrgan_model = self._load_esrgan(str(weights_path))
            else:
                logger.warning(
                    "ESRGAN weights not found at %s — falling back to bicubic "
                    "upscaling. Train/provide ESRGAN weights for sharper results.",
                    weights_path,
                )
                self._backend = "bicubic"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def upscale(self, crop: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return crop
        if self._backend == "esrgan" and self._esrgan_model is not None:
            return self._upscale_esrgan(crop)
        return self._upscale_bicubic(crop)

    # ------------------------------------------------------------------ internals

    def _upscale_bicubic(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        return cv2.resize(
            crop, (w * self._scale, h * self._scale), interpolation=cv2.INTER_CUBIC
        )

    def _load_esrgan(self, weights_path: str):
        try:
            model = torch.load(weights_path, map_location=self._device)
            model.eval()
            model.to(self._device)
            return model
        except Exception as e:
            logger.warning(
                "Failed to load ESRGAN weights from %s (%s) — falling back to bicubic.",
                weights_path, e,
            )
            self._backend = "bicubic"
            return None

    @torch.no_grad()
    def _upscale_esrgan(self, crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)
        output = self._esrgan_model(tensor)
        output = output.squeeze(0).clamp(0, 1).cpu().numpy()
        output = (output.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
