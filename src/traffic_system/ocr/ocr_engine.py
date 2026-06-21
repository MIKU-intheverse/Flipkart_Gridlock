"""
PaddleOCR wrapper for plate text extraction. Isolated into its own module
so the rest of the LPR pipeline depends on a simple (image -> text, confidence)
interface rather than PaddleOCR's own result format.

Import/initialization safety: importing `paddleocr` transitively imports
the PaddlePaddle framework (a C++ extension), which in network-constrained
environments has been observed to hang indefinitely during its own startup
checks — independent of and prior to any explicit model-download step.

A hung *thread* stuck inside a native C-extension import cannot be safely
abandoned: PaddlePaddle registers atexit hooks tied to its native runtime
(see paddle/base/__init__.py), and abandoning a thread mid-import can leave
that native state corrupted in ways that hang or crash the *main* thread
during interpreter shutdown — which is exactly the symptom this module
works around. The only reliably safe way to bound this is to run the
import/initialization in a separate **process**: if it doesn't finish in
time, the child process is killed outright, which cannot corrupt the
parent process's memory space. The actual OCR engine object then has to
live in that same child process for inference too (it can't be pickled
back), so this module runs PaddleOCR as a small persistent worker process
and communicates with it over a pipe.
"""

from __future__ import annotations

import multiprocessing
import traceback
from typing import Optional

import numpy as np

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger

logger = get_logger(__name__)

_INIT_TIMEOUT_SECONDS = 45.0
_INFERENCE_TIMEOUT_SECONDS = 10.0


class OcrEngineLoadError(Exception):
    pass


def _worker_main(lang: str, use_gpu: bool, request_conn, ready_conn) -> None:
    """
    Entry point for the worker process. Initializes PaddleOCR, signals
    readiness (or failure) back to the parent over `ready_conn`, then
    services (image_bytes, shape, dtype) inference requests over
    `request_conn` until the parent closes the connection or the process
    is killed.
    """
    try:
        from paddleocr import PaddleOCR
        engine = PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=use_gpu)
    except Exception as e:  # noqa: BLE001 - report any failure to the parent
        ready_conn.send(("error", f"{e}\n{traceback.format_exc()}"))
        return

    ready_conn.send(("ok", None))

    while True:
        try:
            message = request_conn.recv()
        except (EOFError, OSError):
            break  # parent closed the connection — exit cleanly

        if message is None:  # explicit shutdown sentinel
            break

        image = message
        try:
            result = engine.ocr(image, cls=False)
            if not result or not result[0]:
                request_conn.send(("ok", (None, 0.0)))
                continue

            texts, confidences = [], []
            for line in result[0]:
                text, confidence = line[1]
                texts.append(text)
                confidences.append(confidence)

            if not texts:
                request_conn.send(("ok", (None, 0.0)))
                continue

            combined_text = "".join(texts).upper().replace(" ", "")
            avg_confidence = float(np.mean(confidences))
            request_conn.send(("ok", (combined_text, avg_confidence)))
        except Exception as e:  # noqa: BLE001
            request_conn.send(("error", str(e)))


class PlateOcrEngine:
    """
    Owns a single long-lived PaddleOCR worker process. `read_text()` sends
    one image per call and blocks (with a timeout) for the response —
    functionally identical to calling an in-process model, but immune to
    a hung/corrupted native import taking down the main pipeline process.
    """

    def __init__(self, app_config: AppConfig):
        cfg = app_config.lpr
        self._lang = cfg["ocr_lang"]
        self._use_gpu = cfg["ocr_use_gpu"]
        self._process: Optional[multiprocessing.Process] = None
        self._request_conn = None
        self._start_worker()

    def _start_worker(self) -> None:
        ctx = multiprocessing.get_context("spawn")  # spawn, not fork: safest with C extensions
        parent_request_conn, child_request_conn = ctx.Pipe()
        parent_ready_conn, child_ready_conn = ctx.Pipe()

        process = ctx.Process(
            target=_worker_main,
            args=(self._lang, self._use_gpu, child_request_conn, child_ready_conn),
            daemon=True,   # dies automatically if the parent process dies
        )
        process.start()

        if parent_ready_conn.poll(timeout=_INIT_TIMEOUT_SECONDS):
            status, error_detail = parent_ready_conn.recv()
        else:
            status, error_detail = "timeout", None

        if status != "ok":
            process.kill()   # safe: killing a process, not abandoning a thread
            process.join(timeout=5)
            if status == "timeout":
                raise OcrEngineLoadError(
                    f"Initializing PaddleOCR did not complete within "
                    f"{_INIT_TIMEOUT_SECONDS:.0f}s in its worker process "
                    f"(this typically means the environment cannot reach "
                    f"PaddleOCR's model-download servers and the import/init "
                    f"hung rather than failing cleanly). Pre-download "
                    f"PaddleOCR's model weights on a machine with network "
                    f"access and point PADDLEOCR_HOME at them, or check "
                    f"network/firewall settings."
                )
            raise OcrEngineLoadError(f"Failed to initialize PaddleOCR: {error_detail}")

        self._process = process
        self._request_conn = parent_request_conn
        logger.info("PaddleOCR worker process ready (pid=%d, lang=%s, gpu=%s)",
                    process.pid, self._lang, self._use_gpu)

    def read_text(self, plate_crop: np.ndarray) -> tuple[str | None, float]:
        if plate_crop is None or plate_crop.size == 0:
            return None, 0.0

        if self._process is None or not self._process.is_alive():
            logger.warning("PaddleOCR worker process is not running; skipping OCR for this crop.")
            return None, 0.0

        try:
            self._request_conn.send(plate_crop)
        except (BrokenPipeError, OSError) as e:
            logger.warning("Failed to send crop to PaddleOCR worker: %s", e)
            return None, 0.0

        if not self._request_conn.poll(timeout=_INFERENCE_TIMEOUT_SECONDS):
            logger.warning(
                "PaddleOCR worker did not respond within %.0fs; treating this "
                "plate as unread rather than blocking the pipeline.",
                _INFERENCE_TIMEOUT_SECONDS,
            )
            return None, 0.0

        status, payload = self._request_conn.recv()
        if status == "error":
            logger.warning("PaddleOCR inference failed: %s", payload)
            return None, 0.0

        return payload

    def shutdown(self) -> None:
        """Optional explicit cleanup; safe to skip since the worker is a
        daemon process and is reaped automatically when the parent exits."""
        if self._process is not None and self._process.is_alive():
            try:
                self._request_conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
