from __future__ import annotations

import asyncio
import base64
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .api.routes import get_service_instances, router as api_router
from .models.schemas import DetectionBox, LiveFrameMessage, SafetyAlert
from .utils.config import get_settings
from .utils.logging import setup_logger


logger = setup_logger(__name__)

_APP_VERSION = "1.0.0"


# ----------------------------------------------------------------------
# WebSocket Broadcast Manager
# ----------------------------------------------------------------------
class ConnectionManager:
    """
    Track currently connected WebSocket clients and broadcast messages
    to all (or a subset) of them.

    Fully thread-safe: WebSocket send operations are dispatched via the
    event loop onto which each client was accepted, using per-connection
    asyncio queues consumed by a lightweight per-client sender task.
    """

    def __init__(self) -> None:
        self._active: Dict[WebSocket, "asyncio.Queue[LiveFrameMessage]"] = {}
        self._sender_tasks: Dict[WebSocket, asyncio.Task] = {}
        self._lock = threading.Lock()

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._active)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        loop = asyncio.get_running_loop()
        with self._lock:
            q: "asyncio.Queue[LiveFrameMessage]" = asyncio.Queue(maxsize=128)
            self._active[ws] = q
            self._sender_tasks[ws] = loop.create_task(self._drain(ws, q))
        logger.info("WebSocket client connected. Total: %d", self.client_count)

    async def _drain(self, ws: WebSocket, q: "asyncio.Queue[LiveFrameMessage]") -> None:
        try:
            while True:
                msg = await q.get()
                try:
                    await ws.send_json(msg.model_dump(mode="json"))
                except Exception:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("WS drain task exiting: %s", exc)

    def disconnect(self, ws: WebSocket) -> None:
        with self._lock:
            q = self._active.pop(ws, None)
            task = self._sender_tasks.pop(ws, None)
        if task is not None and not task.done():
            task.cancel()
        if q is not None:
            # Drain any pending references
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        logger.info("WebSocket client disconnected. Total: %d", self.client_count)

    def broadcast(self, message: LiveFrameMessage) -> None:
        """
        Enqueue *message* to every connected client.

        Safe to call from non-async threads (camera/inference producer).
        Drops messages when a client queue is full instead of blocking the
        producer.
        """
        with self._lock:
            items = list(self._active.items())
        for ws, q in items:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                logger.debug("WS queue full for client; dropping frame.")

    def broadcast_dict(self, payload: Dict[str, Any]) -> None:
        try:
            self.broadcast(LiveFrameMessage(**payload))
        except Exception as exc:
            logger.debug("broadcast_dict validate failed: %s", exc)


# ----------------------------------------------------------------------
# Camera / Inference producer thread
# ----------------------------------------------------------------------
class LiveProducer:
    """
    Background thread that polls the camera at camera rate, runs inference
    at the configured INFERENCE_FPS throttle, and pushes detections, alerts
    and (optionally) JPEG frames to the ConnectionManager.

    Stopped cleanly via :meth:`stop` from the shutdown lifespan.
    """

    def __init__(self, services: Dict[str, Any], manager: ConnectionManager,
                 send_jpeg: bool = True, jpeg_quality: int = 70) -> None:
        self._services = services
        self._manager = manager
        self._send_jpeg = send_jpeg
        self._jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._periodic_status_interval = 5.0

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._loop = loop
        self._thread = threading.Thread(target=self._run, name="live-producer", daemon=True)
        self._thread.start()
        logger.info("LiveProducer started.")

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        logger.info("LiveProducer stopped.")

    # ------------------------------------------------------------------
    @staticmethod
    def _encode_jpeg(frame: np.ndarray, quality: int) -> Optional[str]:
        if frame is None or frame.size == 0:
            return None
        try:
            import cv2  # type: ignore

            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            if not ok or buf is None:
                return None
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception as exc:
            logger.debug("JPEG encode failed: %s", exc)
            return None

    def _make_status_message(self) -> LiveFrameMessage:
        cam = self._services["camera"]
        status = cam.get_status()
        metrics = self._services["inference"].get_metrics()
        info = self._services["inference"].provider_info()
        return LiveFrameMessage(
            type="status",
            backend=info.get("backend"),
            status={
                "camera": {
                    "index": status.index,
                    "opened": status.opened,
                    "width": status.width,
                    "height": status.height,
                    "frames_read": status.frames_read,
                    "last_error": status.last_error,
                    "backend": status.backend,
                },
                "metrics": metrics,
            },
        )

    def _run(self) -> None:
        cam = self._services["camera"]
        inference = self._services["inference"]
        detector = self._services["detector"]
        safety = self._services["safety"]
        last_status_ts = 0.0

        try:
            cam.start()
            inference.start()
        except Exception as exc:
            logger.exception("LiveProducer initialisation error: %s", exc)
            self._manager.broadcast(
                LiveFrameMessage(type="error", error=f"Producer init failed: {exc}")
            )
            return

        while not self._stop_event.is_set():
            try:
                now = time.perf_counter()
                if now - last_status_ts >= self._periodic_status_interval:
                    last_status_ts = now
                    try:
                        status_msg = self._make_status_message()
                        self._manager.broadcast(status_msg)
                    except Exception as exc:
                        logger.debug("Status emit failed: %s", exc)

                if self._manager.client_count == 0:
                    time.sleep(0.1)
                    continue

                frame = cam.read_frame(force=False)
                if frame is None:
                    time.sleep(0.03)
                    continue

                detections: List[DetectionBox] = detector.detect_objects(frame)
                alerts: List[SafetyAlert] = safety.analyze_frame(detections)
                info = inference.provider_info()

                base_payload: Dict[str, Any] = {
                    "backend": info.get("backend"),
                    "detections": detections,
                    "alerts": alerts,
                }

                if detections:
                    self._manager.broadcast(
                        LiveFrameMessage(type="detections", **base_payload)
                    )
                if alerts:
                    self._manager.broadcast(
                        LiveFrameMessage(type="alerts", **base_payload)
                    )
                if self._send_jpeg:
                    if detections:
                        annotated = detector.draw_bounding_boxes(frame, detections)
                        jpeg_frame = annotated if annotated is not None else frame
                    else:
                        jpeg_frame = frame
                    jpeg_b64 = self._encode_jpeg(jpeg_frame, self._jpeg_quality)
                    if jpeg_b64 is not None:
                        msg = LiveFrameMessage(
                            type="frame",
                            backend=info.get("backend"),
                            detections=detections,
                            alerts=alerts,
                            jpeg_base64=jpeg_b64,
                        )
                        self._manager.broadcast(msg)
            except Exception as exc:
                logger.exception("LiveProducer iteration error: %s", exc)
                try:
                    self._manager.broadcast(
                        LiveFrameMessage(type="error", error=str(exc))
                    )
                except Exception:
                    pass
                time.sleep(0.2)


# ----------------------------------------------------------------------
# FastAPI app factory
# ----------------------------------------------------------------------
def _install_signal_handlers(app: FastAPI) -> None:
    def _handle(signum, frame):
        logger.info("Received signal %s; triggering shutdown.", signum)
        try:
            live = getattr(app.state, "live_producer", None)
            if live is not None:
                live.stop()
        except Exception:
            pass
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)
    except (ValueError, OSError):
        # Not in main thread - skip
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: load hardware config + models on startup, release
    camera/inference resources on shutdown.
    """
    settings = get_settings()
    logger.info(
        "Starting SnapSight-AI backend v%s host=%s port=%s preferred_backend=%s fallback=%s",
        _APP_VERSION,
        settings.BACKEND_HOST,
        settings.BACKEND_PORT,
        settings.PREFERRED_BACKEND,
        settings.FALLBACK_BACKEND,
    )
    services = get_service_instances()
    hw_detector = services["hardware"]
    inference = services["inference"]
    camera = services["camera"]

    hw = hw_detector.detect()
    logger.info(
        "Hardware ready: cpu=%s ram=%.1fGB backends=%s",
        hw.cpu[:48],
        hw.ram_gb,
        hw.available_backends,
    )

    inference.start()
    info = inference.provider_info()
    logger.info(
        "Inference provider loaded: name=%s backend=%s device=%s loaded=%s load_ms=%s",
        info.get("name"),
        info.get("backend"),
        info.get("device"),
        info.get("loaded"),
        info.get("load_time_ms"),
    )

    if settings.CAMERA_INDEX is not None:
        try:
            camera.start()
        except Exception as exc:
            logger.warning("Camera pre-start failed (will retry on demand): %s", exc)

    manager = ConnectionManager()
    producer = LiveProducer(services, manager, send_jpeg=True, jpeg_quality=70)
    loop = asyncio.get_running_loop()
    producer.start(loop)

    app.state.services = services
    app.state.ws_manager = manager
    app.state.live_producer = producer

    _install_signal_handlers(app)

    yield

    logger.info("Shutting down SnapSight-AI backend.")
    try:
        producer.stop()
    except Exception:
        pass
    try:
        camera.stop()
    except Exception:
        pass
    try:
        inference.shutdown()
    except Exception:
        pass
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Application factory - build and configure the FastAPI app."""
    settings = get_settings()

    app = FastAPI(
        title="SnapSight AI Backend",
        description=(
            "Modular FastAPI backend for camera-based AI assistive features. "
            "Provides detection, OCR, scene description, chat, STT, safety "
            "analysis and a live WebSocket feed with JPEG frames & alerts."
        ),
        version=_APP_VERSION,
        lifespan=lifespan,
    )

    origins = [
        "*",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    # ------------------------------------------------------------------
    # Top-level aliases for convenience (not versioned)
    # ------------------------------------------------------------------
    @app.get("/health", tags=["system"])
    async def top_level_health():
        from .api.routes import get_health  # noqa: F402

        return await get_health()

    # ------------------------------------------------------------------
    # WebSocket endpoint
    # ------------------------------------------------------------------
    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket):
        manager: ConnectionManager = app.state.ws_manager
        await manager.connect(websocket)
        try:
            # Send an immediate status welcome frame
            try:
                loop = asyncio.get_running_loop()
                status_msg = await loop.run_in_executor(
                    None, _make_status_msg_sync, app.state.services
                )
                if status_msg is not None:
                    await websocket.send_json(status_msg.model_dump(mode="json"))
            except Exception as exc:
                logger.debug("WS welcome status failed: %s", exc)

            while True:
                # Receive any client message (keepalive ping / control)
                try:
                    data = await websocket.receive_text()
                except WebSocketDisconnect:
                    break
                if not data:
                    continue
                text = data.strip().lower()
                if text in ("ping", ""):
                    try:
                        await websocket.send_json(
                            {"type": "pong", "timestamp": time.time()}
                        )
                    except Exception:
                        break
                else:
                    logger.debug("WS unexpected text: %r", data[:120])
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.exception("WebSocket error: %s", exc)
        finally:
            manager.disconnect(websocket)

    return app


def _make_status_msg_sync(services: Dict[str, Any]) -> Optional[LiveFrameMessage]:
    try:
        cam = services["camera"]
        status = cam.get_status()
        metrics = services["inference"].get_metrics()
        info = services["inference"].provider_info()
        return LiveFrameMessage(
            type="status",
            backend=info.get("backend"),
            status={
                "camera": {
                    "index": status.index,
                    "opened": status.opened,
                    "width": status.width,
                    "height": status.height,
                    "frames_read": status.frames_read,
                    "last_error": status.last_error,
                    "backend": status.backend,
                },
                "metrics": metrics,
            },
        )
    except Exception as exc:
        logger.debug("_make_status_msg_sync failed: %s", exc)
        return None


app = create_app()


def _run_dev():  # pragma: no cover - manual entry point
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    _run_dev()


__all__ = [
    "app",
    "create_app",
    "ConnectionManager",
    "LiveProducer",
]
