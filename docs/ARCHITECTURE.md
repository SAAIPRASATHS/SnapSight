# ARCHITECTURE — SnapSight AI

Deep-dive into SnapSight's design: component breakdown, data flow, the
`ModelProvider` abstraction, and how the frontend and backend collaborate over
REST + WebSockets.

---

## 1. High-Level Goals

Every architectural decision in SnapSight traces back to four non-negotiable
requirements:

1. **Privacy by default.** Pixels and audio never leave the machine. No
   telemetry, no cloud round-trips unless a user explicitly flips
   `ENABLE_CLOUD_APIS=true`.
2. **Snapdragon NPU-first, but portable.** The best experience is on Hexagon,
   but the same binary tree must start cleanly on a random x86 dev laptop.
3. **Per-task independent throughput.** Detection at 5 FPS should not starve
   OCR running at 1 FPS; each task owns its worker thread / asyncio task.
4. **Zero surprises for ops people.** Standard tech (FastAPI, React/Vite,
   YAML config) so any engineer can read the codebase and ship it.

---

## 2. Layer Diagram (Detailed)

```
                    ┌─────────────────────────────────────┐
                    │         BROWSER / UI THREAD         │
                    │                                     │
                    │  · React state, Redux / Zustand     │
                    │  · getUserMedia() → <video> tag     │
                    │  · Canvas overlay for bounding box  │
                    │  · Transcript + chat DOM            │
                    └──────────────┬──────────────────────┘
                                   │  ws:// + http:// (localhost)
                 ┌─────────────────┴─────────────────┐
                 │   FASTAPI PROCESS (single host)   │
                 │                                   │
                 │  ┌─────────────────────────────┐  │
   REST  ───────►  │  routers/                     │  │
   GET /health    │   · health_router.py          │  │
   POST /detect   │   · detection_router.py       │  │
   POST /ocr      │   · ocr_router.py             │  │
   POST /vlm      │   · vlm_router.py             │  │
   GET  /models   │   · stt_router.py             │  │
                 │   · config_router.py           │  │
                 │  └──────────────┬──────────────┘  │
                 │                 │                 │
                 │  ┌──────────────▼──────────────┐  │
   WS    ───────►  │  websocket endpoints         │  │
   /ws/camera     │   · ws_camera.py              │  │
   /ws/stt        │   · ws_stt.py                 │  │
   /ws/vlm        │   · ws_vlm.py                 │  │
                 │  └──────────────┬──────────────┘  │
                 │                 │                 │
                 │  ┌──────────────▼──────────────┐  │
                 │  │  services / orchestration   │  │
                 │  │   · CameraLoopService       │  │
                 │  │   · InferenceScheduler      │  │
                 │  │   · STTAudioStream          │  │
                 │  │   · BenchmarkRunner         │  │
                 │  └──────────────┬──────────────┘  │
                 │                 │                 │
                 │  ┌──────────────▼──────────────┐  │
                 │  │  core / ModelProvider ABC   │  │◄────── contracts
                 │  │  ┌────────────────────────┐ │  │
                 │  │  │ class ModelProvider:   │ │  │
                 │  │  │   +load(config)        │ │  │
                 │  │  │   +detect(frame)       │ │  │
                 │  │  │   +transcribe(audio)   │ │  │
                 │  │  │   +ocr(frame)          │ │  │
                 │  │  │   +vlm(frame, prompt)  │ │  │
                 │  │  └────────────────────────┘ │  │
                 │  └──┬──────────┬──────────┬─────┘  │
                 │     │          │          │        │
                 │  ┌──▼───┐  ┌───▼──┐  ┌───▼─────┐  │
                 │  │ QAI  │  │ ONNX │  │ PyTorch │  │
                 │  │ Hub  │  │ Rt.  │  │         │  │   ═╗ fallback
                 │  │ Prov.│  │ Prov.│  │  Prov.  │  │    ║
                 │  └──┬───┘  └──┬───┘  └──┬──────┘  │    ║
                 │     │         │         │         │  ┌─▼──┐
                 └─────┼─────────┼─────────┼─────────┘  │Mock│
                       │         │         │            │Prov│
                ┌──────▼─────┐ ┌▼────────┐┌▼────────┐  └─┬──┘
                │  QNN SDK   │ │  ONNX   ││ PyTorch │    │
                │ (Hexagon   │ │ Runtime ││         │    │
                │  NPU/GPU)  │ │ DML/CPU ││ CPU/CUDA│    │
                └────────────┘ └─────────┘└─────────┘    │
                                                         │
                ┌────────────────────────────────────┐   │
                │ config/hardware.yaml + .env       │◄──┘
                │ models/<task>/<weights>           │
                └────────────────────────────────────┘
```

---

## 3. Component Breakdown

### 3.1 Frontend (React + Vite)

| File / Module | Responsibility |
|---|---|
| `App.tsx` + router | Top-level layout, tab switcher between Detect / STT / OCR / VLM. |
| `CameraView.tsx` | Opens `navigator.mediaDevices.getUserMedia()`, renders `<video>` + overlay `<canvas>`. Draws bounding boxes received on `/ws/camera`. |
| `TranscriptView.tsx` | Renders the rolling STT transcript window, auto-scroll, copy-to-clipboard. |
| `ChatView.tsx` | VLM chat UI: textbox for prompts, streamed token rendering over `/ws/vlm` SSE-like WebSocket channel. |
| `OCRView.tsx` | OCR trigger button, overlay text layer on the frame, per-region text copy. |
| `apiClient.ts` | Shared `axios` + native `WebSocket` wrappers that read `BACKEND_HOST` / `BACKEND_PORT` from Vite env vars. Implements reconnect + exponential back-off. |
| `hooks/useBackendStatus.ts` | Polls `/health` every 5 s. Shows a red banner if backend is unreachable. |

### 3.2 Backend (FastAPI + Uvicorn)

#### Routers (`backend/routers/*.py`)

**Stateless, thin HTTP adapters.** They validate input with Pydantic models,
hand off to a service, and serialize the Pydantic response back to JSON.
*No inference happens inside a router.*

#### WebSocket Endpoints (`backend/ws/*.py`)

WebSocket endpoints are stateful per connection:

- **`/ws/camera`** — The frontend sends base64-encoded JPEG frames; the backend
  runs the detection loop at `INFERENCE_FPS` and streams back `{boxes, labels,
  scores, fps}` messages. Frame throttling is handled on the server side so a
  fast browser sender cannot flood the NPU.
- **`/ws/stt`** — Frontend streams 16 kHz mono PCM chunks via binary WS
  messages; backend accumulates a rolling 30 s buffer and calls
  `ModelProvider.transcribe()` on VAD-detected utterance boundaries. Partial
  transcripts are echoed back immediately.
- **`/ws/vlm`** — JSON request/response channel. Client sends `{frame_id,
  prompt, max_new_tokens}`, server streams back one token per WebSocket text
  message until a `[DONE]` sentinel.

#### Core Services (`backend/services/*.py`)

| Service | Purpose |
|---|---|
| `CameraLoopService` | Owns one `asyncio.Task` per live WS camera connection. Pulls frames, runs the detection model at the configured cadence, overlays results for OCR/VLM. |
| `InferenceScheduler` | Implements task-level rate-limiting + thread-pool sizing. Uses the configured `PREFERRED_BACKEND` to pick an appropriate worker count (NPU jobs are serialized because Hexagon has one command queue; CPU jobs can parallelize across `os.cpu_count()`). |
| `STTAudioStream` | Ring buffer + VAD + transcript aggregator. |
| `BenchmarkRunner` | Invoked by `benchmarks/run_benchmarks.py`. Spins up synthetic frames / audio, runs each model N times, records wall-clock + psutil counters. |

### 3.3 ModelProvider Layer (`backend/providers/*.py`)

The heart of SnapSight's portability story. **All four capabilities share one
abstract base class** so every backend is a first-class citizen:

```
ModelProvider (ABC)
  ├── QualcommAIHubProvider
  │     └── uses qai-hub SDK, dispatches to Hexagon NPU
  ├── ONNXProvider
  │     └── uses onnxruntime (CPU / DirectML EP / CUDA EP)
  ├── PyTorchProvider
  │     └── uses torch + torchvision (CPU / CUDA / MPS)
  └── MockProvider
        └── deterministic, no-op impl; zero dependencies
```

**Initialization contract:**

1. `ModelProviderFactory.resolve(preferred, fallback, hw_config)` loops over
   candidates and calls `.probe()` on each.
2. `.probe()` returns `True` iff the runtime is importable AND the weight file
   from `hardware.yaml` exists on disk AND the runtime can open that file.
3. First provider that probes successfully is cached on the FastAPI app
   singleton; if none do, the backend starts but all inference endpoints
   return `503 Service Unavailable` with a human-readable reason.

This means a Snapdragon laptop missing a weight file will gracefully degrade
rather than crash on import.

---

## 4. Data Flow

### 4.1 Real-Time Object Detection (Hot Path)

```
  Browser                              Backend
    │                                     │
    │  getUserMedia → <video>             │
    │                                     │
    │  setInterval(33 ms)                 │
    │   └─ canvas.toDataURL(jpeg)         │
    │  ──────────── WS binary msg ───────►│
    │                                     │  CameraLoopService.dequeue()
    │                                     │    → throttle to INFERENCE_FPS
    │                                     │    → decode jpeg → numpy H×W×3
    │                                     │  provider.detect(frame)
    │                                     │    → QNN / ONNX / Torch forward
    │                                     │    → NMS → boxes/confidences
    │                                     │
    │  ◄──────── WS JSON msg ─────────────│
    │    {boxes, labels, scores, fps}     │
    │                                     │
    │  requestAnimationFrame draw()       │
    │    └─ canvas strokeRect() + text    │
    ▼                                     ▼
```

Key optimizations:
- Frames use **JPEG binary**, not base64, on the wire.
- A single detection result may be replayed for 2–3 UI frames if the inference
  cadence (5 FPS) is slower than the display refresh rate.
- NMS happens inside the provider (not in routers) so each backend can use the
  most optimized path for its runtime.

### 4.2 VLM Q&A (Latency-Sensitive)

```
  User clicks "Ask about this frame"
    │
    ├─ UI freezes current <video> frame id
    ├─ User types prompt → [Send]
    │
    └─ WS JSON → /ws/vlm:  {frame_id, prompt, max_new_tokens=256}
           Backend:
             1. Pull cached numpy frame from CameraLoopService
             2. provider.vlm(frame, prompt) → generator of str tokens
             3. For each token: await ws.send_text(tok)
             4. Send final "[DONE]" + usage stats
    ┌──────────────────────────────────────────┐
    │  Browser: stream tokens into the chat    │
    │  bubble one character at a time via      │
    │  incremental DOM append                  │
    └──────────────────────────────────────────┘
```

---

## 5. Modular ModelProvider Pattern in Depth

Why one ABC with four methods rather than four separate ABCs per task? Because
**models are increasingly multimodal**. A single shared backbone (e.g. a
future Llama 4 vision-moE on Qualcomm AI Hub) could serve detection + VLM +
OCR embeddings from one forward pass. The unified interface lets us refactor
internally *without touching routers or services*.

### Provider Implementation Checklist

Every new provider (e.g. a future `TensorRTProvider` or `OpenVINOProvider`)
must:

1. Implement `probe(config: HardwareConfig) -> bool` — quick check, no heavy
   work, must not throw.
2. Implement `load(config: HardwareConfig) -> None` — load weights into device
   memory; called exactly once at startup.
3. For each of the four abstract methods (`detect`, `transcribe`, `ocr`,
   `vlm`): either provide a real implementation OR raise
   `NotImplementedForProviderError` (callers will fall back to the next-best
   provider if configured, or return a clear `501 Not Implemented`).
4. Accept and return the **exact Pydantic shapes** declared in
   `backend/schemas/`. Any deviation breaks the frontend contract.
5. Honor `CONFIDENCE_THRESHOLD` from config as an **additional** post-filter,
   even if the underlying runtime already filters.

### AUTO Resolution Order

`PREFERRED_BACKEND=AUTO` expands on startup to the following ordered probe
list. The first fully-loaded provider wins.

```
If `device.vendor == Qualcomm` and `device.platform == Snapdragon`:
  1. QualcommAIHubProvider
  2. ONNXProvider     (DirectML EP on Windows, CPU EP elsewhere)
  3. PyTorchProvider  (CPU)
  4. MockProvider
Else:
  1. ONNXProvider     (best EP available)
  2. PyTorchProvider
  3. MockProvider
```

---

## 6. Frontend / Backend Separation Principles

We intentionally separate the two processes even for single-host deployment:

| Concern | Frontend owns | Backend owns |
|---|---|---|
| **Latency** | Sub-16 ms visual rendering, WebSocket buffering, reconnection. | Sub-100 ms first-token for VLM, deterministic INFERENCE_FPS. |
| **State** | Current UI tab, transcript scroll position, chat history, unsent prompt. | Loaded model weights, per-connection camera ring buffers, hardware counters. |
| **Failure domain** | Browser crash / tab close is harmless. | Backend crash must not corrupt on-disk model files; auto-restart via a service wrapper (NSSM / systemd) is expected. |
| **Auth | Not needed (localhost only in v1). | Not needed — loopback only via BACKEND_HOST defaults. Can add API keys later without frontend changes. |

---

## 7. WebSocket Flow Specification

All WebSocket endpoints use the following sub-protocol conventions so the
frontend can share one reconnection library:

### 7.1 Connection Lifecycle

1. Client opens WS → server responds with a single JSON text message:
   ```json
   {"type":"hello","protocol_version":1,"server_time_utc":"..."}
   ```
2. Heartbeat: server sends `{"type":"ping"}` every 15 s. Client MUST reply
   with `{"type":"pong"}` within 5 s or the connection is dropped.
3. Client or server may send `{"type":"error","code":...,"message":"..."}` at
   any time. Critical errors set `code >= 4000` and the peer will close.
4. Graceful close: client sends `{"type":"bye"}`, server echoes and closes
   with code 1000.

### 7.2 Channel Payload Contract

| Endpoint | Direction | Type | Payload Shape |
|---|---|---|---|
| `/ws/camera` | C→S | binary | JPEG bytes (no envelope, no header) |
| `/ws/camera` | S→C | `detection_result` | `{type, seq, fps, boxes:[{x1,y1,x2,y2,label,score}], ocr_regions:[...]}` |
| `/ws/stt`    | C→S | binary | 16 kHz s16le mono PCM chunks (≥ 20 ms each) |
| `/ws/stt`    | S→C | `stt_partial` / `stt_final` | `{type, seq, text, is_final, latency_ms}` |
| `/ws/vlm`    | C→S | `vlm_request` | `{type, frame_id, prompt, max_new_tokens, temperature}` |
| `/ws/vlm`    | S→C | `vlm_token` then `vlm_done` | `{type, token}` repeated, finally `{type, done, total_tokens, tpot_ms}` |

All S→C messages include a monotonically increasing `seq` integer so the UI
can detect (and ignore) reordered packets under load.
