# MODEL SETUP — SnapSight AI ModelProvider Configuration

SnapSight ships with a pluggable `ModelProvider` architecture. This guide
walks through configuring each supported provider from scratch, then shows
how to wire them into `config/hardware.yaml`.

---

## Contents

1. [Quick Start Decision Tree](#1-quick-start-decision-tree)
2. [Provider 1 — QualcommAIHubProvider (Recommended for Snapdragon)](#2-provider-1--qualcommaihubprovider-recommended-for-snapdragon)
3. [Provider 2 — ONNXProvider (Fallback / x86 / Linux)](#3-provider-2--onnxprovider-fallback--x86--linux)
4. [Provider 3 — PyTorchProvider (Research / Dev)](#4-provider-3--pytorchprovider-research--dev)
5. [Provider 4 — MockProvider (Default, No Setup Needed)](#5-provider-4--mockprovider-default-no-setup-needed)
6. [Configuring `hardware.yaml`](#6-configuring-hardwareyaml)
7. [Validation](#7-validation)
8. [Switching Providers at Runtime](#8-switching-providers-at-runtime)

---

## 1. Quick Start Decision Tree

```
Start
  │
  ├─ Running on Snapdragon Windows PC?
  │    ├─ YES → Pick QualcommAIHubProvider (§2) → best FPS + NPU offload
  │    └─ NO  → Continue
  │
  ├─ Have pre-exported *.onnx files and want cross-platform portability?
  │    ├─ YES → Pick ONNXProvider (§3)
  │    └─ NO  → Continue
  │
  ├─ Model authoring / testing, happy with CPU speed?
  │    ├─ YES → Pick PyTorchProvider (§4)
  │    └─ NO  → Continue
  │
  └─ Everything else / just want the UI to boot?
       └─ MockProvider (§5) — always works, zero downloads
```

You can also mix providers per-task via `hardware.yaml` overrides (see §6).

---

## 2. Provider 1 — QualcommAIHubProvider (Recommended for Snapdragon)

This provider wraps the official **Qualcomm AI Hub** Python SDK and produces
the highest-throughput, lowest-power execution paths on Snapdragon X Elite /
X Plus / 8cx Gen 4 silicon.

### 2.1 Install the `qai-hub` Package

```bash
# With your Python venv active:
pip install --upgrade pip
pip install qai-hub

# Verify
qai-hub --version
# Expected output example: qai-hub, version 2.13.0
```

> On corporate networks you may need to set `PIP_EXTRA_INDEX_URL` to point at
> your internal PyPI mirror, or set `HTTPS_PROXY` before the install command.

### 2.2 Authenticate (If Required by Your Org)

Some enterprise catalogs require a Qualcomm account token:

```bash
qai-hub auth login
# … browser opens, complete SSO, return to shell
qai-hub auth status
# Expected: Logged in as <your-email>
```

Skip this step if you are only using public catalog entries.

### 2.3 Download Optimized Models

Run each command **from the repository root** so output folders land under
`./models/<task>/`:

```bash
# Task 1 — Object Detection: YOLOv8n INT8 quantized for Hexagon
qai-hub download object-detection/yolov8n-quant \
  --output models/object_detection/yolov8n-quant

# Task 2 — Speech-to-Text: Whisper Tiny INT8 (multilingual)
qai-hub download speech/whisper-tiny-quant \
  --output models/speech_to_text/whisper-tiny-quant

# Task 3 — OCR: PaddleOCR Mobile INT8 (det + rec bundled)
qai-hub download ocr/paddleocr-mobile-quant \
  --output models/ocr/paddleocr-mobile-quant

# Task 4 — Vision-Language: MobileVLM INT8 (decoder-only 1.4B)
qai-hub download vlm/mobilenet-vlm-quant \
  --output models/vision_language/mobilenet-vlm-quant
```

Each invocation produces a **directory** (not a single file) with:

```
models/object_detection/yolov8n-quant/
├── model.json                  # Hub metadata + runtime opset
├── model.safetensors / *.bin   # Weights
├── qnn_context.bin             # Precompiled Hexagon QNN graph binary
└── calibration.yaml            # Quantization stats
```

### 2.4 Test the Hub Runtime in Isolation

Before wiring into SnapSight:

```bash
qai-hub test models/object_detection/yolov8n-quant \
  --input models/object_detection/yolov8n-quant/sample_input_0.npy
```

Look for `Runtime: Hexagon NPU (QNN)` at the top of the output. If you see
`Runtime: Kryo CPU (fallback)` then one or more ops are not accelerated —
that's still functional but not maximally efficient.

### 2.5 Enable in SnapSight

In `.env`:

```
PREFERRED_BACKEND=AUTO        # or =QUALCOMM_AI_HUB to force it
FALLBACK_BACKEND=ONNX         # or =CPU
```

In `config/hardware.yaml`:

```yaml
device:
  vendor: Qualcomm
  platform: Snapdragon
  model: AUTO_DETECT
inference:
  preferred_backend: AUTO
  fallback_backend: CPU
models:
  object_detection: "object_detection/yolov8n-quant"
  speech_to_text:  "speech_to_text/whisper-tiny-quant"
  ocr:              "ocr/paddleocr-mobile-quant"
  vision_language:  "vision_language/mobilenet-vlm-quant"
```

---

## 3. Provider 2 — ONNXProvider (Fallback / x86 / Linux)

ONNX Runtime ("ORT") is the portable workhorse. On Snapdragon Windows it
targets the Adreno GPU via DirectML EP; on x86 it uses CPU EP or CUDA EP.

### 3.1 Install the Runtime

The base `onnxruntime==1.19.2` in `backend/requirements.txt` is the CPU EP.
For accelerated execution add one (not both) of:

```bash
# Windows (Snapdragon Adreno GPU or x86 iGPU/dGPU):
pip install onnxruntime-directml==1.19.2

# x86 Linux with NVIDIA GPU:
pip install onnxruntime-gpu==1.19.2
```

Verify the EP list:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expected (DirectML Windows): ['TensorrtExecutionProvider', 'CUDAExecutionProvider',
#  'DmlExecutionProvider', 'CPUExecutionProvider']
```

### 3.2 Obtain ONNX Weights

#### Option A — Export from official repos (recommended, reproducible)

```bash
# ----- YOLOv8n detection -----
pip install ultralytics==8.2.0
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=17 simplify
# Produces yolov8n.onnx — move it:
mv yolov8n.onnx models/object_detection/

# ----- Whisper-tiny STT -----
# Use huggingface optimum to export:
pip install optimum onnxruntime transformers accelerate
optimum-cli export onnx \
  --model openai/whisper-tiny \
  --task automatic-speech-recognition \
  --atol 1e-4 \
  models/speech_to_text/whisper-tiny-onnx/

# ----- PaddleOCR -----
# Follow Paddle's official onnx export guide and place:
#   models/ocr/ppocr_det.onnx
#   models/ocr/ppocr_rec.onnx

# ----- Moondream2 (lightweight VLM) -----
optimum-cli export onnx \
  --model vikhyatk/moondream2 \
  --task image-text-to-text \
  models/vision_language/moondream2-onnx/
```

#### Option B — Download community ONNX files directly

Trusted mirrors (pick files matching your architecture / opset):

- ONNX Model Zoo: https://github.com/onnx/models
- Hugging Face `onnx-community` org: https://huggingface.co/onnx-community
- Ultralytics assets page: https://github.com/ultralytics/assets/releases

### 3.3 Enable in SnapSight

`.env`:

```
PREFERRED_BACKEND=ONNX
FALLBACK_BACKEND=CPU
```

`config/hardware.yaml`:

```yaml
inference:
  preferred_backend: ONNX
  fallback_backend: CPU
models:
  object_detection: "object_detection/yolov8n.onnx"
  speech_to_text:  "speech_to_text/whisper-tiny-onnx/model.onnx"
  ocr:              "ocr/ppocr_det.onnx"         # see NOTE below
  vision_language:  "vision_language/moondream2-onnx/model.onnx"
```

> **NOTE on OCR:** Paddle-style OCR uses two nets (det + rec). When only one
> path is given in `hardware.yaml`, the `ONNXProvider` will look for a sibling
> `*_rec.onnx` next to `*_det.onnx` by naming convention. Explicit overrides
> can be supplied via `provider_options` in the YAML file.

### 3.4 Runtime Provider Selection Hints

By default `ONNXProvider` picks the best EP available in this order:
`QNN EP (if installed) → DirectML → CUDA → TensorRT → CPU`. Force a
specific EP via `ONNXRUNTIME_PROVIDERS` env var:

```bash
# Example: force DirectML even if QNN is present
set ONNXRUNTIME_PROVIDERS=DmlExecutionProvider,CPUExecutionProvider
```

---

## 4. Provider 3 — PyTorchProvider (Research / Dev)

Use this provider when iterating on model architectures before conversion to
ONNX / QNN. Performance is CPU-only on Snapdragon (the NPU is not exposed
through native PyTorch today).

### 4.1 Install

`torch`, `torchvision`, and `torchaudio` are already pinned in
`backend/requirements.txt`. Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 4.2 Acquire Weights

Load and save in native PyTorch format from the canonical source:

```bash
# YOLO via torch hub
python -c "
import torch
m = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
torch.save(m.state_dict(), 'models/object_detection/yolov5n.pt')
"

# Whisper tiny via openai-whisper
pip install openai-whisper
python -c "
import whisper
m = whisper.load_model('tiny')
m.save(str('models/speech_to_text/whisper-tiny.pt'))
"
```

For VLM / OCR, save your own `state_dict()` or use Hugging Face
`save_pretrained()` — `PyTorchProvider` understands both raw `.pt` files and
HF directory layouts.

### 4.3 Enable in SnapSight

`.env`:

```
PREFERRED_BACKEND=PYTORCH
FALLBACK_BACKEND=CPU
```

`config/hardware.yaml`:

```yaml
inference:
  preferred_backend: PYTORCH
  fallback_backend: CPU
models:
  object_detection: "object_detection/yolov5n.pt"
  speech_to_text:  "speech_to_text/whisper-tiny.pt"
  ocr:              ""                          # leave blank if no model
  vision_language:  ""
```

---

## 5. Provider 4 — MockProvider (Default, No Setup Needed)

`MockProvider` is SnapSight's **no-op, zero-dependency, always-works**
fallback. It ships deterministic fake outputs:

- **Detection:** Returns a single bounding box in the frame center labeled
  `"mock_object"` with score `0.99`.
- **STT:** Echoes transcribed audio length as `"[mock transcription of N samples]"`.
- **OCR:** Returns placeholder text `"Mock OCR output line 1\nMock OCR output line 2"`.
- **VLM:** Responds with `"This is a mock VLM response. Configure real weights in hardware.yaml."`.

### 5.1 When to Use It

- First boot / smoke testing (no model downloads required).
- Frontend development with live reload against a stable backend.
- CI unit tests for routers / WebSocket plumbing.
- Benchmarking the *overhead* of the stack without inference cost.

### 5.2 Enable

Set either explicitly:

```
PREFERRED_BACKEND=MOCK
```

or leave everything else misconfigured — MockProvider is always the last
fallback after all other `.probe()` calls fail.

---

## 6. Configuring `hardware.yaml`

### 6.1 Full Example (Snapdragon, Qualcomm AI Hub — All Tasks)

```yaml
device:
  vendor: Qualcomm
  platform: Snapdragon
  model: "Snapdragon X Elite X1E-80-100"

inference:
  preferred_backend: AUTO
  fallback_backend: CPU

models:
  object_detection: "object_detection/yolov8n-quant"
  speech_to_text:  "speech_to_text/whisper-tiny-quant"
  ocr:              "ocr/paddleocr-mobile-quant"
  vision_language:  "vision_language/mobilenet-vlm-quant"
```

### 6.2 Partial Example (x86 Dev Laptop — Detection Only)

```yaml
device:
  vendor: Intel
  platform: x86_64
  model: "Core Ultra 7 155H"

inference:
  preferred_backend: ONNX
  fallback_backend: MOCK

models:
  object_detection: "object_detection/yolov8n.onnx"
  speech_to_text:  ""
  ocr:              ""
  vision_language:  ""
```

### 6.3 Per-Task Provider Overrides (Advanced)

Need detection on the NPU but VLM via ONNX? Add a `provider_overrides` block:

```yaml
inference:
  preferred_backend: QUALCOMM_AI_HUB
  fallback_backend: CPU
  provider_overrides:
    # Keys match the top-level models.* keys above:
    vision_language: ONNX

models:
  object_detection: "object_detection/yolov8n-quant"
  speech_to_text:  "speech_to_text/whisper-tiny-quant"
  ocr:              "ocr/paddleocr-mobile-quant"
  vision_language:  "vision_language/moondream2-onnx/model.onnx"
```

The override accepts any provider enum value (`QUALCOMM_AI_HUB`, `ONNX`,
`PYTORCH`, `MOCK`).

---

## 7. Validation

After changing `hardware.yaml`, **restart the backend** and run:

```bash
# 1. Which provider loaded?
curl -s http://localhost:8000/api/v1/config/runtime | python -m json.tool
# Expected:
# {
#   "active_provider": "QualcommAIHubProvider",
#   "fallback_provider": "ONNXProvider",
#   "device": { "vendor": "Qualcomm", "platform": "Snapdragon", "model": "..." },
#   "loaded_tasks": [ "object_detection", "speech_to_text", "ocr", "vision_language" ]
# }

# 2. Does each loaded task accept a synthetic input?
# Detection (you need a test image; use the bundled samples or any JPEG):
curl -X POST http://localhost:8000/api/v1/detect \
  -F "image=@tests/fixtures/sample_car.jpg" \
  | python -m json.tool
# Expected 200 OK with a list of detections (even if empty array [])

# 3. WebSocket smoke test (camera optional)
python backend/scripts/ws_smoke_test.py --duration 3
```

If `loaded_tasks` in (1) is missing an entry you expected, check the backend
startup log at `LOG_LEVEL=INFO` — each `.probe()` failure prints a one-line
reason that points to the missing weight file or runtime import.

---

## 8. Switching Providers at Runtime

Providers are loaded **once per backend process** at app startup time. To
switch between them without editing YAML:

```bash
# Temporarily force a different backend via env var (takes precedence over YAML)
set SNAPSIGHT_PREFERRED_BACKEND=ONNX
set SNAPSIGHT_FALLBACK_BACKEND=MOCK
cd backend && uvicorn main:app --reload
```

Env var precedence order (highest first):

1. `SNAPSIGHT_PREFERRED_BACKEND`
2. `PREFERRED_BACKEND` (set in `.env`)
3. `inference.preferred_backend` (set in `config/hardware.yaml`)

This lets you run quick A/B benchmarks without touching config files.
