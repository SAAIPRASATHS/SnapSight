# ./models/ — Model Weights Directory

This directory stores **on-device AI model weight files** used by SnapSight's
`ModelProvider` implementations. Nothing in here should ever be committed to
Git — the root `.gitignore` is already configured to block the most common
weight-file extensions.

## Expected Layout

Organize weights by task. The `config/hardware.yaml` `models.*` keys reference
individual files (or folders for multi-file Hub packages) inside this tree.

```
models/
├── README.md                 ← this file
├── object_detection/
│   └── yolov8n-quant/        ← Qualcomm AI Hub packaged model (folder)
│   └── yolov8n.onnx          ← generic ONNX weight (file)
├── speech_to_text/
│   └── whisper-tiny-quant/   ← Hub folder
│   └── whisper-tiny.pt       ← PyTorch checkpoint
├── ocr/
│   └── paddleocr-mobile-quant/
│   └── ppocr_det.onnx
│   └── ppocr_rec.onnx
└── vision_language/
    └── mobilenet-vlm-quant/
    └── minigpt4.onnx
```

You do **not** need all four tasks populated. Leave the corresponding
`hardware.yaml` key as `""` and SnapSight will simply not expose that capability
in the UI.

---

## 1. Downloading Qualcomm AI Hub Models (Recommended for Snapdragon)

The `qai-hub` CLI ships optimized, Hexagon-NPU-ready variants of the most
popular models. This is the path you want on a Snapdragon HP PC.

### Prerequisites

- Python 3.10 – 3.12 on Windows 11 (Snapdragon or x86).
- Pip access to the official Qualcomm AI Hub PyPI index.

### Install the Hub CLI

```bash
pip install --upgrade qai-hub
```

Verify installation:

```bash
qai-hub --version
```

### Search the Catalog

```bash
qai-hub search yolo
qai-hub search whisper
qai-hub search ocr
qai-hub search "vision language"
```

### Download into This Folder

Run each download **from the repository root** so the paths line up with
`hardware.yaml`:

```bash
# Object detection — YOLOv8n, INT8 quantized for Hexagon
qai-hub download object-detection/yolov8n-quant \
  --output models/object_detection/yolov8n-quant

# Speech-to-text — Whisper Tiny, INT8
qai-hub download speech/whisper-tiny-quant \
  --output models/speech_to_text/whisper-tiny-quant

# OCR — PaddleOCR mobile, INT8
qai-hub download ocr/paddleocr-mobile-quant \
  --output models/ocr/paddleocr-mobile-quant

# Vision-language model — MobileVLM, INT8
qai-hub download vlm/mobilenet-vlm-quant \
  --output models/vision_language/mobilenet-vlm-quant
```

Each command produces a **folder** (not a single file). After download, set
`hardware.yaml` to point at the folder, e.g.:

```yaml
models:
  object_detection: "object_detection/yolov8n-quant"
```

---

## 2. Downloading Generic ONNX Models (Fallback / x86 / Linux)

If you don't have access to the Qualcomm AI Hub (e.g. running on a dev laptop
without Hexagon hardware), drop standard `.onnx` files into the matching
subfolder. Trusted sources:

| Task | Source | Direct link / command |
|---|---|---|
| Object Detection (YOLOv8n) | Ultralytics | `pip install ultralytics && yolo export model=yolov8n.pt format=onnx imgsz=640` → move `yolov8n.onnx` to `models/object_detection/` |
| Speech-to-Text (Whisper-tiny) | Hugging Face | https://huggingface.co/onnx-community/whisper-tiny → download `model.onnx` as `models/speech_to_text/whisper-tiny.onnx` |
| OCR (PaddleOCR) | PaddlePaddle | https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/deploy/paddle2onnx/README_en.md → export `det` + `rec` ONNX files into `models/ocr/` |
| Vision-Language | Moondream / Hugging Face | https://huggingface.co/vikhyatk/moondream2 → export to ONNX and place in `models/vision_language/` |

Set `hardware.yaml` to point at the `.onnx` file directly, e.g.:

```yaml
models:
  object_detection: "object_detection/yolov8n.onnx"
```

---

## 3. Downloading PyTorch Models (Development / Research)

PyTorch checkpoints work via `PyTorchProvider` on CPU, CUDA (discrete GPUs), or
MPS (Apple Silicon). They are the least optimized path on Snapdragon and exist
mostly for model authors to test new architectures before converting them to
ONNX or QNN.

```bash
# Example: YOLOv8n via torch hub. From ./models/object_detection/:
python -c "import torch; torch.hub.load('ultralytics/yolov8', 'yolov8n').save('yolov8n.pt')"
```

```yaml
models:
  object_detection: "object_detection/yolov8n.pt"
```

---

## .gitignore Note

The **repository root** `.gitignore` already excludes:

```
models/*.onnx
models/*.pt
models/*.pth
models/*.bin
```

This catches most flat weight files. For multi-file models (especially Qualcomm
AI Hub folders that contain `model.json`, `*.so` / `*.dll` context binaries,
calibration data, etc.) **add a line per folder** to the root `.gitignore` if
you want to be extra safe, e.g.:

```
models/object_detection/yolov8n-quant/
models/speech_to_text/whisper-tiny-quant/
models/ocr/paddleocr-mobile-quant/
models/vision_language/mobilenet-vlm-quant/
```

Alternatively, add `models/*/` and keep only this `README.md` tracked.
