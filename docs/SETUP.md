# SETUP GUIDE — SnapSight AI

Step-by-step environment setup for **Windows (Snapdragon / x86)** and
**generic Linux** dev machines. Assumes you have cloned the repo and are
working inside the `snapsight-ai/` directory.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Windows Setup (Snapdragon or x86)](#2-windows-setup-snapdragon-or-x86)
3. [Linux Setup (Ubuntu 22.04 / Debian 12)](#3-linux-setup-ubuntu-2204--debian-12)
4. [Verifying the Install](#4-verifying-the-install)
5. [First Run (Smoke Test)](#5-first-run-smoke-test)
6. [Troubleshooting Common Issues](#6-troubleshooting-common-issues)

---

## 1. Prerequisites

### Hardware

| Tier | Recommended | Minimum |
|---|---|---|
| Target device | HP OmniBook Ultra / EliteBook with **Snapdragon X Elite / X Plus** | Any Snapdragon 8cx Gen 3+ Windows PC |
| Dev machine | x86-64 Windows 11 / Ubuntu 22.04, 16 GB RAM, 50 GB free SSD | 8 GB RAM, 20 GB free |

### Software (Both Platforms)

- **Python 3.10, 3.11, or 3.12** (3.13 not yet supported by `qai-hub` and some
  ONNX packages).
- **Node.js 20 LTS** (Iron) or **22 LTS** (Jod).
- **Git 2.40+**.
- On Windows: **PowerShell 7** recommended, but built-in PowerShell 5 works.

---

## 2. Windows Setup (Snapdragon or x86)

The steps below work identically on ARM64 Windows (Snapdragon) and x64 Windows
(traditional laptops). The only difference is which Python / Node installers
you grab — use the ARM64 builds on Snapdragon.

### 2.1 Install Python

1. Download Python **3.11 ARM64** (Snapdragon) or **3.11 x64** (x86) from
   https://www.python.org/downloads/release/python-3119/
2. Run the installer. **CHECK** ☑ *Add python.exe to PATH* on the first screen,
   then click **Install Now**.
3. Open a fresh PowerShell window and verify:
   ```powershell
   python --version
   # Expected: Python 3.11.x
   python -c "import platform; print(platform.machine())"
   # Expected on Snapdragon: ARM64
   # Expected on x86:        AMD64
   ```

### 2.2 Install Node.js

1. Grab the **LTS** `.msi` for your architecture from
   https://nodejs.org/en/download — ARM64 for Snapdragon, x64 for x86.
2. Install with defaults.
3. Verify:
   ```powershell
   node --version   # v20.x or v22.x
   npm  --version   # 10.x
   ```

### 2.3 Create + Activate the Python Virtual Environment

```powershell
# From the repository root (the folder that contains README.md and .env.example):
cd d:\snap\snapsight-ai

python -m venv .venv

# Activate (note: ExecutionPolicy may prompt once; accept)
.\.venv\Scripts\Activate.ps1

# Prompt should now be prefixed with (.venv)
```

> 💡 **If activation fails with "ExecutionPolicy"**: Run once per machine:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> then retry activation.

### 2.4 Install Backend Dependencies

```powershell
# With the venv still active:
pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt
```

Optional but recommended for Snapdragon targets:

```powershell
pip install qai-hub
qai-hub --version
```

### 2.5 Install Frontend Dependencies

> This step assumes the `frontend/` folder exists with a `package.json`. If
> you haven't scaffolded the frontend yet, come back after you create it.

```powershell
cd frontend
npm install
cd ..
```

### 2.6 Configure `.env`

```powershell
Copy-Item .env.example .env
# Edit .env now in VS Code / Notepad. For a default local install the values
# in .env.example are already fine.
```

---

## 3. Linux Setup (Ubuntu 22.04 / Debian 12)

Linux is supported for **development and x86 inference via ONNX / PyTorch**.
*Qualcomm AI Hub (Hexagon NPU) acceleration is Windows-only on Snapdragon PCs
today; on Linux Snapdragon laptops use `ONNXProvider` with `CPU EP`.*

### 3.1 System Dependencies

```bash
sudo apt update
sudo apt install -y --no-install-recommends \
    build-essential \
    python3.11 python3.11-venv python3.11-dev \
    nodejs npm \
    git \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgl1 libgthread-2.0-0 libpango-1.0-0 libcairo2 \
    libxcb-cursor0 libxkbcommon0 libgbm1 libasound2
```

If `nodejs` is too old (< 20):

```bash
# Install Node 22 via nodesource:
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

### 3.2 Virtual Environment

```bash
cd ~/snap/snapsight-ai        # or wherever you cloned

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt
```

### 3.3 Frontend

```bash
cd frontend && npm install && cd ..
```

### 3.4 `.env`

```bash
cp .env.example .env
$EDITOR .env
```

### 3.5 Linux Webcam Permissions

Ensure your user is in the `video` group so OpenCV can open `/dev/video0`:

```bash
sudo usermod -aG video $USER
newgrp video        # apply immediately without logout
```

---

## 4. Verifying the Install

Run these checks **with the Python venv active** from the repo root. Expected
output is shown inline.

```bash
# 1. Python imports
python -c "
import fastapi, uvicorn, pydantic, cv2, onnxruntime, torch, yaml
import numpy, PIL, psutil, httpx, aiofiles
print('✅  All core backend packages import OK')
"

# 2. qai-hub (if installed — Snapdragon / Windows only; skip on Linux)
qai-hub --version
# Expected: qai-hub, version 2.x or newer

# 3. Frontend binaries
node -v && npm -v
# Expected: v20+ / 10+

# 4. Config file parses
python -c "
import yaml
with open('config/hardware.yaml') as f:
    cfg = yaml.safe_load(f)
assert cfg['device']['vendor'] == 'Qualcomm'
print('✅  hardware.yaml parses OK')
"
```

Any failure in step 1 means a dependency did not install — head to the
[Troubleshooting section](#6-troubleshooting-common-issues).

---

## 5. First Run (Smoke Test)

Start two terminal windows — **one for backend, one for frontend**. Both need
the venv active.

### Terminal 1: Backend

```bash
# Repo root, venv active
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Wait for `Uvicorn running on http://0.0.0.0:8000`. In a browser, visit
http://localhost:8000/health — you should see:

```json
{"status":"ok","provider":"MockProvider","uptime_seconds":...}
```

The provider shows `MockProvider` until you configure real models in
`hardware.yaml` — that's expected for a smoke test.

### Terminal 2: Frontend

```bash
# Repo root
cd frontend
npm run dev
```

Visit http://localhost:5173. The SnapSight dashboard should load, camera
permissions should prompt, and the top bar should show a green
"Backend: online" indicator.

---

## 6. Troubleshooting Common Issues

### 6.1 `pip install -r backend/requirements.txt` fails

| Symptom | Fix |
|---|---|
| `opencv-python` / `numpy` wheel build fails on Windows ARM64 | Use the **latest Python 3.11** installer (3.11.8+ has full ARM64 wheels). Upgrade pip: `python -m pip install --upgrade pip` before retrying. |
| `torch` cannot find a matching distribution on ARM64 Linux | PyTorch only publishes ARM64 Linux wheels for Python 3.10–3.11. Pin: `pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu` |
| `onnxruntime` fails to load shared libs on Debian headless | `sudo apt install libgomp1 libprotobuf-dev` |

### 6.2 Backend starts, but `/health` shows `provider=MockProvider` unexpectedly

Most likely cause: the weight path in `hardware.yaml` is empty or wrong.

```bash
python -c "
import yaml, os
cfg = yaml.safe_load(open('config/hardware.yaml'))
for task, path in cfg['models'].items():
    full = os.path.join('models', path) if path else '(unset)'
    ok = path and os.path.exists(full)
    print(f'{task:20s} path={full:55s} exists={ok}')
"
```

For each `exists=False`, re-download the model per [docs/MODEL_SETUP.md](MODEL_SETUP.md)
and re-check `hardware.yaml`.

### 6.3 Frontend cannot reach backend (CORS, connection refused, etc.)

- **`BACKEND_HOST=0.0.0.0` in `.env`** is required so uvicorn binds to all
  interfaces. `127.0.0.1` works too but only for same-machine access.
- **Check ports are free**:
  ```powershell
  # Windows
  netstat -ano | findstr ":8000"
  netstat -ano | findstr ":5173"
  ```
  ```bash
  # Linux
  ss -tlnp | grep -E ':(8000|5173)'
  ```
  If another process holds the port either kill it (`taskkill /PID xxxx /F`
  on Windows, `kill xxxx` on Linux) or pick different ports in `.env`.
- **CORS**: by default the backend allows `http://localhost:5173` and
  `http://127.0.0.1:5173`. If you run the frontend on a different origin, add
  it to the `CORS_ORIGINS` list in `backend/main.py`.

### 6.4 Camera is black / `CAMERA_INDEX=0` not found

- Open the Windows Camera app or Linux `cheese` first to confirm the hardware
  works at the OS level.
- Try `CAMERA_INDEX=1` (and increment up to 4) in `.env` — some laptops
  expose the IR camera at index 0.
- On Windows ARM64, some USB UVC cameras need the manufacturer's ARM64
  driver; the built-in inbox class driver works for integrated webcams only.
- On Linux, double-check group membership: `groups $USER | grep video`.

### 6.5 `qai-hub download` fails with "network" or "403"

- Corporate proxies: set `HTTPS_PROXY=http://proxy.corp:8080` in your shell
  before running the command.
- Offline Snapdragon target: download on an internet-connected dev machine
  (x86 is fine — the weight packages are architecture-agnostic) and then copy
  the `models/` folder across via USB / SMB.

### 6.6 NPU not utilized even with Qualcomm AI Hub installed

Confirm the runtime picked the right provider:

```bash
curl -s http://localhost:8000/api/v1/config/runtime | python -m json.tool
```

If it reports `QualcommAIHubProvider` but NPU utilization in Task Manager is
flat, the model you downloaded may not yet have Hexagon kernels for every
operator and is silently falling back to CPU per-op. Use a Hub *"-quant"*
model (INT8) and retry — those have the highest NPU coverage.

### 6.7 STT audio is garbled on Linux

- `PulseAudio` + `ALSA` plug-in stack can resample. Record a 5 s clip via
  `arecord -f S16_LE -r 16000 -c 1 test.wav` and play it back with `aplay`.
  If it sounds wrong, update `pavucontrol` configuration profile.
- SnapSight's frontend expects 16 kHz s16le mono. Verify the browser's
  `MediaRecorder` is configured that way in `frontend/src/hooks/useAudioLoop.ts`.

### 6.8 Vite "WebSocket HMR connection failed" banner

Non-fatal for SnapSight function but annoying in dev. Either:
- Allow the Vite HMR port through Windows Firewall (private network profile),
  or
- Use `npm run build + npm run preview` instead of `npm run dev`.

### 6.9 `ModuleNotFoundError: No module named ...` after `git pull`

Dependencies drift. Re-sync:

```bash
pip install --upgrade -r backend/requirements.txt
cd frontend && npm install && cd ..
```
