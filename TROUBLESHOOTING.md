# FlyBrain Troubleshooting Guide

## Common Issues & Diagnoses

### 1. Vulkan Device Initialization Failure
- **Symptom**: `RuntimeError: No Vulkan physical devices found!`
- **Cause**: The AMD Radeon graphics driver is disabled or Vulkan runtime is not installed.
- **Fix**: Check `vulkaninfo.exe --summary` in terminal. Verify that `C:\Windows\System32\vulkan-1.dll` exists. If running on a system without a Vulkan GPU, FlyBrain automatically falls back to the high-performance CPU reference engine.

### 2. Missing Virtual Environment
- **Symptom**: `Error: Python virtual environment not found in .venv.`
- **Cause**: `.venv` has not been set up.
- **Fix**: Run:
  ```powershell
  uv venv .venv --python 3.12
  uv pip install numpy scipy pillow sounddevice soundfile vulkan fastapi uvicorn websockets diffusers transformers accelerate torch torchvision
  ```

### 3. Speech Synthesis Audio Output
- **Symptom**: Speech synthesis fails to write WAV file.
- **Cause**: PowerShell execution policy restricted or SAPI audio device busy.
- **Fix**: Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or ensure the target output folder has write permissions.

### 4. Port 8080 Conflict
- **Symptom**: `OSError: [WinError 10048] Only one usage of each socket address is normally permitted`
- **Fix**: Run on a different port using `python src/main.py --mode run --port 8081`.
