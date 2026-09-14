@echo off
title FlyBrain Autonomous Organism Dashboard
echo ========================================================
echo   FlyBrain: Vulkan-First Biological Connectome System   
echo   MaleCNS Dataset Integration (Janelia FlyEM)           
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Error: Python virtual environment not found in .venv.
    echo Please run 'uv venv .venv --python 3.12' first.
    pause
    exit /b 1
)

echo Launching FlyBrain Organism System on AMD Radeon GPU / Vulkan...
.venv\Scripts\python.exe src\main.py --mode run --host 127.0.0.1 --port 8080
pause
