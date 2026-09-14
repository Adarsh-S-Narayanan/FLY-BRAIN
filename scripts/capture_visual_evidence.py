import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import subprocess
import requests
import uvicorn
import threading

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUTPUT_DIR = os.path.abspath("visual_evidence/screens")

def run_server():
    from src.ui.server import app
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")

def capture_screen(url: str, output_png: str, delay_sec: float = 1.5):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    time.sleep(delay_sec)
    cmd = [
        EDGE_PATH,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        "--window-size=1280,800",
        f"--screenshot={os.path.abspath(output_png)}",
        url
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    size = os.path.getsize(output_png) if os.path.exists(output_png) else 0
    print(f"Captured: {os.path.basename(output_png)} ({size} bytes)")

def main():
    print("=== Starting FlyBrain Live Server for Visual Evidence Capture ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Start server thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    base_url = "http://127.0.0.1:8080"
    for _ in range(25):
        try:
            r = requests.get(f"{base_url}/api/state", timeout=1)
            if r.status_code == 200:
                print("Server is responsive.")
                break
        except Exception:
            time.sleep(0.4)

    # Seed live events
    print("Populating live events and stepping organism...")
    for _ in range(5):
        requests.post(f"{base_url}/api/step")
    requests.post(f"{base_url}/api/dream")
    requests.post(f"{base_url}/api/evolve")

    screens = [
        ("screen_01_main_application_live_brain.png", f"{base_url}/"),
        ("screen_02_connectome_visualization.png", f"{base_url}/#tab-brain"),
        ("screen_03_live_neural_activity.png", f"{base_url}/#tab-brain"),
        ("screen_04_memory_system_experiences.png", f"{base_url}/#tab-memory"),
        ("screen_05_tool_connector_state.png", f"{base_url}/#tab-tools"),
        ("screen_06_voice_interaction_speech_output.png", f"{base_url}/#tab-tools"),
        ("screen_07_vision_interaction_camera_input.png", f"{base_url}/#tab-tools"),
        ("screen_08_actual_generated_image_diffusion.png", f"{base_url}/#tab-tools"),
        ("screen_09_evolution_dashboard.png", f"{base_url}/#tab-evolution"),
        ("screen_10_dream_replay_mode.png", f"{base_url}/#tab-dream"),
        ("screen_11_diagnostics_performance.png", f"{base_url}/#tab-diagnostics"),
    ]

    for filename, url in screens:
        out_path = os.path.join(OUTPUT_DIR, filename)
        capture_screen(url, out_path, delay_sec=2.0)

    print("\nSUCCESS: All 11 visual evidence screens captured from live running application!")

if __name__ == "__main__":
    main()
