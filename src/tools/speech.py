import os
import subprocess
import soundfile as sf
import numpy as np
from typing import Dict, Any
from src.tools.base import ToolConnector

class SpeakConnector(ToolConnector):
    def __init__(self, audio_dir: str = "visual_evidence/audio"):
        super().__init__(
            name="speak",
            description="Synthesizes and speaks text audio using local Windows SAPI TTS and records WAV artifact.",
            timeout_sec=15.0
        )
        self.audio_dir = audio_dir
        os.makedirs(self.audio_dir, exist_ok=True)

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "play_sound": {"type": "boolean", "default": False}
            },
            "required": ["text"]
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text_spoken": {"type": "string"},
                "wav_file": {"type": "string"},
                "duration_sec": {"type": "number"},
                "rms_energy": {"type": "number"}
            },
            "required": ["text_spoken", "wav_file", "duration_sec", "rms_energy"]
        }

    def _execute(self, params: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        text = str(params["text"])
        wav_filename = f"speech_{execution_id[:8]}.wav"
        wav_path = os.path.abspath(os.path.join(self.audio_dir, wav_filename))

        # Windows PowerShell SAPI script
        ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('{wav_path}')
$synth.Speak('{text}')
$synth.Dispose()
"""
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=self.timeout_sec
        )
        if proc.returncode != 0 or not os.path.exists(wav_path):
            raise RuntimeError(f"Speech synthesis failed: {proc.stderr}")

        # Measure generated audio
        data, sr = sf.read(wav_path)
        duration = float(len(data) / sr)
        rms = float(np.sqrt(np.mean(data**2)))

        return {
            "text_spoken": text,
            "wav_file": wav_path,
            "duration_sec": round(duration, 3),
            "rms_energy": round(rms, 4)
        }
