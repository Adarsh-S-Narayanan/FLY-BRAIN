import os
import subprocess
import soundfile as sf
import numpy as np

def synthesize_speech(text: str, output_wav: str) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('{os.path.abspath(output_wav)}')
$synth.Speak('{text}')
$synth.Dispose()
"""
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], capture_output=True, text=True)
    return res.returncode == 0 and os.path.exists(output_wav)

def test_speech_system():
    print("=== Testing Local Speech Synthesis and Audio Analysis ===")
    wav_path = os.path.join("diagnostics", "test_speech_out.wav")
    success = synthesize_speech("FlyBrain neural connectome speech system active.", wav_path)
    assert success, "Speech synthesis failed!"
    
    file_size = os.path.getsize(wav_path)
    print(f"Generated WAV file: {wav_path} ({file_size} bytes)")
    assert file_size > 1000, f"WAV file too small: {file_size}"
    
    # Read with soundfile and compute audio features (VAD & energy)
    data, samplerate = sf.read(wav_path)
    duration = len(data) / samplerate
    rms_energy = float(np.sqrt(np.mean(data**2)))
    print(f"Audio duration: {duration:.2f}s, Samplerate: {samplerate}Hz, RMS Energy: {rms_energy:.4f}")
    assert duration > 0.5, f"Audio duration too short: {duration}"
    assert rms_energy > 0.01, f"Audio RMS energy too low: {rms_energy}"
    print("SUCCESS: Real local speech synthesis and audio reading verified!")

if __name__ == "__main__":
    test_speech_system()
