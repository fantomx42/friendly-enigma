import os
import subprocess
import sys
import time
import numpy as np

# Try importing dependencies
try:
    import whisper
    import sounddevice as sd
    from scipy.io.wavfile import write
except ImportError:
    print("Voice dependencies missing. Run ./setup_voice.sh")
    whisper = None
    sd = None

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "voice_models")
PIPER_BIN = os.path.join(MODELS_DIR, "piper")
VOICE_MODEL = os.path.join(MODELS_DIR, "en_US-lessac-medium.onnx")

class VoiceManager:
    def __init__(self):
        self.enabled = False
        if os.path.exists(PIPER_BIN) and whisper is not None:
            self.enabled = True
            print("[Voice] Loading Whisper Model (base)...")
            self.stt_model = whisper.load_model("base")
            print("[Voice] Whisper Loaded.")
        else:
            print("[Voice] Voice disabled. Prerequisites missing.")

    def speak(self, text: str):
        """Synthesizes speech using Piper."""
        if not self.enabled:
            print(f"[Voice] (TTS Disabled) Ralph says: {text}")
            return

        try:
            # echo "text" | ./piper --model ... --output_file /tmp/say.wav
            cmd = [
                PIPER_BIN,
                "--model", VOICE_MODEL,
                "--output_file", "/tmp/ralph_speak.wav"
            ]
            
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
            process.communicate(input=text)
            
            # Play the audio (aplay for Linux)
            subprocess.run(["aplay", "-q", "/tmp/ralph_speak.wav"])
            
        except Exception as e:
            print(f"[Voice] TTS Error: {e}")

    def listen(self, duration: int = 5) -> str:
        """Records audio for N seconds and transcribes it."""
        if not self.enabled:
            return input("[Voice Disabled] Type your command: ")

        print(f"[Voice] Listening for {duration}s...")
        fs = 44100  # Sample rate
        
        try:
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()  # Wait until recording is finished
            
            # Save to temp file
            from scipy.io.wavfile import write
            write("/tmp/ralph_listen.wav", fs, recording)
            
            # Transcribe
            result = self.stt_model.transcribe("/tmp/ralph_listen.wav")
            text = result["text"].strip()
            print(f"[Voice] Heard: '{text}'")
            return text
            
        except Exception as e:
            print(f"[Voice] STT Error: {e}")
            return ""

# Global Instance
voice = VoiceManager()
