"""
voice/capture.py -- record audio from the microphone.

    audio = capture_audio(seconds=7)          # fixed length, with countdown
    audio = capture_audio_pyaudio(seconds=7)   # fallback if sounddevice misbehaves
    audio = record_until_silence()             # stop when the user stops talking (VAD)

All return a float32 mono numpy array, ready for voice.stt.transcribe().
(NEXUS `nexus_capture_audio()` equivalent.)
"""
import time

from core.conf import get


def capture_audio(seconds: int = None, sample_rate: int = None):
    """Record `seconds` of mono audio at `sample_rate` and return a float32 array.

    Prints a per-second countdown so the user knows how long is left.
    """
    import sounddevice as sd
    import numpy as np
    seconds = int(seconds or get("audio.seconds", 7))
    sample_rate = int(sample_rate or get("audio.sample_rate", 16000))

    print(f"🎙️  Recording for {seconds} seconds — speak now...")
    recording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                       channels=1, dtype="float32")
    if get("voice.countdown", True):
        for remaining in range(seconds, 0, -1):
            print(f"   {remaining}s remaining...", flush=True)
            time.sleep(1)
    sd.wait()
    print("✅ Recording finished.")
    return recording.reshape(-1).astype(np.float32)


def capture_audio_pyaudio(seconds: int = None, sample_rate: int = None):
    """PyAudio fallback recorder (use if sounddevice has driver issues)."""
    import pyaudio
    import numpy as np
    seconds = int(seconds or get("audio.seconds", 7))
    sample_rate = int(sample_rate or get("audio.sample_rate", 16000))
    chunk = 1024

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                     input=True, frames_per_buffer=chunk)
    print(f"🎙️  Recording for {seconds} seconds (PyAudio)...")
    frames = []
    for _ in range(int(sample_rate / chunk * seconds)):
        frames.append(stream.read(chunk, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("✅ Recording finished.")
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


def record_until_silence(max_seconds: float = 15.0, sample_rate: int = None):
    """Record until the speaker pauses (energy-based VAD). Good for natural turns."""
    import sounddevice as sd
    import numpy as np
    sample_rate = int(sample_rate or get("audio.sample_rate", 16000))
    thresh = get("voice.silence_threshold", 0.01)
    silence_max = get("voice.silence_max_seconds", 2.0)
    block = int(0.1 * sample_rate)

    frames, started, silent, elapsed = [], False, 0.0, 0.0
    print("🎙️  Listening... (stops automatically when you pause)")
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        while elapsed < max_seconds:
            data, _ = stream.read(block)
            mono = data.reshape(-1)
            frames.append(mono)
            level = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
            elapsed += block / sample_rate
            if level >= thresh:
                started, silent = True, 0.0
            elif started:
                silent += block / sample_rate
                if silent >= silence_max:
                    break
    print("✅ Recording finished.")
    return np.concatenate(frames).astype(np.float32) if frames else np.zeros(0, dtype=np.float32)


def list_devices():
    """Print available audio input devices (debug your mic before the exam)."""
    import sounddevice as sd
    print(sd.query_devices())
