"""
voice/stt.py -- speech-to-text with Whisper (and a SpeechRecognition fallback).

The Whisper model is loaded ONCE and cached (loading is slow + memory heavy).

    res = transcribe(audio_array, sample_rate)   # -> {'text','language','confidence'}
    res = transcribe_file("clip.mp3")            # wav/mp3 file -> same dict
    txt = transcribe_sr(audio_array)             # via SpeechRecognition (offline whisper)

(NEXUS `nexus_transcribe()` equivalent. Confidence = 'high' if text is long enough.)
"""
from core.conf import get

_MODEL = None
_MODEL_SIZE = None


def load_model(size: str = None):
    """Load (and cache) the Whisper model. Only sizes you cached at home work offline."""
    global _MODEL, _MODEL_SIZE
    import whisper
    size = size or get("whisper.model_size", "base")
    if _MODEL is None or _MODEL_SIZE != size:
        print(f"[whisper] loading '{size}' model (first time is slow)...")
        _MODEL = whisper.load_model(size)
        _MODEL_SIZE = size
    return _MODEL


def _confidence(text: str) -> str:
    return "high" if len(text) > get("voice.confidence_min_chars", 10) else "low"


def transcribe(audio_array, sample_rate: int = None) -> dict:
    """Transcribe a numpy audio array. Returns {'text','language','confidence'}."""
    from voice.audio_io import to_float32_mono_16k
    sample_rate = int(sample_rate or get("audio.sample_rate", 16000))
    audio, _ = to_float32_mono_16k(audio_array, sample_rate, 16000)

    model = load_model()
    result = model.transcribe(audio, language=get("whisper.language"),
                              fp16=get("whisper.fp16", False))
    text = (result.get("text") or "").strip()
    return {"text": text, "language": result.get("language", "?"),
            "confidence": _confidence(text)}


def transcribe_file(path: str) -> dict:
    """Transcribe a wav/mp3 file directly (Whisper handles decoding via ffmpeg)."""
    model = load_model()
    result = model.transcribe(path, language=get("whisper.language"),
                              fp16=get("whisper.fp16", False))
    text = (result.get("text") or "").strip()
    return {"text": text, "language": result.get("language", "?"),
            "confidence": _confidence(text)}


def transcribe_sr(audio_array, sample_rate: int = None) -> str:
    """Alternative STT via the SpeechRecognition package using its OFFLINE whisper
    engine (never recognize_google -- that needs internet)."""
    import speech_recognition as sr_lib
    import numpy as np
    sample_rate = int(sample_rate or get("audio.sample_rate", 16000))
    pcm = (np.clip(np.asarray(audio_array, dtype="float32"), -1, 1) * 32767).astype("int16")
    audio = sr_lib.AudioData(pcm.tobytes(), sample_rate, 2)
    recognizer = sr_lib.Recognizer()
    return recognizer.recognize_whisper(audio, model=get("whisper.model_size", "base"))


def transcribe_sr_from_mic() -> str:
    """Listen on the mic via SpeechRecognition and transcribe offline with whisper."""
    import speech_recognition as sr_lib
    recognizer = sr_lib.Recognizer()
    with sr_lib.Microphone(sample_rate=get("audio.sample_rate", 16000)) as source:
        print("🎙️  Listening (SpeechRecognition)...")
        audio = recognizer.listen(source)
    return recognizer.recognize_whisper(audio, model=get("whisper.model_size", "base"))


def detect_language(audio_array, sample_rate: int = None) -> str:
    """Return the detected language code (delegates to transcribe for robustness)."""
    return transcribe(audio_array, sample_rate)["language"]
