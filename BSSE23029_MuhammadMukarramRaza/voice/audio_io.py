"""
voice/audio_io.py -- the audio FORMAT plumbing everything else depends on.

The #1 cause of garbage transcripts is feeding Whisper the wrong format.
Whisper wants:  float32, mono, 16 kHz, samples in [-1, 1].
gradio's mic gives either (sample_rate, int16 ndarray) OR a temp filepath.
These helpers normalise all of that.

    audio, sr = from_gradio(gr_audio)        # handles BOTH gradio formats
    audio, sr = to_float32_mono_16k(data, sr)
    audio, sr = load_audio("clip.mp3")       # wav or mp3 (mp3 needs ffmpeg)
    save_wav("out.wav", audio, sr)
"""
from core.conf import get


def _resample(data, sr: int, target_sr: int):
    """Linear resample with numpy only (scipy is not in the install list)."""
    import numpy as np
    if sr == target_sr or len(data) == 0:
        return np.asarray(data, dtype="float32")
    n = int(round(len(data) * target_sr / sr))
    if n <= 0:
        return np.zeros(0, dtype="float32")
    x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, data).astype("float32")


def to_float32_mono_16k(data, sr: int, target_sr: int = 16000):
    """Convert any int/float, mono/stereo array at `sr` -> float32 mono [-1,1] @ target_sr."""
    import numpy as np
    a = np.asarray(data)
    # dtype -> float32 in [-1, 1]
    if a.dtype == np.int16:
        a = a.astype("float32") / 32768.0
    elif a.dtype == np.int32:
        a = a.astype("float32") / 2147483648.0
    elif a.dtype == np.uint8:
        a = (a.astype("float32") - 128.0) / 128.0
    else:
        a = a.astype("float32")
    # stereo -> mono
    if a.ndim == 2:
        a = a.mean(axis=1)
    a = _resample(a, sr, target_sr)
    return np.clip(a, -1.0, 1.0).astype("float32"), target_sr


def from_gradio(audio, target_sr: int = 16000):
    """Normalise gradio audio (either (sr, ndarray) or a filepath str) -> (float32, sr)."""
    import numpy as np
    if audio is None:
        return np.zeros(0, dtype="float32"), target_sr
    if isinstance(audio, str):
        return load_audio(audio, target_sr)
    if isinstance(audio, (tuple, list)) and len(audio) == 2:
        sr, data = audio
        return to_float32_mono_16k(data, sr, target_sr)
    raise ValueError(f"Unrecognised gradio audio format: {type(audio)}")


def load_audio(path: str, target_sr: int = 16000):
    """Load wav/flac/ogg (soundfile) or mp3/m4a (pydub+ffmpeg) -> (float32 mono, sr)."""
    import numpy as np
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        return to_float32_mono_16k(data, sr, target_sr)
    except Exception:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path)
        data = np.array(seg.get_array_of_samples())
        if seg.channels == 2:
            data = data.reshape((-1, 2))
        maxv = float(1 << (8 * seg.sample_width - 1))
        data = data.astype("float32") / maxv
        return to_float32_mono_16k(data, seg.frame_rate, target_sr)


def save_wav(path: str, audio, sr: int = 16000) -> str:
    """Write a float32 array to a WAV file (soundfile)."""
    import soundfile as sf
    import numpy as np
    sf.write(path, np.asarray(audio, dtype="float32"), sr)
    return path


def save_mp3(path: str, audio, sr: int = 16000) -> str:
    """Write a float32 array to MP3 (pydub -> needs ffmpeg installed)."""
    import numpy as np
    from pydub import AudioSegment
    pcm = (np.clip(np.asarray(audio, dtype="float32"), -1, 1) * 32767).astype("int16")
    seg = AudioSegment(pcm.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    seg.export(path, format="mp3")
    return path


def rms(audio) -> float:
    """Root-mean-square loudness (for silence detection)."""
    import numpy as np
    a = np.asarray(audio, dtype="float32")
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def normalize(audio):
    """Scale so the loudest sample is 1.0 (avoids clipping / quiet recordings)."""
    import numpy as np
    a = np.asarray(audio, dtype="float32")
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    return (a / peak).astype("float32") if peak > 0 else a


def trim_silence(audio, sr: int = 16000, thresh: float = None):
    """Trim leading/trailing quiet samples below `thresh` RMS (windowed)."""
    import numpy as np
    a = np.asarray(audio, dtype="float32")
    if a.size == 0:
        return a
    thresh = get("voice.silence_threshold", 0.01) if thresh is None else thresh
    win = max(1, int(0.02 * sr))
    energy = np.array([np.sqrt(np.mean(a[i:i + win] ** 2)) for i in range(0, len(a), win)])
    loud = np.where(energy >= thresh)[0]
    if len(loud) == 0:
        return a
    start = loud[0] * win
    end = min(len(a), (loud[-1] + 1) * win)
    return a[start:end]


def play(audio, sr: int = 16000):
    """Play a float32 array through the speakers (sounddevice)."""
    import sounddevice as sd
    import numpy as np
    sd.play(np.asarray(audio, dtype="float32"), sr)
    sd.wait()
