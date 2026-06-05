"""
vision/backend.py  --  backend detection and model path resolution.

Priority order when backend = "auto":
    1. mediapipe   (if installed)
    2. dnn         (if YuNet model file exists)
    3. opencv      (LBP cascades — always available)

Every function here is safe to call even if a backend isn't available.
Import errors and missing files are caught and logged once, then remembered.

Usage:
    from vision.backend import BACKEND, yunet_path, fer_path, lbp_cascade
"""

import os
from functools import lru_cache
from pathlib import Path

from core.conf import get

# ── model file locations ──────────────────────────────────────────────────────
_MODELS_DIR = Path(__file__).parent.parent / "models"
_YUNET_FILE  = _MODELS_DIR / "yunet.onnx"
_FER_FILE    = _MODELS_DIR / "emotion_ferplus.onnx"


def yunet_path() -> str | None:
    return str(_YUNET_FILE) if _YUNET_FILE.exists() else None


def fer_path() -> str | None:
    return str(_FER_FILE) if _FER_FILE.exists() else None


# ── availability checks (run once, cached) ────────────────────────────────────

@lru_cache(maxsize=1)
def mediapipe_available() -> bool:
    """
    Check whether mediapipe is importable.

    On Apple Silicon, mediapipe bundles its own libopencv which conflicts with
    our cv2 at the Objective-C class registration level. Importing cv2 FIRST
    (so it wins the class registration) prevents the conflict and lets the
    mediapipe import succeed. We also catch all exceptions, not just ImportError,
    because the conflict can surface as OSError / RuntimeError on the first load.
    """
    try:
        import cv2          # force cv2 to load first → wins objc class registration
        import mediapipe    # noqa: F401
        return True
    except Exception:       # catches ImportError, OSError, RuntimeError from lib conflict
        return False


@lru_cache(maxsize=1)
def yunet_available() -> bool:
    if not _YUNET_FILE.exists():
        return False
    try:
        import cv2
        # FaceDetectorYN was added in OpenCV 4.5.4
        return hasattr(cv2, "FaceDetectorYN")
    except Exception:
        return False


@lru_cache(maxsize=1)
def fer_available() -> bool:
    if not _FER_FILE.exists():
        return False
    try:
        import cv2
        return True
    except Exception:
        return False


# ── backend selection ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def resolve_backend() -> str:
    """
    Return the effective backend string.
    Reads vision.backend from config; resolves "auto" to best available.
    Result is cached — call reset_backend_cache() if config changes at runtime.
    """
    pref = get("vision.backend", "auto")

    if pref == "mediapipe":
        if mediapipe_available():
            return "mediapipe"
        print("[vision] mediapipe requested but not installed — falling back to dnn/opencv")

    if pref == "dnn":
        if yunet_available():
            return "dnn"
        print("[vision] dnn requested but yunet.onnx not found — falling back to opencv")

    if pref == "opencv":
        return "opencv"

    # "auto" — pick best available
    if mediapipe_available():
        return "mediapipe"
    if yunet_available():
        return "dnn"
    return "opencv"


def reset_backend_cache():
    resolve_backend.cache_clear()
    mediapipe_available.cache_clear()
    yunet_available.cache_clear()
    fer_available.cache_clear()


# ── module-level BACKEND constant ─────────────────────────────────────────────
# On Apple Silicon, mediapipe import can fail at first try (objc class conflict)
# but succeed after cv2 is loaded. We clear the cache and re-resolve once so
# that the stale "dnn" result from a failed first attempt is corrected.
reset_backend_cache()           # clear any stale first-import cache result
BACKEND: str = resolve_backend() # re-resolve with cv2 already in sys.modules

# If auto-resolution still picked dnn but mediapipe is actually available
# (can happen when mediapipe import is a no-op on retry), force a final check.
if BACKEND != "mediapipe" and get("vision.backend", "auto") == "auto":
    if mediapipe_available():
        resolve_backend.cache_clear()
        BACKEND = resolve_backend()


# ── LBP cascade helper ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def lbp_face_cascade():
    """LBP frontal-face cascade — 3-5× faster than Haar at similar accuracy."""
    import cv2
    # lbpcascade_frontalface_improved is the best LBP option in OpenCV
    candidates = [
        "lbpcascade_frontalface_improved.xml",
        "lbpcascade_frontalface.xml",
        "haarcascade_frontalface_alt2.xml",   # fallback to a better Haar
        "haarcascade_frontalface_default.xml",
    ]
    for name in candidates:
        path = os.path.join(cv2.data.haarcascades, name)
        if os.path.exists(path):
            clf = cv2.CascadeClassifier(path)
            if not clf.empty():
                return clf
    return None


def print_backend_summary():
    print(f"[vision] backend      : {BACKEND}")
    print(f"[vision] mediapipe    : {mediapipe_available()}")
    print(f"[vision] yunet.onnx   : {yunet_available()} ({yunet_path() or 'not found'})")
    print(f"[vision] fer model    : {fer_available()} ({fer_path() or 'not found'})")
