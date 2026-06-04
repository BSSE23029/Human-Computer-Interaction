"""
core/provider.py  --  the LIVE singleton.

One object, initialised once at startup, readable and writable from any module.
No parameters passed across call boundaries — every module imports LIVE directly.

Three namespaces:
    LIVE.vision   updated by the camera thread (15–30 fps)
    LIVE.audio    updated by capture / STT calls
    LIVE.session  reference to the SessionState (same object, always current)

Thread safety: every public write goes through a threading.Lock.
Reads are lock-free for performance; stale reads are acceptable (the camera
thread writes faster than any consumer reads).

Usage:
    from core.provider import LIVE
    LIVE.vision.label          # latest detected labels (or None)
    LIVE.vision.frame          # latest BGR numpy frame (or None)
    LIVE.audio.transcript      # latest STT result dict (or None)
    LIVE.session               # the SessionState — full turn history
"""

import threading
import time
from typing import Optional


class _VisionState:
    """Written by the camera worker thread."""
    __slots__ = ("frame", "label", "running", "last_updated")

    def __init__(self):
        self.frame:        Optional[object] = None   # BGR numpy array
        self.label:        Optional[dict]   = None   # bridge output dict
        self.running:      bool             = False
        self.last_updated: float            = 0.0    # time.time()

    def is_stale(self, max_age_seconds: float = 2.0) -> bool:
        """True if no frame has arrived recently (camera frozen / stopped)."""
        return (time.time() - self.last_updated) > max_age_seconds

    def update(self, frame, label: dict) -> None:
        self.frame        = frame
        self.label        = label
        self.running      = True
        self.last_updated = time.time()

    def stop(self) -> None:
        self.running = False

    def snapshot(self) -> dict:
        """Return a safe copy of current label for pipeline use.
        Returns a null label if camera is not running or frame is stale.
        """
        if not self.running or self.is_stale():
            return {
                "present": False, "faces": 0, "mood": "no_face",
                "head_zone": None, "gesture": "none", "fingers": 0,
                "_stale": True,
            }
        return dict(self.label) if self.label else {
            "present": False, "faces": 0, "mood": "no_face",
            "head_zone": None, "gesture": "none", "fingers": 0,
        }


class _AudioState:
    """Written by STT / capture calls."""
    __slots__ = ("buffer", "transcript", "recording", "last_updated")

    def __init__(self):
        self.buffer:       Optional[object] = None   # float32 numpy array
        self.transcript:   Optional[dict]   = None   # {text, language, confidence}
        self.recording:    bool             = False
        self.last_updated: float            = 0.0

    def set_transcript(self, result: dict) -> None:
        self.transcript   = result
        self.last_updated = time.time()

    def clear(self) -> None:
        self.buffer     = None
        self.transcript = None
        self.recording  = False


class _Provider:
    """
    The global live-state provider.

    Attributes are public by design — all modules read directly.
    Writes that must be atomic use the lock context manager.
    """

    def __init__(self):
        self.vision  = _VisionState()
        self.audio   = _AudioState()
        self.session = None       # set to a SessionState at app startup
        self._lock   = threading.Lock()

    # ── context manager for multi-field atomic writes ─────────────────────
    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *_):
        self._lock.release()

    # ── convenience reads used by stages ─────────────────────────────────
    def vision_snapshot(self) -> dict:
        """Thread-safe snapshot of the latest vision label."""
        return self.vision.snapshot()

    def latest_transcript(self) -> Optional[dict]:
        return self.audio.transcript

    def latest_frame(self):
        return self.vision.frame

    # ── lifecycle ─────────────────────────────────────────────────────────
    def init_session(self, session) -> None:
        """Call once at startup with the SessionState."""
        self.session = session

    def reset(self) -> None:
        """Called on session reset — clears accumulated state."""
        self.audio.clear()
        self.vision.stop()
        if self.session:
            self.session.reset()


# ── module-level singleton ────────────────────────────────────────────────────
LIVE = _Provider()
