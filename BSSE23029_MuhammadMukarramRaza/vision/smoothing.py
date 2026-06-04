"""
vision/smoothing.py  --  temporal stabilisers for vision outputs.

All parameters come from config (vision.smoothing.*).

    EMA            exponential moving average for scalar sensor values
    KalmanScalar   1-D Kalman filter for continuous landmark values
    MajorityBuffer deque(maxlen=N) + mode() for discrete labels (vowel/gesture)
    HoldTimer      require a label to be held for N seconds before committing
"""

import math
import time
from collections import Counter, deque
from core.conf import get


# ── EMA ───────────────────────────────────────────────────────────────────────
class EMA:
    """Exponential Moving Average.  alpha=0.35 → moderate smoothing (reference guide)."""

    def __init__(self, alpha: float = None):
        self.alpha = alpha if alpha is not None else float(get("vision.smoothing.ema_alpha", 0.35))
        self._v = None

    def update(self, x: float) -> float:
        if self._v is None:
            self._v = float(x)
        else:
            self._v = self.alpha * float(x) + (1.0 - self.alpha) * self._v
        return self._v

    def reset(self):
        self._v = None

    @property
    def value(self):
        return self._v


# ── Kalman (1-D scalar) ───────────────────────────────────────────────────────
class KalmanScalar:
    """
    Simple 1-D Kalman filter for smoothing noisy MediaPipe landmark coordinates.

    Q (process noise) = 1e-3  — allows smooth continuous motion
    R (measurement noise) = 1e-1 — trusts the MP measurement but damps micro-jitter
    These are the exact values from the reference guide / config.py code-dump.
    """

    def __init__(self, q: float = None, r: float = None):
        self.q = q if q is not None else float(get("vision.smoothing.kalman_process_noise", 1e-3))
        self.r = r if r is not None else float(get("vision.smoothing.kalman_measure_noise", 1e-1))
        self._x = None    # state estimate
        self._p = 1.0     # estimate error covariance

    def update(self, measurement: float) -> float:
        z = float(measurement)
        if self._x is None:
            self._x = z
            return z
        # predict
        p_pred = self._p + self.q
        # update
        k      = p_pred / (p_pred + self.r)
        self._x = self._x + k * (z - self._x)
        self._p = (1.0 - k) * p_pred
        return self._x

    def reset(self):
        self._x = None
        self._p = 1.0

    @property
    def value(self):
        return self._x


# ── Majority-vote buffer ──────────────────────────────────────────────────────
class MajorityBuffer:
    """
    deque(maxlen=N) with mode() output.  Used for vowel / gesture stabilisation.
    N = 12 frames (reference guide: Vowel / Gesture Stabilizer).
    """

    def __init__(self, maxlen: int = None):
        n = maxlen if maxlen is not None else int(get("vision.smoothing.gesture_history", 12))
        self._buf = deque(maxlen=n)

    def update(self, label) -> object:
        self._buf.append(label)
        return self.mode()

    def mode(self):
        if not self._buf:
            return None
        return Counter(self._buf).most_common(1)[0][0]

    def clear(self):
        self._buf.clear()

    def full(self) -> bool:
        return len(self._buf) == self._buf.maxlen


# ── Hold timer ────────────────────────────────────────────────────────────────
class HoldTimer:
    """
    Require a label to be held stable for `seconds` before it is committed.
    Reference: Game Input Match — Hold duration >= 1.0 seconds.
    """

    def __init__(self, seconds: float = None):
        self.seconds = seconds if seconds is not None else float(
            get("vision.gesture.hold_seconds", 1.0))
        self._candidate  = None
        self._start_time = 0.0
        self._committed  = None

    def update(self, label) -> object | None:
        """
        Feed the current frame label.
        Returns the committed label once it has been held long enough,
        otherwise returns None.
        """
        now = time.time()
        if label != self._candidate:
            self._candidate  = label
            self._start_time = now

        if label is not None and (now - self._start_time) >= self.seconds:
            if self._committed != label:
                self._committed = label
            return self._committed

        return self._committed   # keep returning last committed while waiting

    def reset(self):
        self._candidate  = None
        self._start_time = 0.0
        self._committed  = None

    @property
    def committed(self):
        return self._committed
