"""
lip_detector.py — MediaPipe Face Mesh integration with Kalman-filtered landmarks.

Responsibilities
----------------
* Load the Face Mesh model once and expose detect()
* Return raw + smoothed landmark arrays for the lip region
* Apply a per-landmark 2-D Kalman filter for temporal stability
"""

import numpy as np
import mediapipe as mp
from config import (
    OUTER_LIP_IDX, INNER_LIP_IDX, LM,
    KALMAN_PROCESS_NOISE, KALMAN_MEASURE_NOISE,
)


# ─────────────────────────────────────────────────────────────────────────────
class KalmanLandmark:
    """Lightweight 2-D Kalman filter (constant-velocity model) for one point."""

    def __init__(self):
        dt = 1.0
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        q = KALMAN_PROCESS_NOISE
        r = KALMAN_MEASURE_NOISE
        self.Q = np.eye(4) * q
        self.R = np.eye(2) * r
        self.P = np.eye(4) * 1.0
        self.x = np.zeros((4, 1))
        self.initialized = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        z = measurement.reshape(2, 1)
        if not self.initialized:
            self.x[:2] = z
            self.initialized = True
            return measurement

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        y  = z - self.H @ self.x
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[:2].flatten()


# ─────────────────────────────────────────────────────────────────────────────
class LipDetector:
    """
    Wraps MediaPipe Face Mesh, extracts and smooths all lip-related landmarks.
    """

    def __init__(self, max_faces: int = 1, min_detect_conf: float = 0.6,
                 min_track_conf: float = 0.5):

        self._mp_fm = mp.solutions.face_mesh
        self._face_mesh = self._mp_fm.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,           # enables iris + detailed lip pts
            min_detection_confidence=min_detect_conf,
            min_tracking_confidence=min_track_conf,
        )

        # Collect all unique landmark indices we care about
        all_ids = set(OUTER_LIP_IDX + INNER_LIP_IDX + list(LM.values()))
        self._tracked_ids = sorted(all_ids)

        # One Kalman filter per tracked landmark
        self._kalman: dict[int, KalmanLandmark] = {
            idx: KalmanLandmark() for idx in self._tracked_ids
        }

    # ─── Public API ──────────────────────────────────────────────────────────
    def detect(self, bgr_frame: np.ndarray) -> dict | None:
        """
        Run face mesh on *bgr_frame*.

        Returns
        -------
        dict with keys:
            'raw'      : {landmark_id: (x_px, y_px, z_norm)}
            'smooth'   : {landmark_id: (x_px, y_px)}     ← Kalman smoothed
            'outer_lip': Nx2 array  (smoothed pixel coords)
            'inner_lip': Nx2 array
            'frame_wh' : (width, height)
        or None if no face detected.
        """
        h, w = bgr_frame.shape[:2]
        rgb = bgr_frame[:, :, ::-1]          # BGR → RGB (no copy overhead)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        lm_list = face.landmark

        raw    = {}
        smooth = {}

        for idx in self._tracked_ids:
            lm = lm_list[idx]
            px = lm.x * w
            py = lm.y * h
            pz = lm.z          # normalised depth (negative = closer)
            raw[idx] = (px, py, pz)
            sx, sy = self._kalman[idx].update(np.array([px, py]))
            smooth[idx] = (sx, sy)

        outer = np.array([smooth[i] for i in OUTER_LIP_IDX], dtype=np.float32)
        inner = np.array([smooth[i] for i in INNER_LIP_IDX], dtype=np.float32)

        return {
            "raw":       raw,
            "smooth":    smooth,
            "outer_lip": outer,
            "inner_lip": inner,
            "frame_wh":  (w, h),
        }

    def release(self):
        self._face_mesh.close()
