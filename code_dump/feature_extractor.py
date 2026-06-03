"""
feature_extractor.py — Geometric + shape feature computation from lip landmarks.

Features extracted
------------------
mar         Mouth Aspect Ratio   = inner_height / outer_width
width_norm  Outer mouth width    normalised by inter-cheek distance
roundness   Hu-moment circularity of the inner lip opening
corner_lift Vertical displacement of lip corners (normalised by face height)
lip_ratio   Upper-lip thickness / lower-lip thickness
v_asym      Vertical asymmetry:  (upper_gap − lower_gap) / total_inner_gap
"""

import numpy as np
import cv2
from config import LM, FEATURE_SMOOTH_ALPHA


class FeatureExtractor:

    def __init__(self):
        self._prev: dict | None = None          # previous smoothed features

    # ─── Public API ──────────────────────────────────────────────────────────
    def extract(self, detection: dict) -> dict:
        """
        Parameters
        ----------
        detection : output dict from LipDetector.detect()

        Returns
        -------
        dict of float features, EMA-smoothed across frames.
        """
        sm  = detection["smooth"]
        w, h = detection["frame_wh"]

        raw_feat = self._compute_raw(sm, w, h)
        smooth   = self._ema_smooth(raw_feat)
        return smooth

    # ─── Internal helpers ────────────────────────────────────────────────────
    def _pt(self, smooth: dict, key: str) -> np.ndarray:
        return np.array(smooth[LM[key]], dtype=float)

    def _compute_raw(self, sm: dict, fw: int, fh: int) -> dict:
        # Convenience point getter
        lc  = np.array(sm[LM["left_corner"]],    dtype=float)
        rc  = np.array(sm[LM["right_corner"]],   dtype=float)
        uom = np.array(sm[LM["upper_outer_mid"]], dtype=float)
        lom = np.array(sm[LM["lower_outer_mid"]], dtype=float)
        uim = np.array(sm[LM["upper_inner_mid"]], dtype=float)
        lim = np.array(sm[LM["lower_inner_mid"]], dtype=float)
        ul  = np.array(sm[LM["upper_left"]],      dtype=float)
        ur  = np.array(sm[LM["upper_right"]],     dtype=float)
        ll  = np.array(sm[LM["lower_left"]],      dtype=float)
        lr  = np.array(sm[LM["lower_right"]],     dtype=float)
        lck = np.array(sm[LM["left_cheek"]],      dtype=float)
        rck = np.array(sm[LM["right_cheek"]],     dtype=float)
        nose = np.array(sm[LM["nose_tip"]],       dtype=float)
        chin = np.array(sm[LM["chin"]],           dtype=float)

        # ── 1. Mouth Aspect Ratio (MAR) ──────────────────────────────────────
        outer_width  = np.linalg.norm(rc - lc) + 1e-6
        inner_height = max(0.0, (lim[1] - uim[1]))
        mar          = inner_height / outer_width

        # ── 2. Width Normalised ───────────────────────────────────────────────
        cheek_dist  = np.linalg.norm(rck - lck) + 1e-6
        width_norm  = outer_width / cheek_dist

        # ── 3. Roundness (circularity of inner lip polygon) ───────────────────
        # Build inner lip contour from tracked inner points
        inner_pts = np.array([
            sm[LM["left_corner"]],
            sm[LM["upper_left"]],
            sm[LM["upper_inner_mid"]],
            sm[LM["upper_right"]],
            sm[LM["right_corner"]],
            sm[LM["lower_right"]],
            sm[LM["lower_inner_mid"]],
            sm[LM["lower_left"]],
        ], dtype=np.float32)

        area      = cv2.contourArea(inner_pts.reshape(-1, 1, 2))
        perimeter = cv2.arcLength(inner_pts.reshape(-1, 1, 2), closed=True) + 1e-6
        roundness = min(1.0, (4 * np.pi * area) / (perimeter ** 2))

        # ── 4. Corner Lift ────────────────────────────────────────────────────
        # Midpoint of the corners vs reference line between ul/ur midpoint
        face_height = np.linalg.norm(chin - nose) + 1e-6
        ref_y       = (ul[1] + ur[1]) / 2.0
        corner_y    = (lc[1] + rc[1]) / 2.0
        corner_lift = (ref_y - corner_y) / face_height   # + = corners above reference (smile)

        # ── 5. Lip Ratio (upper thickness / lower thickness) ─────────────────
        upper_mid_y = (uom[1] + uim[1]) / 2.0
        lower_mid_y = (lim[1] + lom[1]) / 2.0
        ref_gap_y   = (uim[1] + lim[1]) / 2.0            # inner gap midpoint
        upper_thick = max(0.0, ref_gap_y - uom[1])
        lower_thick = max(0.0, lom[1] - ref_gap_y)
        lip_ratio   = upper_thick / (lower_thick + 1e-6)

        # ── 6. Vertical Asymmetry ─────────────────────────────────────────────
        total_gap = inner_height + 1e-6
        upper_gap = max(0.0, uim[1] - uom[1])            # upper lip opens upward
        lower_gap = max(0.0, lom[1] - lim[1])            # lower lip opens downward
        v_asym    = (upper_gap - lower_gap) / total_gap

        return {
            "mar":         float(np.clip(mar,         0, 1.5)),
            "width_norm":  float(np.clip(width_norm,  0, 1.0)),
            "roundness":   float(np.clip(roundness,   0, 1.0)),
            "corner_lift": float(np.clip(corner_lift,-0.15, 0.15)),
            "lip_ratio":   float(np.clip(lip_ratio,   0, 3.0)),
            "v_asym":      float(np.clip(v_asym,     -1, 1)),
        }

    def _ema_smooth(self, raw: dict) -> dict:
        if self._prev is None:
            self._prev = dict(raw)
            return dict(raw)
        α = FEATURE_SMOOTH_ALPHA
        smoothed = {k: α * raw[k] + (1 - α) * self._prev[k] for k in raw}
        self._prev = smoothed
        return smoothed
