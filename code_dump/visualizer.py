"""
visualizer.py — Professional OpenCV HUD for the lip vowel detection system.

Layout
------
  ┌──────────────────────────────────────────────────────┬────────────────────┐
  │                                                      │  SIDE PANEL (HUD)  │
  │          LIVE CAMERA FEED                            │  ● Vowel Display   │
  │          (with lip overlay)                          │  ● Confidence Bar  │
  │                                                      │  ● Radar Chart     │
  │                                                      │  ● Feature Table   │
  │                                                      │  ● Timeline Strip  │
  └──────────────────────────────────────────────────────┴────────────────────┘
"""

import cv2
import numpy as np
import math
from collections import deque
from config import COLORS, HUD_WIDTH, RADAR_RADIUS, FONT_SCALE, VOWEL_PROFILES


_FONT      = cv2.FONT_HERSHEY_DUPLEX
_FONT_MONO = cv2.FONT_HERSHEY_PLAIN

TIMELINE_LEN = 80   # number of frames stored in timeline strip


class Visualizer:

    def __init__(self):
        self._timeline: deque = deque(maxlen=TIMELINE_LEN)
        self._conf_trails: dict = {v: deque(maxlen=TIMELINE_LEN)
                                   for v in ["A", "E", "I", "O", "U"]}

    # ──────────────────────────────────────────────────────────────────────────
    def render(
        self,
        frame:       np.ndarray,
        detection:   dict | None,
        features:    dict | None,
        vowel:       str,
        confidence:  float,
        all_scores:  dict,
        fps:         float,
        calibrating: bool,
    ) -> np.ndarray:
        """Compose the full display frame."""

        vis = frame.copy()

        # ── Draw lip overlays on camera feed ──────────────────────────────────
        if detection is not None:
            self._draw_lip_mesh(vis, detection)

        # ── Build side panel ──────────────────────────────────────────────────
        h = vis.shape[0]
        panel = np.full((h, HUD_WIDTH, 3), COLORS["bg"], dtype=np.uint8)
        self._draw_grid(panel)

        y = 20
        y = self._draw_title(panel, y)
        y = self._draw_fps(panel, y, fps)
        y = self._draw_vowel_display(panel, y, vowel, confidence)
        y = self._draw_confidence_bars(panel, y, all_scores)
        y = self._draw_radar(panel, y, all_scores)
        if features:
            y = self._draw_feature_table(panel, y, features)
        y = self._draw_timeline(panel, y, vowel)

        if calibrating:
            self._draw_calibration_overlay(vis)

        # ── Combine ───────────────────────────────────────────────────────────
        out = np.hstack([vis, panel])
        return out

    # ──────────────────────────────────────────────────────────────────────────
    #  Lip mesh overlay
    # ──────────────────────────────────────────────────────────────────────────
    def _draw_lip_mesh(self, img: np.ndarray, det: dict):
        outer = det["outer_lip"].astype(np.int32)
        inner = det["inner_lip"].astype(np.int32)

        # Filled outer lip (semi-transparent)
        overlay = img.copy()
        cv2.fillPoly(overlay, [outer], COLORS["outer_lip"])
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

        # Outer contour
        cv2.polylines(img, [outer], isClosed=True,
                      color=COLORS["outer_lip"], thickness=2, lineType=cv2.LINE_AA)

        # Inner contour
        cv2.polylines(img, [inner], isClosed=True,
                      color=COLORS["inner_lip"], thickness=1, lineType=cv2.LINE_AA)

        # Key landmark dots
        for xy in [*outer, *inner]:
            cv2.circle(img, tuple(xy), 2, COLORS["landmark"], -1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────────────
    #  Panel sections
    # ──────────────────────────────────────────────────────────────────────────
    def _draw_grid(self, panel):
        for x in range(0, HUD_WIDTH, 20):
            cv2.line(panel, (x, 0), (x, panel.shape[0]),
                     COLORS["grid"], 1)
        for y in range(0, panel.shape[0], 20):
            cv2.line(panel, (0, y), (HUD_WIDTH, y),
                     COLORS["grid"], 1)

    def _draw_title(self, panel, y):
        cv2.putText(panel, "LIP VOWEL DETECTOR", (10, y + 14),
                    _FONT, 0.52, COLORS["text_main"], 1, cv2.LINE_AA)
        cv2.line(panel, (8, y + 22), (HUD_WIDTH - 8, y + 22),
                 COLORS["hud_border"], 1)
        return y + 32

    def _draw_fps(self, panel, y, fps):
        cv2.putText(panel, f"FPS  {fps:5.1f}", (10, y + 12),
                    _FONT_MONO, 1.0, COLORS["text_dim"], 1, cv2.LINE_AA)
        return y + 20

    def _draw_vowel_display(self, panel, y, vowel, confidence):
        color = COLORS.get(vowel, COLORS["NONE"])

        # Big vowel letter
        size = 3.5
        (tw, th), _ = cv2.getTextSize(vowel if vowel != "NONE" else "·",
                                      _FONT, size, 4)
        cx = HUD_WIDTH // 2 - tw // 2
        cv2.putText(panel, vowel if vowel != "NONE" else "·",
                    (cx, y + int(th * 1.05)),
                    _FONT, size, color, 4, cv2.LINE_AA)

        y += int(th * 1.2) + 8

        # Confidence percentage
        conf_str = f"{confidence * 100:.0f}%"
        (cw, _), _ = cv2.getTextSize(conf_str, _FONT, 0.7, 1)
        cv2.putText(panel, conf_str, (HUD_WIDTH // 2 - cw // 2, y + 14),
                    _FONT, 0.7, color, 1, cv2.LINE_AA)

        # Confidence bar
        bar_x, bar_w, bar_h = 12, HUD_WIDTH - 24, 8
        y += 22
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h),
                      COLORS["hud_border"], -1)
        filled = int(bar_w * confidence)
        if filled > 0:
            cv2.rectangle(panel, (bar_x, y), (bar_x + filled, y + bar_h),
                          color, -1)
        y += bar_h + 10
        cv2.line(panel, (8, y), (HUD_WIDTH - 8, y), COLORS["hud_border"], 1)
        return y + 8

    def _draw_confidence_bars(self, panel, y, all_scores):
        cv2.putText(panel, "SCORES", (10, y + 10),
                    _FONT_MONO, 1.0, COLORS["text_dim"], 1, cv2.LINE_AA)
        y += 14
        bar_total = HUD_WIDTH - 70
        for vowel in ["A", "E", "I", "O", "U"]:
            score = all_scores.get(vowel, 0.0)
            color = COLORS[vowel]
            cv2.putText(panel, vowel, (12, y + 10),
                        _FONT, 0.5, color, 1, cv2.LINE_AA)
            bx = 32
            cv2.rectangle(panel, (bx, y + 2), (bx + bar_total, y + 11),
                          COLORS["hud_border"], -1)
            fill = int(bar_total * score)
            if fill > 0:
                cv2.rectangle(panel, (bx, y + 2), (bx + fill, y + 11),
                              color, -1)
            pct = f"{score * 100:4.1f}%"
            cv2.putText(panel, pct, (bx + bar_total + 4, y + 11),
                        _FONT_MONO, 0.85, COLORS["text_dim"], 1, cv2.LINE_AA)
            y += 16

        cv2.line(panel, (8, y + 2), (HUD_WIDTH - 8, y + 2),
                 COLORS["hud_border"], 1)
        return y + 10

    def _draw_radar(self, panel, y, all_scores):
        """Pentagon radar chart — one axis per vowel."""
        cx   = HUD_WIDTH // 2
        cy   = y + RADAR_RADIUS + 10
        R    = RADAR_RADIUS
        vowels = ["A", "E", "I", "O", "U"]
        n    = len(vowels)
        angles = [math.pi / 2 + 2 * math.pi * i / n for i in range(n)]

        # Grid rings
        for ring in [0.33, 0.66, 1.0]:
            pts = []
            for a in angles:
                px = int(cx + R * ring * math.cos(a))
                py = int(cy - R * ring * math.sin(a))
                pts.append((px, py))
            cv2.polylines(panel, [np.array(pts, np.int32)], True,
                          COLORS["grid"], 1, cv2.LINE_AA)

        # Axis lines and labels
        for i, (vowel, angle) in enumerate(zip(vowels, angles)):
            ex = int(cx + R * math.cos(angle))
            ey = int(cy - R * math.sin(angle))
            cv2.line(panel, (cx, cy), (ex, ey), COLORS["grid"], 1, cv2.LINE_AA)
            lx = int(cx + (R + 14) * math.cos(angle))
            ly = int(cy - (R + 14) * math.sin(angle))
            (tw, th), _ = cv2.getTextSize(vowel, _FONT, 0.45, 1)
            cv2.putText(panel, vowel, (lx - tw // 2, ly + th // 2),
                        _FONT, 0.45, COLORS[vowel], 1, cv2.LINE_AA)

        # Filled polygon
        score_pts = []
        for vowel, angle in zip(vowels, angles):
            s = all_scores.get(vowel, 0.0)
            px = int(cx + R * s * math.cos(angle))
            py = int(cy - R * s * math.sin(angle))
            score_pts.append((px, py))

        overlay = panel.copy()
        cv2.fillPoly(overlay, [np.array(score_pts, np.int32)], (80, 140, 255))
        cv2.addWeighted(overlay, 0.35, panel, 0.65, 0, panel)
        cv2.polylines(panel, [np.array(score_pts, np.int32)], True,
                      (120, 180, 255), 2, cv2.LINE_AA)

        for (px, py) in score_pts:
            cv2.circle(panel, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)

        new_y = cy + R + 14
        cv2.line(panel, (8, new_y), (HUD_WIDTH - 8, new_y),
                 COLORS["hud_border"], 1)
        return new_y + 8

    def _draw_feature_table(self, panel, y, features):
        cv2.putText(panel, "FEATURES", (10, y + 10),
                    _FONT_MONO, 1.0, COLORS["text_dim"], 1, cv2.LINE_AA)
        y += 14
        rows = [
            ("MAR",       features["mar"],         0.0, 0.8),
            ("Width",     features["width_norm"],   0.2, 0.7),
            ("Round",     features["roundness"],    0.0, 1.0),
            ("Lift",      features["corner_lift"], -0.1, 0.1),
            ("LipR",      features["lip_ratio"],    0.3, 2.0),
            ("Asym",      features["v_asym"],       -0.5, 0.5),
        ]
        col_w = (HUD_WIDTH - 20) // 2
        for i, (name, val, lo, hi) in enumerate(rows):
            col = 10 if i % 2 == 0 else 10 + col_w
            row_y = y + (i // 2) * 14
            norm = float(np.clip((val - lo) / (hi - lo + 1e-9), 0, 1))
            bar_w = int((col_w - 50) * norm)
            cv2.putText(panel, f"{name}", (col, row_y + 10),
                        _FONT_MONO, 0.85, COLORS["text_dim"], 1, cv2.LINE_AA)
            cv2.rectangle(panel, (col + 32, row_y + 2),
                          (col + 32 + col_w - 52, row_y + 9),
                          COLORS["hud_border"], -1)
            cv2.rectangle(panel, (col + 32, row_y + 2),
                          (col + 32 + bar_w, row_y + 9),
                          (100, 180, 255), -1)

        rows_count = math.ceil(len(rows) / 2)
        y += rows_count * 14 + 4
        cv2.line(panel, (8, y), (HUD_WIDTH - 8, y), COLORS["hud_border"], 1)
        return y + 8

    def _draw_timeline(self, panel, y, vowel):
        """Scrolling history strip."""
        self._timeline.append(vowel)

        cv2.putText(panel, "HISTORY", (10, y + 10),
                    _FONT_MONO, 1.0, COLORS["text_dim"], 1, cv2.LINE_AA)
        y += 14
        strip_h  = 24
        cell_w   = max(1, (HUD_WIDTH - 16) // TIMELINE_LEN)
        for i, v in enumerate(self._timeline):
            color = COLORS.get(v, COLORS["NONE"])
            x0 = 8 + i * cell_w
            cv2.rectangle(panel, (x0, y), (x0 + cell_w - 1, y + strip_h),
                          color, -1)

        # Current label overlay
        if vowel != "NONE":
            cv2.putText(panel, vowel,
                        (HUD_WIDTH // 2 - 5, y + strip_h - 4),
                        _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return y + strip_h + 6

    def _draw_calibration_overlay(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 100, 200), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        msg = "CALIBRATING — Hold neutral face"
        (tw, _), _ = cv2.getTextSize(msg, _FONT, 0.7, 2)
        cx = frame.shape[1] // 2 - tw // 2
        cy = frame.shape[0] // 2
        cv2.putText(frame, msg, (cx, cy), _FONT, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
