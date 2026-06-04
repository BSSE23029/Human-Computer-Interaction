"""
vision/draw.py  --  elegant overlay drawing for all vision modalities.

Design language:
  • Face / hand mesh: fine semi-transparent lines, not opaque rectangles
  • HUD: one bottom status bar (semi-transparent panel) instead of scattered text
  • Color coding: green = positive, amber = caution, red = alert
  • Every element respects config vision.hud.*

Public API (all functions take a BGR numpy frame and draw in-place):
    draw_face_mesh(frame, face_landmarks, h, w)
    draw_hand_mesh(frame, hand_landmarks)
    draw_face_box(frame, bbox, color, label)
    draw_status_bar(frame, slots)        ← the bottom HUD panel
    draw_chip(frame, text, pos, color)   ← small label badge
    draw_ear_gauge(frame, ear, pos)      ← eye openness bar
    hud_color(tier)                      ← tier string → BGR color
"""

import cv2
import numpy as np
from core.conf import get


# ── palette (BGR) ─────────────────────────────────────────────────────────────
_GREEN    = (80,  220,  80)
_AMBER    = (30,  160, 240)
_RED      = (60,   60, 220)
_CYAN     = (220, 220,  40)
_WHITE    = (240, 240, 240)
_DARK     = ( 10,  10,  10)
_MESH_COL = ( 60, 180,  60)    # face mesh lines
_HAND_COL = (200, 130,  30)    # hand mesh lines

_TIER_COLORS = {
    "THRIVING":   _GREEN,
    "CONTENT":    _GREEN,
    "NEUTRAL":    _CYAN,
    "STRESSED":   _AMBER,
    "DISTRESSED": _AMBER,
    "CRISIS":     _RED,
    "UNKNOWN":    _WHITE,
}

_MOOD_COLORS = {
    "happy":     _GREEN,
    "neutral":   _WHITE,
    "surprised": _CYAN,
    "sleepy":    _AMBER,
    "sad":       _AMBER,
    "angry":     _RED,
    "no_face":   _DARK,
}


def hud_color(tier: str) -> tuple:
    return _TIER_COLORS.get(tier.upper() if tier else "", _WHITE)


def mood_color(mood: str) -> tuple:
    return _MOOD_COLORS.get(mood.lower() if mood else "", _WHITE)


# ── face mesh (MediaPipe FaceMesh tessellation) ───────────────────────────────

# Pre-computed FACEMESH_TESSELATION connections (468-point mesh)
# We store a minimal contour set for efficiency
_FACE_CONTOUR = None

def _get_face_contour():
    """Lazy-load the face contour connection list from MediaPipe."""
    global _FACE_CONTOUR
    if _FACE_CONTOUR is None:
        try:
            import mediapipe as mp
            _FACE_CONTOUR = mp.solutions.face_mesh.FACEMESH_CONTOURS
        except Exception:
            _FACE_CONTOUR = set()
    return _FACE_CONTOUR


def draw_face_mesh(frame, face_landmarks, h: int, w: int,
                   color=_MESH_COL, alpha: float = 0.35, thickness: int = 1):
    """
    Draw the FaceMesh 468-point contour on `frame`.
    Uses a transparent overlay (alpha blend) so it doesn't overwhelm the image.
    """
    if face_landmarks is None:
        return

    connections = _get_face_contour()
    if not connections:
        return

    # Draw onto an overlay, then blend
    overlay = frame.copy()
    lm = face_landmarks.landmark
    for conn in connections:
        i, j = conn
        pt1 = (int(lm[i].x * w), int(lm[i].y * h))
        pt2 = (int(lm[j].x * w), int(lm[j].y * h))
        cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_hand_mesh(frame, hand_landmarks, color=_HAND_COL, alpha: float = 0.55):
    """
    Draw MediaPipe hand connections on `frame` with alpha blend.
    hand_landmarks is a single hand NormalizedLandmarkList.
    """
    if hand_landmarks is None:
        return
    try:
        import mediapipe as mp
        h, w = frame.shape[:2]
        overlay = frame.copy()
        mp_draw = mp.solutions.drawing_utils
        mp_draw.draw_landmarks(
            overlay, hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=color, thickness=2, circle_radius=3),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=color, thickness=2),
        )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    except Exception:
        pass


# ── face / bbox drawing ───────────────────────────────────────────────────────

def draw_face_box(frame, bbox: tuple, color=_GREEN,
                  label: str = None, thickness: int = 2):
    """Rounded-corner face box + optional label chip above it."""
    if not bbox:
        return
    x, y, w, h = [int(v) for v in bbox]
    # draw four corner brackets instead of full rectangle — cleaner look
    seg = min(w, h) // 4
    pts = [
        [(x, y+seg), (x, y), (x+seg, y)],
        [(x+w-seg, y), (x+w, y), (x+w, y+seg)],
        [(x+w, y+h-seg), (x+w, y+h), (x+w-seg, y+h)],
        [(x+seg, y+h), (x, y+h), (x, y+h-seg)],
    ]
    for corner in pts:
        for i in range(len(corner)-1):
            cv2.line(frame, corner[i], corner[i+1], color, thickness, cv2.LINE_AA)

    if label:
        draw_chip(frame, label, (x, y - 8), color)


def draw_chip(frame, text: str, pos: tuple, color=_WHITE,
              scale: float = 0.55, pad: int = 4):
    """Small pill-shaped label: filled rounded rectangle + white text."""
    if not text:
        return
    x, y = int(pos[0]), int(pos[1])
    font  = cv2.FONT_HERSHEY_SIMPLEX
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x1, y1 = x - pad, y - th - pad
    x2, y2 = x + tw + pad, y + pad
    # filled background
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    # dark text on colored chip
    txt_col = _DARK if sum(color) > 300 else _WHITE
    cv2.putText(frame, text, (x, y), font, scale, txt_col, thick, cv2.LINE_AA)


# ── EAR gauge bar ─────────────────────────────────────────────────────────────

def draw_ear_gauge(frame, ear: float, pos: tuple,
                   width: int = 60, height: int = 8, threshold: float = 0.25):
    """Horizontal fill bar showing eye openness. Red when closed, green when open."""
    x, y = int(pos[0]), int(pos[1])
    fill  = int(min(1.0, ear / 0.45) * width)
    color = _RED if ear < threshold else _GREEN
    cv2.rectangle(frame, (x, y), (x + width, y + height), _DARK, -1)
    cv2.rectangle(frame, (x, y), (x + fill, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), _WHITE, 1)


# ── status bar (bottom panel) ─────────────────────────────────────────────────

def draw_status_bar(frame, slots: list, height: int = 34, alpha: float = 0.72):
    """
    Single semi-transparent panel across the bottom of the frame.

    `slots` is a list of (text, color) tuples displayed left-to-right with dividers.
    Example:
        [("😟 STRESSED", _AMBER), ("WELLBEING", _WHITE), ("Blinks: 5", _CYAN), ...]
    """
    if not slots:
        return

    fh, fw = frame.shape[:2]
    y_top  = fh - height

    # semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_top), (fw, fh), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # thin top border
    cv2.line(frame, (0, y_top), (fw, y_top), (80, 80, 80), 1)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thick = 1
    x     = 10
    cy    = y_top + height // 2 + 6   # vertical centre of bar

    for i, (text, color) in enumerate(slots):
        if not text:
            continue
        cv2.putText(frame, text, (x, cy), font, scale, color, thick, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
        x += tw + 8
        # divider between slots
        if i < len(slots) - 1:
            cv2.line(frame, (x, y_top + 6), (x, fh - 6), (80, 80, 80), 1)
            x += 8


# ── corner label (minimal — for FPS + backend badge) ─────────────────────────

def draw_corner_badge(frame, text: str, corner: str = "tr",
                      color=_CYAN, scale: float = 0.45):
    """Tiny badge in a corner. corner = 'tl' | 'tr' | 'bl' | 'br'."""
    if not text:
        return
    fh, fw = frame.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    pad = 6
    if corner == "tl":   x, y = pad, th + pad
    elif corner == "tr": x, y = fw - tw - pad, th + pad
    elif corner == "bl": x, y = pad, fh - pad
    else:                x, y = fw - tw - pad, fh - pad

    # tiny dark backing
    cv2.rectangle(frame, (x-2, y-th-2), (x+tw+2, y+2), (0,0,0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, 1, cv2.LINE_AA)
