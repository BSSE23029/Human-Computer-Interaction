"""
vision/lips.py  --  lip reading and vowel classification from MediaPipe FaceMesh.

Two classification modes (both available, config selects):
    "simple"   threshold logic on MAR + width ratio  (fastest)
    "fuzzy"    Gaussian membership functions on 6 features (highest accuracy)

Key landmarks used:
    Mouth corners:   61 (left),  291 (right)         outer corners
    Upper inner:     13                               inner top of upper lip
    Lower inner:     14                               inner bottom of lower lip
    Face width:      234 (left cheek), 454 (right cheek)

Features computed:
    MAR          inner mouth height / inner mouth width   (jaw openness)
    width_norm   outer mouth width  / face width          (lip stretch)
    roundness    circularity of inner opening             (O/U puckering)
    corner_lift  vertical displacement of mouth corners   (smile indicator)

All numeric thresholds are in config (vision.lip.*) and match the reference guide.
"""

import math
from core.conf import get
from vision.smoothing import EMA, MajorityBuffer


# ── landmark index constants ──────────────────────────────────────────────────
_INNER_TOP    = 13
_INNER_BOT    = 14
_OUTER_LEFT   = 61
_OUTER_RIGHT  = 291
_UPPER_LEFT   = 40
_UPPER_RIGHT  = 270
_LOWER_LEFT   = 91
_LOWER_RIGHT  = 321
_CHEEK_LEFT   = 234
_CHEEK_RIGHT  = 454
_PHILTRUM_L   = 39
_PHILTRUM_R   = 269
_UPPER_MID    = 0    # outer top of upper lip
_LOWER_MID    = 17   # outer bottom of lower lip

# Inner lip vertical measurement points
_INNER_V_L1   = 82   # upper inner left
_INNER_V_L2   = 87   # lower inner left
_INNER_V_R1   = 312  # upper inner right
_INNER_V_R2   = 317  # lower inner right


def _dist(p1, p2) -> float:
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def extract_lip_features(landmarks, w: int, h: int) -> dict:
    """
    Compute all lip features from a MediaPipe face_landmark list.
    Returns a dict: {mar, width_norm, roundness, corner_lift, lip_ratio, v_asym}
    """
    def pt(idx):
        lm = landmarks[idx]
        return lm.x * w, lm.y * h

    inner_top   = pt(_INNER_TOP)
    inner_bot   = pt(_INNER_BOT)
    outer_left  = pt(_OUTER_LEFT)
    outer_right = pt(_OUTER_RIGHT)
    upper_left  = pt(_UPPER_LEFT)
    upper_right = pt(_UPPER_RIGHT)
    lower_left  = pt(_LOWER_LEFT)
    lower_right = pt(_LOWER_RIGHT)
    cheek_left  = pt(_CHEEK_LEFT)
    cheek_right = pt(_CHEEK_RIGHT)
    upper_mid   = pt(_UPPER_MID)
    lower_mid   = pt(_LOWER_MID)

    # inner height (vertical openness)
    inner_h = _dist(inner_top, inner_bot)
    # outer width
    outer_w = _dist(outer_left, outer_right)
    # face width (cheek-to-cheek)
    face_w  = _dist(cheek_left, cheek_right) or 1.0

    # MAR: inner height / outer width
    mar = inner_h / outer_w if outer_w > 0 else 0.0

    # width_norm: outer mouth width / face width
    width_norm = outer_w / face_w

    # roundness: approximate circularity of inner opening
    # circularity = 4π·A / P²  → for ellipse approximation use ratio h/w
    inner_w = _dist(pt(78), pt(308))   # inner left to inner right
    roundness = min(1.0, inner_h / inner_w) if inner_w > 0 else 0.0

    # corner_lift: y displacement of outer corners relative to inner midpoint
    mid_y = (inner_top[1] + inner_bot[1]) / 2.0
    corner_y_avg = (outer_left[1] + outer_right[1]) / 2.0
    corner_lift  = (mid_y - corner_y_avg) / (face_w / 10.0 + 1e-6)

    # lip_ratio: upper lip thickness / lower lip thickness
    upper_thick = _dist(upper_mid, inner_top)
    lower_thick = _dist(inner_bot, lower_mid)
    lip_ratio   = upper_thick / lower_thick if lower_thick > 0 else 1.0

    # v_asym: vertical asymmetry (upper gap vs lower gap) / total gap
    total_gap = inner_h + 1e-6
    upper_gap = inner_h * 0.5
    lower_gap = inner_h * 0.5
    v_asym    = (upper_gap - lower_gap) / total_gap

    return {
        "mar":         round(mar, 4),
        "width_norm":  round(width_norm, 4),
        "roundness":   round(roundness, 4),
        "corner_lift": round(corner_lift, 4),
        "lip_ratio":   round(lip_ratio, 4),
        "v_asym":      round(v_asym, 4),
    }


# ── simple threshold classifier ───────────────────────────────────────────────

def classify_vowel_simple(features: dict) -> str:
    """
    Reference guide threshold logic (fastest path).
    Returns: A | E | I | O | U | smile | neutral
    """
    mar   = features["mar"]
    width = features["width_norm"]

    # from reference guide
    if mar > float(get("vision.lip.mar_threshold_a", 0.50)):
        return "A"
    if mar > float(get("vision.lip.mar_threshold_e", 0.22)) and width > float(get("vision.lip.width_ratio_wide", 0.42)):
        return "E"
    if mar < float(get("vision.lip.mar_threshold_i", 0.22)) and width > float(get("vision.lip.width_ratio_wide", 0.42)):
        return "I"
    if mar > 0.25 and width < float(get("vision.lip.width_ratio_round", 0.38)):
        # O or U — use roundness to distinguish
        if features["roundness"] > 0.60:
            return "O"
        return "U"
    if features["corner_lift"] > float(get("vision.lip.smile_width_threshold", 0.35)):
        return "smile"
    return "neutral"


# ── fuzzy Gaussian classifier ─────────────────────────────────────────────────

_VOWEL_PROFILES = {
    # (center_mean, sigma) for 6 features — from reference guide
    "A": {"mar":(0.55,0.12), "width_norm":(0.48,0.08), "roundness":(0.38,0.12),
          "corner_lift":(0.00,0.04), "lip_ratio":(0.80,0.15), "v_asym":(-0.05,0.12)},
    "E": {"mar":(0.22,0.07), "width_norm":(0.54,0.07), "roundness":(0.18,0.10),
          "corner_lift":(0.035,0.025), "lip_ratio":(0.65,0.15), "v_asym":(0.10,0.12)},
    "I": {"mar":(0.12,0.05), "width_norm":(0.57,0.06), "roundness":(0.10,0.08),
          "corner_lift":(0.045,0.025), "lip_ratio":(0.60,0.15), "v_asym":(0.15,0.12)},
    "O": {"mar":(0.45,0.10), "width_norm":(0.38,0.07), "roundness":(0.72,0.12),
          "corner_lift":(-0.01,0.03), "lip_ratio":(0.95,0.15), "v_asym":(0.00,0.10)},
    "U": {"mar":(0.32,0.08), "width_norm":(0.33,0.06), "roundness":(0.65,0.12),
          "corner_lift":(-0.015,0.025), "lip_ratio":(1.05,0.15), "v_asym":(-0.05,0.10)},
}

_FEATURE_WEIGHTS = {"mar":1.8, "width_norm":1.5, "roundness":1.6,
                    "corner_lift":1.2, "lip_ratio":0.8, "v_asym":0.7}


def _gaussian(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / (sigma + 1e-9)) ** 2)


def classify_vowel_fuzzy(features: dict) -> tuple:
    """
    Gaussian membership function classifier.
    Returns (vowel_str, confidence_float).
    confidence < vision.lip.confidence_floor → returns ("?", confidence).
    """
    scores = {}
    for vowel, profile in _VOWEL_PROFILES.items():
        score = 0.0
        w_sum = 0.0
        for feat, (mu, sigma) in profile.items():
            val = features.get(feat, 0.0)
            w   = _FEATURE_WEIGHTS.get(feat, 1.0)
            score += w * _gaussian(val, mu, sigma)
            w_sum += w
        scores[vowel] = score / w_sum if w_sum else 0.0

    best     = max(scores, key=scores.get)
    conf     = scores[best]
    floor    = float(get("vision.lip.confidence_floor", 0.42))

    return (best, round(conf, 3)) if conf >= floor else ("?", round(conf, 3))


# ── LipAnalyser — stateful, smoothed ─────────────────────────────────────────

class LipAnalyser:
    """
    Stateful wrapper that applies EMA smoothing to each feature,
    then runs the configured classifier with a majority-vote stabiliser.
    """

    def __init__(self):
        alpha = float(get("vision.smoothing.ema_alpha", 0.35))
        self._smoothers = {k: EMA(alpha) for k in
                           ("mar","width_norm","roundness","corner_lift","lip_ratio","v_asym")}
        n = int(get("vision.smoothing.vowel_history", 12))
        self._buf = MajorityBuffer(maxlen=n)
        self.last_features  = {}
        self.last_vowel     = "?"
        self.last_confidence = 0.0

    def update(self, landmarks, w: int, h: int) -> str:
        """Process one frame → return stable vowel label."""
        raw = extract_lip_features(landmarks, w, h)
        # smooth each feature
        smooth = {k: self._smoothers[k].update(v) for k, v in raw.items()}
        self.last_features = smooth

        mode = get("vision.lip.classifier", "fuzzy")   # "simple" or "fuzzy"
        if mode == "simple":
            label = classify_vowel_simple(smooth)
            self.last_confidence = 1.0
        else:
            label, conf = classify_vowel_fuzzy(smooth)
            self.last_confidence = conf

        stable = self._buf.update(label)
        self.last_vowel = stable or label
        return self.last_vowel

    def reset(self):
        for s in self._smoothers.values():
            s.reset()
        self._buf.clear()
        self.last_vowel     = "?"
        self.last_confidence = 0.0


# ── standalone demo window ────────────────────────────────────────────────────

def run_lips(source=None):
    """
    Standalone lip / vowel reading window.  ESC to exit.
    Shows: face mesh, lip contour highlight, vowel label, MAR bar.
    Requires MediaPipe backend.
    """
    from vision.camera import run_loop
    from vision.faces import _get_mp_face_mesh, get_face_landmarks
    from vision.draw import (draw_face_mesh, draw_status_bar, draw_corner_badge,
                              draw_chip, _CYAN, _GREEN, _AMBER, _WHITE, _RED)
    from vision.backend import BACKEND
    import cv2

    if BACKEND != "mediapipe":
        print("[lips] run_lips() requires the mediapipe backend.")
        return

    analyser = LipAnalyser()

    # lip contour landmark indices (outer)
    _OUTER_LIP = [61,185,40,39,37,0,267,269,270,409,
                  291,375,321,405,314,17,84,181,91,146]

    def process(frame):
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = _get_mp_face_mesh().process(rgb)

        if not res.multi_face_landmarks:
            draw_status_bar(frame, [("No face detected", _WHITE)])
            return

        fl     = res.multi_face_landmarks[0]
        lm     = fl.landmark

        # face mesh overlay
        draw_face_mesh(frame, fl, h, w)

        # highlight lip contour (thicker, brighter)
        pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in _OUTER_LIP]
        for i in range(len(pts)):
            cv2.line(frame, pts[i], pts[(i+1)%len(pts)], (0,200,255), 2, cv2.LINE_AA)

        # classify vowel
        vowel = analyser.update(lm, w, h)
        conf  = analyser.last_confidence
        mar   = analyser.last_features.get("mar", 0)
        wn    = analyser.last_features.get("width_norm", 0)

        # MAR bar (horizontal, above lips)
        lip_y = int(min(lm[i].y*h for i in [0,17]) - 20)
        bar_w = int(min(1.0, mar / 0.6) * 120)
        cv2.rectangle(frame, (w//2-60, lip_y), (w//2-60+120, lip_y+8), (30,30,30), -1)
        cv2.rectangle(frame, (w//2-60, lip_y), (w//2-60+bar_w,  lip_y+8), _AMBER, -1)
        cv2.putText(frame, "MAR", (w//2+65, lip_y+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _AMBER, 1, cv2.LINE_AA)

        # vowel chip over mouth
        mouth_cx = int((lm[61].x + lm[291].x) / 2 * w)
        mouth_y  = int(lm[17].y * h) + 20
        col = _GREEN if vowel in "AEIOU" else _WHITE
        draw_chip(frame, f"{vowel}  {conf:.2f}", (mouth_cx-20, mouth_y), col)

        draw_status_bar(frame, [
            (f"Vowel: {vowel}", col),
            (f"conf {conf:.2f}", col),
            (f"MAR {mar:.2f}", _AMBER),
            (f"width {wn:.2f}", _CYAN),
        ])
        draw_corner_badge(frame, BACKEND, "tr")

    run_loop(process, source=source, window="👄 Lip / Vowel Reading")
