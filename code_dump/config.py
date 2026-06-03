"""
config.py — Central configuration for the Lip Vowel Detection System
"""

# ─── MediaPipe Face Mesh Lip Landmark Indices ────────────────────────────────
# Outer lip contour (clockwise from left corner)
OUTER_LIP_IDX = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]

# Inner lip contour (clockwise from left inner corner)
INNER_LIP_IDX = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]

# Critical measurement landmarks
LM = {
    "left_corner":       61,   # Outer left mouth corner
    "right_corner":      291,  # Outer right mouth corner
    "upper_outer_mid":   0,    # Outermost top of upper lip
    "lower_outer_mid":   17,   # Outermost bottom of lower lip
    "upper_inner_mid":   13,   # Inner top of upper lip
    "lower_inner_mid":   14,   # Inner bottom of lower lip
    "upper_left":        40,   # Upper lip left quarter
    "upper_right":       270,  # Upper lip right quarter
    "lower_left":        91,   # Lower lip left quarter
    "lower_right":       321,  # Lower lip right quarter
    # Philtrum / lip bow
    "philtrum_left":     39,
    "philtrum_right":    269,
    # Outer corners for cheek distance
    "left_cheek":        234,
    "right_cheek":       454,
    # Nose tip (reference)
    "nose_tip":          1,
    # Chin (reference)
    "chin":              199,
}

# ─── Kalman Filter Config ────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE  = 1e-3
KALMAN_MEASURE_NOISE  = 1e-1

# ─── Temporal Smoothing ──────────────────────────────────────────────────────
FEATURE_SMOOTH_ALPHA  = 0.35   # EMA alpha for feature smoothing
VOWEL_HISTORY_LEN     = 12     # frames for majority-vote stabilisation
VOWEL_CONFIDENCE_MIN  = 0.42   # minimum confidence to display a vowel

# ─── Vowel Fuzzy-Logic Profiles ──────────────────────────────────────────────
# Each vowel has target ranges for 6 features:
#   mar        : Mouth Aspect Ratio   (inner_height / outer_width)
#   width_norm : Outer width / face_width
#   roundness  : circularity of inner opening  [0=flat, 1=perfect circle]
#   corner_lift: y-displacement of corners (+ = smile, - = frown)
#   lip_ratio  : upper_thickness / lower_thickness
#   v_asym     : (upper_gap - lower_gap) / total_gap  [-1…+1]
#
# Format: (center, sigma)  — Gaussian membership function

VOWEL_PROFILES = {
    "A": {
        "mar":         (0.55, 0.12),
        "width_norm":  (0.48, 0.08),
        "roundness":   (0.38, 0.12),
        "corner_lift": (0.00, 0.04),
        "lip_ratio":   (0.80, 0.15),
        "v_asym":      (-0.05, 0.12),
    },
    "E": {
        "mar":         (0.22, 0.07),
        "width_norm":  (0.54, 0.07),
        "roundness":   (0.18, 0.10),
        "corner_lift": (0.035, 0.025),
        "lip_ratio":   (0.65, 0.15),
        "v_asym":      (0.10, 0.12),
    },
    "I": {
        "mar":         (0.12, 0.05),
        "width_norm":  (0.57, 0.06),
        "roundness":   (0.10, 0.08),
        "corner_lift": (0.045, 0.025),
        "lip_ratio":   (0.60, 0.15),
        "v_asym":      (0.15, 0.12),
    },
    "O": {
        "mar":         (0.45, 0.10),
        "width_norm":  (0.38, 0.07),
        "roundness":   (0.72, 0.12),
        "corner_lift": (-0.01, 0.03),
        "lip_ratio":   (0.95, 0.15),
        "v_asym":      (0.00, 0.10),
    },
    "U": {
        "mar":         (0.32, 0.08),
        "width_norm":  (0.33, 0.06),
        "roundness":   (0.65, 0.12),
        "corner_lift": (-0.015, 0.025),
        "lip_ratio":   (1.05, 0.15),
        "v_asym":      (-0.05, 0.10),
    },
}

# Feature weights (importance in classification)
FEATURE_WEIGHTS = {
    "mar":         1.8,
    "width_norm":  1.5,
    "roundness":   1.6,
    "corner_lift": 1.2,
    "lip_ratio":   0.8,
    "v_asym":      0.7,
}

# ─── Visualisation ───────────────────────────────────────────────────────────
COLORS = {
    "A": (  0, 220, 255),   # Cyan-yellow
    "E": ( 80, 255, 120),   # Lime green
    "I": (255, 200,  50),   # Blue-white
    "O": ( 50, 130, 255),   # Orange
    "U": (230,  60, 230),   # Magenta
    "NONE": (160, 160, 160),
    "bg":        (12,  12,  18),
    "grid":      (30,  30,  42),
    "text_main": (240, 240, 255),
    "text_dim":  (100, 100, 130),
    "outer_lip": (255, 100,  80),
    "inner_lip": (255, 180, 140),
    "landmark":  (255, 255, 100),
    "hud_border":(60,  60,  80),
}

HUD_WIDTH    = 320
RADAR_RADIUS = 90
FONT_SCALE   = 0.55
