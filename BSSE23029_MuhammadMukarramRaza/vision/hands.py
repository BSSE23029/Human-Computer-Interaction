"""
vision/hands.py  --  hand gesture recognition.

Reference guide section 4 values (all from config):

    Fingers UP (Index/Mid/Ring/Pinky): Tip.y < PIP.y
        TIP  indices: 8, 12, 16, 20
        PIP  indices: 6, 10, 14, 18

    Thumb UP (Right hand, mirrored webcam): Tip.x < MCP.x  (landmark 4 < landmark 2)
    Thumb UP (Left  hand, mirrored webcam): Tip.x > MCP.x

    Thumbs Up gesture:
        fingers == 1  AND  thumb_tip.y < thumb_ip.y  AND  index_tip.y > index_pip.y

    High Five gesture:
        fingers == 5  AND  wrist.y < screen_height / 1.6

    Hold timer:
        gesture must be held >= 1.0 seconds to register (prevents flickering)

Two backends:
    mediapipe  → 21 landmarks per hand, handedness-aware thumb, multi-hand
    opencv     → skin-mask + convex hull (fallback)
"""

import math
import time

import cv2
import numpy as np

from core.conf import get
from vision.backend import BACKEND
from vision.camera import draw_text, distance
from vision.smoothing import MajorityBuffer, HoldTimer


# ── MediaPipe Hands singleton ─────────────────────────────────────────────────
_mp_hands = None

def _get_mp_hands():
    global _mp_hands
    if _mp_hands is None:
        import mediapipe as mp
        _mp_hands = mp.solutions.hands.Hands(
            max_num_hands            = int(get("vision.mediapipe.max_hands", 2)),
            min_detection_confidence = float(get("vision.mediapipe.min_detection_confidence", 0.7)),
            min_tracking_confidence  = float(get("vision.mediapipe.min_tracking_confidence",  0.7)),
        )
    return _mp_hands


# ── landmark index constants (MediaPipe) ─────────────────────────────────────
_FINGER_TIPS = [8, 12, 16, 20]     # Index, Middle, Ring, Pinky tip
_FINGER_PIPS = [6, 10, 14, 18]     # corresponding PIP joints
_THUMB_TIP   = 4
_THUMB_IP    = 3                   # for Thumbs Up check
_THUMB_MCP   = 2                   # for thumb UP/DOWN check (reference guide)
_INDEX_TIP   = 8
_INDEX_PIP   = 6
_WRIST       = 0


# ─────────────────────────────────────────────────────────────────────────────
# FINGER COUNTING  (per single hand landmarks)
# ─────────────────────────────────────────────────────────────────────────────

def _count_from_landmarks(lm, handedness: str, h: int) -> int:
    """
    Count extended fingers from 21 MediaPipe landmarks.
    Reference guide section 4:
        Index/Mid/Ring/Pinky: Tip.y < PIP.y
        Thumb Right (mirrored): Tip.x < MCP.x
        Thumb Left  (mirrored): Tip.x > MCP.x
    """
    fingers = 0

    # four fingers
    for tip, pip in zip(_FINGER_TIPS, _FINGER_PIPS):
        if lm[tip].y < lm[pip].y:
            fingers += 1

    # thumb — use MCP (landmark 2) per reference guide
    if handedness == "Right":
        if lm[_THUMB_TIP].x < lm[_THUMB_MCP].x:
            fingers += 1
    else:
        if lm[_THUMB_TIP].x > lm[_THUMB_MCP].x:
            fingers += 1

    return min(5, fingers)


def _classify_special_gestures(lm, fingers: int, h: int) -> str | None:
    """
    Check for special gestures that need more than just a finger count.

    Thumbs Up:
        fingers == 1 AND thumb_tip.y < thumb_ip.y AND index_tip.y > index_pip.y
    High Five:
        fingers == 5 AND wrist.y < screen_height / 1.6
    """
    # Thumbs Up
    if fingers == 1:
        if (lm[_THUMB_TIP].y < lm[_THUMB_IP].y and
                lm[_INDEX_TIP].y > lm[_INDEX_PIP].y):
            return "Thumbs Up"

    # High Five
    if fingers == 5:
        y_ratio = float(get("vision.gesture.high_five_y_ratio", 1.6))
        if lm[_WRIST].y * h < h / y_ratio:
            return "High Five"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# GESTURE MAP LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def classify_gesture(count: int) -> tuple:
    """Map finger count → (name, emoji) from config gesture_map."""
    gmap  = get("vision.gesture.gesture_map") or {}
    entry = gmap.get(str(count)) or gmap.get(count)
    if entry:
        return entry[0], entry[1]
    return "Unknown", "❓"


_SPECIAL_EMOJI = {
    "Thumbs Up": "👍",
    "High Five":  "🙌",
}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def count_fingers(frame) -> tuple:
    """Return (finger_count 0-5, contour_or_None). Single hand."""
    if BACKEND == "mediapipe":
        hands = count_all_hands(frame)
        if not hands:
            return 0, None
        return hands[0]["fingers"], None
    return _count_opencv(frame)


def count_all_hands(frame) -> list:
    """
    Return list of hand dicts (one per detected hand):
        {hand, fingers, gesture, emoji, special}
    Draws landmarks on `frame` in-place.
    """
    if BACKEND == "mediapipe":
        return _count_all_mp(frame)
    # opencv path: single hand only
    count, contour = _count_opencv(frame)
    name, emoji    = classify_gesture(count)
    return [{"hand": "Unknown", "fingers": count, "gesture": name,
             "emoji": emoji, "special": None}]


def _count_all_mp(frame) -> list:
    """MediaPipe multi-hand with mesh overlay + special gesture detection."""
    try:
        import mediapipe as mp
        from vision.draw import draw_hand_mesh
        hands_sol = _get_mp_hands()
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = hands_sol.process(rgb)
        h, w = frame.shape[:2]

        if not res.multi_hand_landmarks:
            return []

        results = []
        for hand_lm, hand_cls in zip(res.multi_hand_landmarks, res.multi_handedness):
            handedness = hand_cls.classification[0].label
            lm         = hand_lm.landmark

            fingers = _count_from_landmarks(lm, handedness, h)
            special = _classify_special_gestures(lm, fingers, h)

            if special:
                name, emoji = special, _SPECIAL_EMOJI.get(special, "✋")
            else:
                name, emoji = classify_gesture(fingers)

            # elegant hand mesh (alpha-blended)
            draw_hand_mesh(frame, hand_lm)

            results.append({
                "hand":    handedness,
                "fingers": fingers,
                "gesture": name,
                "emoji":   emoji,
                "special": special,
            })

        return results

    except Exception:
        count, contour = _count_opencv(frame)
        name, emoji    = classify_gesture(count)
        return [{"hand": "Unknown", "fingers": count, "gesture": name,
                 "emoji": emoji, "special": None}]


# ─────────────────────────────────────────────────────────────────────────────
# GESTURE PROCESSOR  (stateful, with HoldTimer + MajorityBuffer)
# ─────────────────────────────────────────────────────────────────────────────

def make_gesture_processor():
    """
    Returns (process_fn, state_dict).
    process_fn(frame) → updates state["committed_gesture"] only after hold_seconds.
    """
    n      = int(get("vision.smoothing.gesture_history", 12))
    _buf   = MajorityBuffer(maxlen=n)
    _hold  = HoldTimer(seconds=float(get("vision.gesture.hold_seconds", 1.0)))
    state  = {"committed_gesture": None, "committed_emoji": "", "all_hands": []}

    def process(frame) -> list:
        hands = count_all_hands(frame)
        state["all_hands"] = hands

        # majority vote on first hand's gesture label
        label = hands[0]["gesture"] if hands else "none"
        stable = _buf.update(label)
        committed = _hold.update(stable)

        if committed and committed != "none":
            state["committed_gesture"] = committed
            # find emoji for committed label
            if hands and hands[0]["gesture"] == committed:
                state["committed_emoji"] = hands[0]["emoji"]

        return hands

    return process, state


# ─────────────────────────────────────────────────────────────────────────────
# OPENCV FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _skin_mask(bgr):
    ycrcb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lo     = np.array(get("vision.gesture.skin_ycrcb.lower", [0, 133, 77]),  dtype="uint8")
    hi     = np.array(get("vision.gesture.skin_ycrcb.upper", [255,173,127]), dtype="uint8")
    mask   = cv2.inRange(ycrcb, lo, hi)
    kernel = np.ones((5,5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(mask, (5,5), 0)


def _largest_contour(mask):
    min_area = get("vision.gesture.min_contour_area", 5000)
    cnts, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= min_area else None


def _count_opencv(frame) -> tuple:
    c = _largest_contour(_skin_mask(frame))
    if c is None: return 0, None

    hull_idx = cv2.convexHull(c, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3: return 0, c

    defects = cv2.convexityDefects(c, hull_idx)
    if defects is None:
        area      = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity  = area / hull_area if hull_area else 1.0
        return (0 if solidity > 0.9 else 1), c

    gaps = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i,0]
        start, end, far = tuple(c[s][0]), tuple(c[e][0]), tuple(c[f][0])
        a  = distance(start, end)
        b  = distance(start, far)
        cc = distance(end,   far)
        if b * cc == 0: continue
        angle = math.acos(max(-1.0, min(1.0, (b*b + cc*cc - a*a) / (2*b*cc))))
        if angle <= math.pi / 2 and d > 10000:
            gaps += 1

    return min(5, gaps + 1 if gaps > 0 else 1), c


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO WINDOWS
# ─────────────────────────────────────────────────────────────────────────────

def run_gesture(source=None):
    """ESC to exit. Shows hand mesh + committed gesture after hold timer."""
    from vision.camera import run_loop
    from vision.draw import draw_status_bar, draw_corner_badge, _CYAN, _WHITE, _GREEN
    process, state = make_gesture_processor()

    def draw(frame):
        hands = process(frame)   # mesh drawn inside count_all_hands

        if not hands:
            draw_status_bar(frame, [("No hand detected", _WHITE)])
            return

        slots = []
        for info in hands:
            slots.append((f"{info['hand']}: {info['gesture']} {info['emoji']} "
                          f"({info['fingers']}f)", _CYAN))

        if state["committed_gesture"]:
            slots.append((f"▶ {state['committed_gesture']} {state['committed_emoji']}",
                          _GREEN))

        draw_status_bar(frame, slots)
        draw_corner_badge(frame, BACKEND, "tr")

    run_loop(draw, source=source, window="✋ Gesture Recognition")


def run_finger_count(source=None):
    """ESC to exit. Minimal window showing only live finger count + mesh."""
    from vision.camera import run_loop
    from vision.draw import draw_status_bar, _CYAN, _WHITE

    def process(frame):
        h, w = frame.shape[:2]
        hands = count_all_hands(frame)   # mesh drawn inside
        if not hands:
            draw_status_bar(frame, [("Show your hand", _WHITE)])
            return
        slots = [(f"{h['hand']} {h['fingers']} fingers", _CYAN) for h in hands]
        draw_status_bar(frame, slots)

    run_loop(process, source=source, window="🖐 Finger Count")
