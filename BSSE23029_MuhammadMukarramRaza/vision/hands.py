"""
vision/hands.py -- hand gesture recognition WITHOUT MediaPipe.

Pure OpenCV: segment skin (YCrCb) -> largest contour -> convex hull +
convexity defects -> count gaps between fingers -> finger count -> gesture.

    count, _ = count_fingers(frame)        # 0..5  (approximate, lighting-sensitive)
    name, emoji = classify_gesture(count)  # ('Peace','✌️')
    run_gesture()                          # live window

Accuracy note: this is a classical approximation of the landmark approach.
Works best with a plain background and decent lighting. Tune vision.skin_ycrcb.
"""
import math

from core.conf import get
from vision.camera import draw_text, distance


def skin_mask(bgr):
    """Binary skin mask via a YCrCb colour range + morphological cleanup."""
    import cv2
    import numpy as np
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lo = np.array(get("vision.gesture.skin_ycrcb.lower", [0, 133, 77]), dtype="uint8")
    hi = np.array(get("vision.gesture.skin_ycrcb.upper", [255, 173, 127]), dtype="uint8")
    mask = cv2.inRange(ycrcb, lo, hi)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def largest_contour(mask, min_area: int = None):
    """Biggest contour in the mask (assumed to be the hand), or None."""
    import cv2
    if min_area is None:
        min_area = get("vision.gesture.min_contour_area", 5000)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= min_area else None


def count_fingers(bgr):
    """Return (finger_count 0..5, contour). Uses convexity-defect gaps + 1."""
    import cv2
    c = largest_contour(skin_mask(bgr))
    if c is None:
        return 0, None
    hull_idx = cv2.convexHull(c, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return 0, c
    defects = cv2.convexityDefects(c, hull_idx)
    if defects is None:
        # No deep gaps: either a fist (0) or one finger. Use solidity to guess.
        area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area else 1.0
        return (0 if solidity > 0.9 else 1), c

    gaps = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start, end, far = tuple(c[s][0]), tuple(c[e][0]), tuple(c[f][0])
        a, b, cc = distance(start, end), distance(start, far), distance(end, far)
        if b * cc == 0:
            continue
        angle = math.acos(max(-1.0, min(1.0, (b * b + cc * cc - a * a) / (2 * b * cc))))
        # a deep (d) and sharp (< 90 deg) valley is the gap between two fingers
        if angle <= math.pi / 2 and d > 10000:
            gaps += 1
    fingers = gaps + 1 if gaps > 0 else 1
    return min(5, fingers), c


def classify_gesture(count: int):
    """Map a finger count to (name, emoji) using the config gesture_map."""
    gmap = get("vision.gesture.gesture_map") or {}
    entry = gmap.get(str(count)) or gmap.get(count)
    if entry:
        return entry[0], entry[1]
    return "Unknown", "❓"


def run_gesture(source=None):
    """Live gesture-recognition window."""
    import cv2
    from vision.camera import run_loop

    def process(frame):
        count, contour = count_fingers(frame)
        name, emoji = classify_gesture(count)
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], -1, (255, 0, 255), 2)
        draw_text(frame, f"Fingers: {count}", (10, 34))
        draw_text(frame, f"Gesture: {name} {emoji}", (10, 72), scale=0.8,
                  color=(0, 255, 255))

    run_loop(process, source=source, window="Gesture Recognition")
