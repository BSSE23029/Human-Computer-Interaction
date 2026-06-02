"""
vision/faces.py -- face / eye / smile detection with Haar cascades (ship inside
opencv-python, so 100% offline). Covers: presence, blink counting, smile-mood,
head position.

    faces = detect_faces(frame)            # [(x,y,w,h), ...]
    mood  = mood_of(frame)                 # 'smiling' / 'neutral' / 'no_face'
    zone  = head_zone(faces[0], frame.shape)
    blinks = run_blink()                   # live window, returns final count

Blink note: with no landmarks we use eye-cascade PRESENCE (eyes visible = open,
not visible = closed) feeding the same open->closed->open counter. Approximate
but works; tune vision.blink.eyes_closed_frames.
"""
import os

from core.conf import get
from vision.camera import draw_text

_CASCADES = {}


def _cascade(key: str):
    """Load + cache a Haar cascade by its config key (vision.haar.<key>)."""
    import cv2
    fname = get(f"vision.haar.{key}")
    if fname not in _CASCADES:
        path = os.path.join(cv2.data.haarcascades, fname)
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            print(f"[faces] WARNING: could not load cascade {fname}")
        _CASCADES[fname] = clf
    return _CASCADES[fname]


def _gray(frame):
    import cv2
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def detect_faces(frame):
    """Return a list of face boxes (x, y, w, h)."""
    gray = _gray(frame)
    faces = _cascade("face").detectMultiScale(
        gray, get("vision.haar.face_scale", 1.1), get("vision.haar.face_neighbors", 5))
    return [tuple(f) for f in faces]


def detect_eyes(face_roi_gray):
    """Return eye boxes within a face ROI (grayscale)."""
    eyes = _cascade("eye").detectMultiScale(
        face_roi_gray, 1.1, get("vision.haar.eye_neighbors", 8))
    return [tuple(e) for e in eyes]


def is_smiling(face_roi_gray) -> bool:
    """True if the smile cascade fires in the lower half of the face ROI."""
    h = face_roi_gray.shape[0]
    lower = face_roi_gray[h // 2:, :]
    smiles = _cascade("smile").detectMultiScale(
        lower, 1.7, get("vision.haar.smile_neighbors", 20))
    return len(smiles) > 0


def mood_of(frame) -> str:
    """Coarse mood from the first detected face: 'smiling' / 'neutral' / 'no_face'."""
    gray = _gray(frame)
    faces = detect_faces(gray)
    if not faces:
        return "no_face"
    x, y, w, h = faces[0]
    roi = gray[y:y + h, x:x + w]
    return "smiling" if is_smiling(roi) else "neutral"


def head_zone(bbox, frame_shape) -> str:
    """Where is the face relative to frame center -> Left/Right/Up/Down/Center."""
    x, y, w, h = bbox
    fh, fw = frame_shape[:2]
    cx, cy = x + w / 2, y + h / 2
    dead = get("vision.head_zones.deadzone_ratio", 0.15)
    dx = (cx - fw / 2) / fw
    dy = (cy - fh / 2) / fh
    horiz = "Left" if dx < -dead else ("Right" if dx > dead else "")
    vert = "Up" if dy < -dead else ("Down" if dy > dead else "")
    return (vert + horiz) or "Center"


def make_blink_processor():
    """Return (process_callback, state). state['blinks'] holds the live count."""
    state = {"blinks": 0, "closed": False, "closed_frames": 0}
    need = get("vision.blink.eyes_closed_frames", 2)

    def process(frame):
        import cv2
        gray = _gray(frame)
        faces = detect_faces(gray)
        eyes_present = False
        if faces:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = gray[y:y + h // 2, x:x + w]           # eyes live in upper half
            eyes = detect_eyes(roi)
            eyes_present = len(eyes) > 0
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh),
                              (0, 255, 0), 1)

        if not eyes_present:
            state["closed_frames"] += 1
            if state["closed_frames"] >= need:
                state["closed"] = True
        else:
            if state["closed"]:
                state["blinks"] += 1
            state["closed"] = False
            state["closed_frames"] = 0

        draw_text(frame, f"Blinks: {state['blinks']}", (10, 34))
        status = "EYES CLOSED" if state["closed"] else "eyes open"
        draw_text(frame, status, (10, 70), scale=0.6,
                  color=(0, 0, 255) if state["closed"] else (0, 255, 0))

    return process, state


def run_blink(source=None) -> int:
    """Live blink-detection window. Returns the final blink count on exit."""
    from vision.camera import run_loop
    process, state = make_blink_processor()
    run_loop(process, source=source, window="Blink Detection")
    print(f"Total blinks: {state['blinks']}")
    return state["blinks"]


def run_mood(source=None):
    """Live smile/neutral mood window."""
    from vision.camera import run_loop

    def process(frame):
        import cv2
        gray = _gray(frame)
        for (x, y, w, h) in detect_faces(gray):
            roi = gray[y:y + h, x:x + w]
            smiling = is_smiling(roi)
            color = (0, 255, 0) if smiling else (0, 200, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            draw_text(frame, "Smiling 😊" if smiling else "Neutral 😐",
                      (x, y - 8), color=color, scale=0.6)

    run_loop(process, source=source, window="Mood Detection")
