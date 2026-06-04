"""
vision/faces.py  --  face detection, landmarks, emotion, blink, head pose.

Reference guide values (all in config, all tunable):
    FaceMesh min_detection_confidence  0.7
    FaceMesh min_tracking_confidence   0.7
    EAR closed threshold               0.25  (eye is shut)
    EAR open  threshold                0.25  (eye is open / blink end)
    Drowsy closed frames               20
    Drowsy awake recovery frames       5
    EMA alpha                          0.35
    Majority vote window               12 frames

Backend paths (selected by vision/backend.py):
    mediapipe → FaceMesh 478 landmarks → EAR blink, landmark head pose, geometric emotion
    dnn       → YuNet neural net face boxes + FER7 emotion
    opencv    → LBP cascade + Haar eye/smile (fastest, lowest quality)
"""

import math
import os
from functools import lru_cache

import cv2
import numpy as np

from core.conf import get
from vision.backend import BACKEND, fer_path, yunet_path, lbp_face_cascade
from vision.camera import draw_text, distance
from vision.smoothing import EMA, KalmanScalar, MajorityBuffer


# ── grayscale helper ──────────────────────────────────────────────────────────
def _gray(frame):
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ── OpenCV cascade fallbacks ──────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _haar_face():
    p = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    return cv2.CascadeClassifier(p)

@lru_cache(maxsize=1)
def _haar_eye():
    p = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
    return cv2.CascadeClassifier(p)

@lru_cache(maxsize=1)
def _haar_smile():
    p = os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml")
    return cv2.CascadeClassifier(p)


# ── MediaPipe singletons ──────────────────────────────────────────────────────
_mp_face_mesh = None

def _get_mp_face_mesh():
    global _mp_face_mesh
    if _mp_face_mesh is None:
        import mediapipe as mp
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces   = int(get("vision.mediapipe.max_faces", 2)),
            refine_landmarks= True,                # iris + detailed lip contours
            min_detection_confidence = float(get("vision.mediapipe.min_detection_confidence", 0.7)),
            min_tracking_confidence  = float(get("vision.mediapipe.min_tracking_confidence",  0.7)),
        )
    return _mp_face_mesh


# ── YuNet singleton ───────────────────────────────────────────────────────────
_yunet_detector = None

def _get_yunet(w: int, h: int):
    global _yunet_detector
    model = yunet_path()
    if model is None:
        return None
    if _yunet_detector is None:
        _yunet_detector = cv2.FaceDetectorYN.create(
            model, "", (w, h),
            score_threshold = float(get("vision.yunet.confidence", 0.6)),
            nms_threshold   = 0.3,
            top_k           = int(get("vision.mediapipe.max_faces", 2)),
        )
    else:
        _yunet_detector.setInputSize((w, h))
    return _yunet_detector


# ── FER emotion model ─────────────────────────────────────────────────────────
_fer_net = None
_FER_LABELS = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]

def _get_fer():
    global _fer_net
    path = fer_path()
    if path and _fer_net is None:
        try:
            _fer_net = cv2.dnn.readNetFromONNX(path)
        except Exception as e:
            print(f"[vision] FER model load failed: {e}")
    return _fer_net


# ═══════════════════════════════════════════════════════════════════════════════
# FACE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_faces(frame) -> list:
    if BACKEND == "mediapipe": return _faces_mediapipe(frame)
    if BACKEND == "dnn":       return _faces_yunet(frame)
    return _faces_opencv(frame)


def _faces_mediapipe(frame) -> list:
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = _get_mp_face_mesh().process(rgb)
        if not res.multi_face_landmarks:
            return []
        h, w = frame.shape[:2]
        boxes = []
        for fl in res.multi_face_landmarks:
            xs = [lm.x * w for lm in fl.landmark]
            ys = [lm.y * h for lm in fl.landmark]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            boxes.append((x1, y1, x2-x1, y2-y1))
        return boxes
    except Exception:
        return _faces_opencv(frame)


def _faces_yunet(frame) -> list:
    h, w = frame.shape[:2]
    try:
        det = _get_yunet(w, h)
        if det is None:
            return _faces_opencv(frame)
        _, faces = det.detect(frame)
        if faces is None:
            return []
        return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]
    except Exception:
        return _faces_opencv(frame)


def _faces_opencv(frame) -> list:
    gray  = _gray(frame)
    clf   = lbp_face_cascade() or _haar_face()
    if clf is None or clf.empty():
        return []
    faces = clf.detectMultiScale(
        gray,
        float(get("vision.haar.face_scale", 1.1)),
        int(get("vision.haar.face_neighbors", 5)),
    )
    return [tuple(f) for f in faces] if len(faces) > 0 else []


def detect_eyes(face_roi_gray) -> list:
    clf   = _haar_eye()
    if clf is None or clf.empty():
        return []
    eyes  = clf.detectMultiScale(face_roi_gray, 1.1, int(get("vision.haar.eye_neighbors", 8)))
    return [tuple(e) for e in eyes] if len(eyes) > 0 else []


# ═══════════════════════════════════════════════════════════════════════════════
# LANDMARK ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

def get_face_landmarks(frame) -> list | None:
    if BACKEND != "mediapipe":
        return None
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = _get_mp_face_mesh().process(rgb)
        return res.multi_face_landmarks or []
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# EMOTION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def mood_of(frame) -> str:
    if BACKEND == "mediapipe": return _mood_mediapipe(frame)
    return _mood_fer_or_smile(frame)


def _mood_mediapipe(frame) -> str:
    """5-state geometric emotion from FaceMesh + corner_lift for smile."""
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = _get_mp_face_mesh().process(rgb)
        if not res.multi_face_landmarks:
            return "no_face"
        h, w = frame.shape[:2]
        lm   = res.multi_face_landmarks[0].landmark

        def pt(idx): return lm[idx].x * w, lm[idx].y * h

        # EAR for drowsy/surprise detection
        def ear(p): return _ear_from_pts(*(pt(i) for i in p))
        avg_ear = (ear((362,385,387,263,373,380)) + ear((33,160,158,133,153,144))) / 2.0

        # lip width ratio (mouth_width / face_width)
        # Reference: smile when width_ratio > 0.35
        lip_w     = distance(pt(61), pt(291))    # outer mouth corners
        face_w    = distance(pt(234), pt(454))   # cheek-to-cheek
        lip_ratio = lip_w / face_w if face_w > 0 else 0.0

        # corner lift: do mouth corners (78, 308) sit above inner bottom lip (14)?
        # Reference guide: lip corners rise above inner bottom lip → smiling
        corner_avg_y   = (pt(78)[1] + pt(308)[1]) / 2.0
        inner_bottom_y = pt(14)[1]
        corner_lifted  = corner_avg_y < inner_bottom_y   # y increases downward

        # brow distance for anger
        brow_d = distance(pt(70), pt(159))
        norm   = (face_w / 10) if face_w > 0 else 10

        if avg_ear < float(get("vision.blink.ear_closed_threshold", 0.25)):
            return "sleepy"
        if avg_ear > 0.35 and lip_ratio < 0.40:
            return "surprised"
        # happy: width_ratio > 0.38 (reference: > 0.35) AND corners lifted
        if lip_ratio > 0.38 and corner_lifted:
            return "happy"
        if brow_d < norm * 0.8:
            return "angry"
        if lip_ratio < 0.33:
            return "sad"
        return "neutral"
    except Exception:
        return "no_face"


def _mood_fer_or_smile(frame) -> str:
    faces = detect_faces(frame)
    if not faces:
        return "no_face"
    fer = _get_fer()
    if fer is not None:
        return _fer_emotion(frame, faces[0], fer)
    gray = _gray(frame)
    x, y, w, h = faces[0]
    return "happy" if is_smiling(gray[y:y+h, x:x+w]) else "neutral"


def _fer_emotion(frame, bbox, net) -> str:
    try:
        x, y, w, h = bbox
        roi  = frame[max(0,y):y+h, max(0,x):x+w]
        if roi.size == 0: return "neutral"
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        blob = cv2.dnn.blobFromImage(gray.astype(np.float32), 1.0/255, (64,64),
                                     mean=(0,), swapRB=False, crop=False)
        net.setInput(blob)
        out   = net.forward()[0]
        return _FER_LABELS[int(np.argmax(out))]
    except Exception:
        return "neutral"


def is_smiling(face_roi_gray) -> bool:
    h   = face_roi_gray.shape[0]
    roi = face_roi_gray[h//2:, :]
    clf = _haar_smile()
    if clf is None or clf.empty(): return False
    return len(clf.detectMultiScale(roi, 1.7, int(get("vision.haar.smile_neighbors", 20)))) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# HEAD POSE  ──  landmark-based (reference guide section 5)
# ═══════════════════════════════════════════════════════════════════════════════

def head_zone_landmarks(landmarks, w: int, h: int) -> str:
    """
    Landmark-based head pose for webcam use.

    Left / Right: nose x position relative to individual eye centers.

    Up / Down: the ORIGINAL reference guide condition
    (nose.y < eye_mid_y) only fires for extreme tilts (~30°+) because
    the nose tip has to cross the eye level — it never does for casual use.

    PRACTICAL FIX: measure the nose's distance from the eye midpoint,
    normalised by face height.  In a neutral forward pose this ratio is
    ~0.28–0.35.  A moderate tilt is detectable at ±0.08 of that.

        neutral : 0.20 ≤ ratio ≤ 0.42
        Up      : ratio < 0.20  (nose unusually close to eyes)
        Down    : ratio > 0.42  (nose unusually far below eyes)

    Left / Right are still compared to individual eye x positions per
    the reference guide; these work correctly without modification.
    """
    try:
        lm = landmarks

        def pt(idx): return lm[idx].x * w, lm[idx].y * h

        nose       = pt(1)     # nose tip
        left_eye   = pt(159)   # left eye centre
        right_eye  = pt(386)   # right eye centre
        chin       = pt(152)
        forehead   = pt(10)

        eye_mid_x  = (left_eye[0] + right_eye[0]) / 2
        eye_mid_y  = (left_eye[1] + right_eye[1]) / 2
        face_height = abs(chin[1] - forehead[1])

        # ── Left / Right (reference guide, works well) ─────────────────
        if nose[0] < left_eye[0]:
            return "Left"
        if nose[0] > right_eye[0]:
            return "Right"

        # ── Up / Down (practical webcam-scale thresholds) ──────────────
        if face_height > 0:
            # normalised nose-to-eye-midpoint distance
            # positive = nose below eyes (always true for a frontal face)
            ratio = (nose[1] - eye_mid_y) / face_height
        else:
            ratio = 0.30

        if ratio < float(get("vision.head_zones.up_ratio",   0.20)):
            return "Up"
        if ratio > float(get("vision.head_zones.down_ratio", 0.42)):
            return "Down"
        return "Forward"
    except Exception:
        return "Forward"


def head_zone(bbox, frame_shape) -> str:
    """Bounding-box fallback (used when landmarks are not available)."""
    x, y, w, h   = bbox
    fh, fw        = frame_shape[:2]
    cx, cy        = x + w / 2, y + h / 2
    dead          = float(get("vision.head_zones.deadzone_ratio", 0.15))
    dx            = (cx - fw / 2) / fw
    dy            = (cy - fh / 2) / fh
    horiz         = "Left" if dx < -dead else ("Right" if dx > dead else "")
    vert          = "Up"   if dy < -dead else ("Down"  if dy > dead else "")
    return (vert + horiz) or "Center"


# ═══════════════════════════════════════════════════════════════════════════════
# EAR HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _ear_from_pts(p1, p2, p3, p4, p5, p6) -> float:
    """(Vertical_1 + Vertical_2) / (2 * Horizontal)  — reference guide formula."""
    vert_1 = distance(p2, p6)
    vert_2 = distance(p3, p5)
    horiz  = distance(p1, p4)
    return (vert_1 + vert_2) / (2.0 * horiz) if horiz > 0 else 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# BLINK PROCESSOR  (stateful, smoothed)
# ═══════════════════════════════════════════════════════════════════════════════

def make_blink_processor():
    """
    Returns (process_fn, state_dict).

    MediaPipe path:
        - EAR from 6 exact landmarks per eye
        - Closed threshold: 0.25, Open: 0.25  (reference guide)
        - Drowsy: closed_frames > 20
        - Awake reset: open_frames >= 5  ← prevents micro-opens from resetting
        - EMA(alpha=0.35) on raw EAR value

    OpenCV path:
        - Eye cascade presence (still reliable for binary open/closed)
    """
    state = {
        "blinks":       0,
        "closed":       False,
        "closed_frames":0,
        "open_frames":  0,
        "ear":          0.3,
        "drowsy":       False,
    }

    if BACKEND == "mediapipe":
        return _make_blink_mp(state), state
    return _make_blink_cascade(state), state


def _make_blink_mp(state: dict):
    EAR_CLOSED    = float(get("vision.blink.ear_closed_threshold", 0.25))
    DROWSY_FRAMES = int(get("vision.drowsy.closed_frames_alert",   20))
    AWAKE_RECOVER = int(get("vision.drowsy.awake_recovery_frames", 5))
    _ear_ema      = EMA(float(get("vision.smoothing.ema_alpha", 0.35)))
    # MediaPipe landmark indices for EAR (reference guide section 2)
    _L = (362, 385, 387, 263, 373, 380)
    _R = (33,  160, 158, 133, 153, 144)

    def process(frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = _get_mp_face_mesh().process(rgb)
            h, w = frame.shape[:2]
            if not res.multi_face_landmarks:
                return
            lm = res.multi_face_landmarks[0].landmark
            def pt(i): return lm[i].x * w, lm[i].y * h

            raw_ear = (_ear_from_pts(*(pt(i) for i in _L)) +
                       _ear_from_pts(*(pt(i) for i in _R))) / 2.0
            ear     = _ear_ema.update(raw_ear)
            state["ear"] = round(ear, 3)

            eyes_closed = ear < EAR_CLOSED

            if eyes_closed:
                state["closed_frames"] += 1
                state["open_frames"]    = 0
                state["closed"]         = True
                # drowsy: eyes shut more than DROWSY_FRAMES
                state["drowsy"] = state["closed_frames"] >= DROWSY_FRAMES
            else:
                state["open_frames"] += 1
                if state["closed"]:
                    # only count blink if eyes re-opened cleanly
                    state["blinks"] += 1
                state["closed"]        = False
                state["closed_frames"] = 0
                # awake reset only after AWAKE_RECOVER open frames
                if state["open_frames"] >= AWAKE_RECOVER:
                    state["drowsy"] = False

            # ── draw face mesh + elegant HUD ─────────────────────
            from vision.draw import (draw_face_mesh, draw_ear_gauge,
                                     draw_status_bar, draw_corner_badge,
                                     _AMBER, _RED, _GREEN, _WHITE, _CYAN)
            # face mesh overlay
            for fl in res.multi_face_landmarks:
                draw_face_mesh(frame, fl, h, w)

            # EAR gauge
            draw_ear_gauge(frame, ear, (10, 12), threshold=EAR_CLOSED)

            drowsy_col = _RED if state["drowsy"] else _GREEN
            slots = [
                (f"Blinks: {state['blinks']}", _CYAN),
                (f"EAR {ear:.2f}", drowsy_col),
                (get("vision.drowsy.alert_message","DROWSY!") if state["drowsy"]
                 else "Eyes open", drowsy_col),
            ]
            draw_status_bar(frame, slots)
            draw_corner_badge(frame, f"mediapipe  {BACKEND}", "tr")

        except Exception:
            pass

    return process


def _make_blink_cascade(state: dict):
    NEED_CLOSED   = int(get("vision.blink.eyes_closed_frames",    2))
    DROWSY_FRAMES = int(get("vision.drowsy.closed_frames_alert",  20))
    AWAKE_RECOVER = int(get("vision.drowsy.awake_recovery_frames", 5))

    def process(frame):
        gray  = _gray(frame)
        faces = detect_faces(frame)
        eyes_present = False
        if faces:
            x, y, w, h = faces[0]
            roi  = gray[y:y + h//2, x:x + w]
            eyes = detect_eyes(roi)
            eyes_present = len(eyes) > 0
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 1)

        if not eyes_present:
            state["closed_frames"] += 1
            state["open_frames"]    = 0
            if state["closed_frames"] >= NEED_CLOSED:
                state["closed"] = True
            state["drowsy"] = state["closed_frames"] >= DROWSY_FRAMES
        else:
            state["open_frames"] += 1
            if state["closed"]:
                state["blinks"] += 1
            state["closed"]        = False
            state["closed_frames"] = 0
            if state["open_frames"] >= AWAKE_RECOVER:
                state["drowsy"] = False

        draw_text(frame, f"Blinks: {state['blinks']}", (10, 34))

    return process


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE DEMO WINDOWS  —  one per modality, all with mesh overlay
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_all_meshes(frame, landmarks_list, h, w):
    """Draw face mesh for every detected face. Called by all standalone runners."""
    from vision.draw import draw_face_mesh
    if landmarks_list:
        for fl in landmarks_list:
            draw_face_mesh(frame, fl, h, w)


def run_blink(source=None) -> int:
    """ESC to exit. Returns total blink count."""
    from vision.camera import run_loop
    from vision.draw import draw_status_bar, draw_ear_gauge, _CYAN, _RED, _GREEN
    process, state = make_blink_processor()

    def draw(frame):
        process(frame)          # blink logic + mesh already drawn inside
        # additional mesh for non-MP path
        if BACKEND != "mediapipe":
            h, w = frame.shape[:2]
            fls = get_face_landmarks(frame)
            _draw_all_meshes(frame, fls, h, w)

    run_loop(draw, source=source, window="🔴 Blink Detection")
    print(f"Total blinks: {state['blinks']}")
    return state["blinks"]


def run_mood(source=None):
    """ESC to exit. Shows 5-class emotion + face mesh."""
    from vision.camera import run_loop
    from vision.draw import (draw_face_mesh, draw_face_box, draw_status_bar,
                              draw_corner_badge, mood_color, _WHITE)

    def process(frame):
        h, w = frame.shape[:2]
        m    = mood_of(frame)
        col  = mood_color(m)
        fls  = get_face_landmarks(frame)
        _draw_all_meshes(frame, fls, h, w)
        for box in detect_faces(frame):
            draw_face_box(frame, box, col, label=m)
        draw_status_bar(frame, [(f"Mood: {m}", col), (f"[{BACKEND}]", _WHITE)])

    run_loop(process, source=source, window="😊 Mood / Emotion")


def run_head_pose(source=None):
    """ESC to exit. Shows landmark-based head pose + face mesh."""
    from vision.camera import run_loop
    from vision.draw import (draw_face_mesh, draw_face_box, draw_status_bar,
                              draw_corner_badge, _CYAN, _WHITE, _AMBER)

    def process(frame):
        h, w = frame.shape[:2]
        fls  = get_face_landmarks(frame)
        _draw_all_meshes(frame, fls, h, w)

        zone = "no_face"
        if fls:
            lm_list = fls[0].landmark
            zone    = head_zone_landmarks(lm_list, w, h)
            for box in detect_faces(frame):
                col = _AMBER if zone != "Forward" else _CYAN
                draw_face_box(frame, box, col, label=zone)
        else:
            for box in detect_faces(frame):
                zone = head_zone(box, frame.shape)
                draw_face_box(frame, box, _CYAN, label=zone)

        draw_status_bar(frame, [(f"Head: {zone}", _CYAN), (f"[{BACKEND}]", _WHITE)])

    run_loop(process, source=source, window="🎯 Head Pose")


def run_drowsy(source=None):
    """ESC to exit. Focused drowsiness monitor with EAR gauge + mesh."""
    from vision.camera import run_loop
    from vision.draw import (draw_face_mesh, draw_ear_gauge, draw_status_bar,
                              draw_corner_badge, _RED, _GREEN, _CYAN, _WHITE)
    process, state = make_blink_processor()

    def draw(frame):
        h, w = frame.shape[:2]
        process(frame)
        if BACKEND != "mediapipe":
            fls = get_face_landmarks(frame)
            _draw_all_meshes(frame, fls, h, w)
        ear = state.get("ear", 0.3)
        draw_ear_gauge(frame, ear, (10, 12), threshold=float(
            get("vision.blink.ear_closed_threshold", 0.25)))
        col = _RED if state.get("drowsy") else _GREEN
        draw_status_bar(frame, [
            (f"EAR: {ear:.3f}", col),
            (f"Blinks: {state['blinks']}", _CYAN),
            ("⚠ DROWSY" if state.get("drowsy") else "Awake", col),
        ])

    run_loop(draw, source=source, window="😴 Drowsiness Monitor")


def run_all_face(source=None):
    """
    Combined face window: mesh + blink + mood + head pose + EAR gauge.
    ESC to exit.
    """
    from vision.camera import run_loop
    from vision.draw import (draw_face_mesh, draw_face_box, draw_ear_gauge,
                              draw_status_bar, draw_corner_badge,
                              mood_color, hud_color, _CYAN, _WHITE, _GREEN, _RED, _AMBER)
    from vision.smoothing import EMA, MajorityBuffer
    _ear_ema  = EMA()
    _mood_buf = MajorityBuffer(maxlen=15)
    blink_proc, blink_state = make_blink_processor()

    def process(frame):
        h, w = frame.shape[:2]

        # face landmarks + mesh
        fls = get_face_landmarks(frame)
        _draw_all_meshes(frame, fls, h, w)

        # blink / drowsy (includes its own mesh for MP path)
        blink_proc(frame)

        # mood
        m        = mood_of(frame)
        m_stable = _mood_buf.update(m)
        m_col    = mood_color(m_stable)

        # head zone
        zone = "no_face"
        if fls:
            zone = head_zone_landmarks(fls[0].landmark, w, h)
        elif detect_faces(frame):
            zone = head_zone(detect_faces(frame)[0], frame.shape)

        # face boxes
        for box in detect_faces(frame):
            draw_face_box(frame, box, m_col)

        # EAR gauge
        ear = blink_state.get("ear", 0.3)
        draw_ear_gauge(frame, ear, (10, 12))

        # status bar
        col_zone = _AMBER if zone not in ("Forward","Center","no_face") else _CYAN
        drowsy_s  = "⚠ DROWSY" if blink_state.get("drowsy") else "Awake"
        col_d     = _RED if blink_state.get("drowsy") else _GREEN
        draw_status_bar(frame, [
            (f"Mood: {m_stable}", m_col),
            (f"Head: {zone}", col_zone),
            (f"Blinks: {blink_state['blinks']}", _CYAN),
            (drowsy_s, col_d),
        ])
        draw_corner_badge(frame, BACKEND, "tr")

    run_loop(process, source=source, window="👁 Face — Full Analysis")
