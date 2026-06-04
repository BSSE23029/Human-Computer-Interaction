"""
vision/bridge.py -- turn a camera frame into a label dict the text engine can use.

llama3 cannot see images, so vision is summarised into words and fed into the
multimodal pipeline as context. All cv2/vision imports are LAZY so that importing
this module never crashes even if OpenCV is not installed.

    label = frame_to_label(frame)          # {present, faces, mood, head_zone, gesture}
    ctx   = vision_context_string(label)   # "A person is visible and appears smiling..."
    label = snapshot()                     # grab one webcam frame + analyse it
"""


def frame_to_label(frame) -> dict:
    """Run the enabled per-frame detectors and return a compact label dict.
    Safe to call even if cv2/vision modules are unavailable — returns a
    null label instead of crashing.
    """
    _null = {"present": False, "faces": 0, "mood": "no_face",
             "head_zone": None, "gesture": "none", "fingers": 0}
    if frame is None:
        return _null

    try:
        from vision import faces as _faces
        from vision import hands as _hands
        from core.conf import get

        face_boxes = _faces.detect_faces(frame) if get("vision.face_detect.enabled", True) else []
        present = len(face_boxes) > 0

        mood = "no_face"
        zone = None
        if present:
            if get("vision.smile_mood.enabled", True):
                mood = _faces.mood_of(frame)
            if get("vision.head_pose.enabled", True):
                zone = _faces.head_zone(face_boxes[0], frame.shape)

        count, gesture_name = 0, "none"
        if get("vision.gesture.enabled", True):
            count, _ = _hands.count_fingers(frame)
            gesture_name, _ = _hands.classify_gesture(count)
            if count == 0:
                gesture_name = "none"

        return {
            "present":   present,
            "faces":     len(face_boxes),
            "mood":      mood,
            "head_zone": zone,
            "gesture":   gesture_name,
            "fingers":   count,
        }
    except Exception as e:
        return {**_null, "_error": str(e)}


def vision_context_string(label: dict) -> str:
    """
    Human-readable sentence injected into the LLM system prompt.
    Includes all active detection outputs so llama3 has full context.
    """
    if not label or not label.get("present"):
        return "No person is visible on camera."
    bits = ["A person is visible on camera"]
    mood = label.get("mood", "")
    if mood and mood not in ("no_face", ""):
        bits.append(f"and appears {mood}")
    zone = label.get("head_zone", "")
    if zone and zone not in ("Center", "Forward", "", None):
        bits.append(f"(head turned {zone.lower()})")
    gesture = label.get("gesture", "none")
    if gesture and gesture != "none":
        bits.append(f"showing a '{gesture}' hand gesture")
    vowel = label.get("vowel", "?")
    if vowel and vowel in "AEIOU":
        bits.append(f"mouthing the vowel '{vowel}'")
    drowsy = label.get("is_drowsy", False)
    if drowsy:
        bits.append("and appears drowsy")
    return " ".join(bits) + "."


def snapshot(source=None) -> dict:
    """Grab one frame from the webcam and analyse it."""
    try:
        from vision.camera import grab_frame
        return frame_to_label(grab_frame(source))
    except Exception:
        return {"present": False, "faces": 0, "mood": "no_face",
                "head_zone": None, "gesture": "none", "fingers": 0}
