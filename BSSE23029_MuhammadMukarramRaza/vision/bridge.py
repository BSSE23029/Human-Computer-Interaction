"""
vision/bridge.py -- turn a camera frame into a label dict the TEXT engine can use.

llama3 cannot see images, so vision is summarised into words and fed into the
multimodal pipeline as context.

    label = frame_to_label(frame)          # {present, faces, mood, head_zone, gesture}
    ctx   = vision_context_string(label)   # "The user is present and appears smiling..."
    label = snapshot()                     # grab one webcam frame + analyse it
"""
from vision import faces as _faces
from vision import hands as _hands


def frame_to_label(frame) -> dict:
    """Run the cheap per-frame detectors and return a compact label dict."""
    if frame is None:
        return {"present": False, "faces": 0, "mood": "no_face",
                "head_zone": None, "gesture": "none", "fingers": 0}
    face_boxes = _faces.detect_faces(frame)
    present = len(face_boxes) > 0
    mood = _faces.mood_of(frame) if present else "no_face"
    zone = _faces.head_zone(face_boxes[0], frame.shape) if present else None
    count, _ = _hands.count_fingers(frame)
    gesture, _emoji = _hands.classify_gesture(count)
    return {
        "present": present,
        "faces": len(face_boxes),
        "mood": mood,
        "head_zone": zone,
        "gesture": gesture if count > 0 else "none",
        "fingers": count,
    }


def vision_context_string(label: dict) -> str:
    """Human-readable sentence for an LLM prompt (so llama3 'knows' what the camera sees)."""
    if not label or not label.get("present"):
        return "No person is visible on camera."
    bits = ["A person is visible on camera"]
    if label.get("mood") and label["mood"] != "no_face":
        bits.append(f"and appears {label['mood']}")
    if label.get("head_zone") and label["head_zone"] != "Center":
        bits.append(f"(head turned {label['head_zone'].lower()})")
    if label.get("gesture") and label["gesture"] != "none":
        bits.append(f"showing a '{label['gesture']}' hand gesture")
    return " ".join(bits) + "."


def snapshot(source=None) -> dict:
    """Grab one frame from the webcam and analyse it (for gradio image input)."""
    from vision.camera import grab_frame
    return frame_to_label(grab_frame(source))
