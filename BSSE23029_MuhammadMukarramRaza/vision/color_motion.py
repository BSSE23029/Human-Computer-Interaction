"""
vision/color_motion.py -- colour tracking + motion detection (pure OpenCV).

    res = detect_color(frame, "red")       # {'present','centroid','area','bbox'}
    md = MotionDetector(); out = md.update(frame)   # {'motion','level','boxes'}
    run_color("red");  run_motion()        # live windows
"""
from core.conf import get
from vision.camera import draw_text


def detect_color(bgr, preset=None, lower=None, upper=None):
    """Find the largest blob of a colour. `preset` names a config.color_presets entry,
    or pass explicit HSV `lower`/`upper`. Returns presence + location info."""
    import cv2
    import numpy as np
    if preset:
        p = (get("vision.color_presets") or {}).get(preset, {})
        lower, upper = p.get("lower"), p.get("upper")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"present": False, "centroid": None, "area": 0, "bbox": None}
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < get("vision.motion.min_area", 500):
        return {"present": False, "centroid": None, "area": int(area), "bbox": None}
    x, y, w, h = cv2.boundingRect(c)
    M = cv2.moments(c)
    cx = int(M["m10"] / M["m00"]) if M["m00"] else x + w // 2
    cy = int(M["m01"] / M["m00"]) if M["m00"] else y + h // 2
    return {"present": True, "centroid": (cx, cy), "area": int(area), "bbox": (x, y, w, h)}


class MotionDetector:
    """Frame-difference OR MOG2 background-subtraction motion detector."""

    def __init__(self, method: str = None):
        self.method = method or get("vision.motion.method", "diff")
        self.prev = None
        self.bg = None
        if self.method == "mog2":
            import cv2
            self.bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    def update(self, frame):
        """Return {'motion':bool, 'level':float(0..1), 'boxes':[(x,y,w,h)]}."""
        import cv2
        import numpy as np
        thresh = get("vision.motion.threshold", 25)
        min_area = get("vision.motion.min_area", 500)

        if self.method == "mog2":
            fg = self.bg.apply(frame)
            _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if self.prev is None:
                self.prev = gray
                return {"motion": False, "level": 0.0, "boxes": []}
            delta = cv2.absdiff(self.prev, gray)
            _, fg = cv2.threshold(delta, thresh, 255, cv2.THRESH_BINARY)
            fg = cv2.dilate(fg, None, iterations=2)
            self.prev = gray

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) >= min_area]
        level = float(np.count_nonzero(fg)) / fg.size if fg.size else 0.0
        return {"motion": len(boxes) > 0, "level": round(level, 4), "boxes": boxes}


def run_color(preset: str = "red", source=None):
    """Live colour-tracking window."""
    import cv2
    from vision.camera import run_loop

    def process(frame):
        res = detect_color(frame, preset=preset)
        if res["present"]:
            x, y, w, h = res["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(frame, res["centroid"], 5, (0, 0, 255), -1)
            draw_text(frame, f"{preset}: {res['centroid']}", (10, 34))
        else:
            draw_text(frame, f"{preset}: not found", (10, 34), color=(0, 0, 255))

    run_loop(process, source=source, window=f"Color Track ({preset})")


def run_motion(source=None):
    """Live motion-detection window."""
    import cv2
    from vision.camera import run_loop
    md = MotionDetector()

    def process(frame):
        out = md.update(frame)
        for (x, y, w, h) in out["boxes"]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"MOTION ({out['level']*100:.1f}%)" if out["motion"] else "still"
        draw_text(frame, label, (10, 34),
                  color=(0, 255, 0) if out["motion"] else (180, 180, 180))

    run_loop(process, source=source, window="Motion Detection")
