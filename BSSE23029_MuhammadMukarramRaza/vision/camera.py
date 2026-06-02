"""
vision/camera.py -- the reusable webcam/video scaffold (pure OpenCV, NO MediaPipe).

Ported from the proven VideoSource + run_detection_pipeline pattern. You write a
tiny `process(frame)` callback that annotates the frame in place; this handles
capture, mirror-flip, FPS, pause (SPACE), replay (R, video files), ESC to exit,
and guaranteed camera release.

    from vision.camera import run_loop, draw_text
    def process(frame):
        draw_text(frame, "Hello", (10, 30))
    run_loop(process, window="Demo")
"""
import math
import time

from core.conf import get


class VideoSource:
    """cv2.VideoCapture wrapper that guarantees release and mirrors the webcam."""

    def __init__(self, source):
        import cv2
        self._source = source
        self._cap = cv2.VideoCapture(source)
        self._last = None
        self._flip = get("vision.flip_webcam", True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    @property
    def is_webcam(self) -> bool:
        return isinstance(self._source, int)

    def read(self):
        import cv2
        ok, frame = self._cap.read()
        if ok:
            if self.is_webcam and self._flip:
                frame = cv2.flip(frame, 1)
            self._last = frame
            return True, frame
        return (False, self._last.copy() if self._last is not None else None)

    def seek_start(self):
        import cv2
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        self._last = None


def draw_text(frame, text, pos, color=(0, 255, 0), scale=0.7, thickness=2, bg=True):
    """Draw readable text with an optional dark background box."""
    import cv2
    if not text:
        return
    x, y = int(pos[0]), int(pos[1])
    if bg:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def distance(p1, p2) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def run_loop(process, source=None, window: str = None, on_key=None, show_fps: bool = True):
    """Standard live loop. `process(frame)` annotates the BGR frame in place.

    Keys: ESC = exit, SPACE = pause/play, R = replay (video files only).
    `on_key(key)` (optional) handles any other keypress.
    """
    import cv2
    source = get("vision.webcam_index", 0) if source is None else source
    window = window or get("ui.app_title", "HCI")
    exit_key = get("ui.exit_key", 27)
    wait_ms = get("vision.wait_key_ms", 1)

    cap = VideoSource(source)
    paused = False
    fps, frames, t0 = 0.0, 0, time.time()
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if cap.is_webcam:
                        print("[camera] no frame — is the webcam free / index correct?")
                        break
                    cap.seek_start()
                    continue
                frames += 1
                if time.time() - t0 >= 1.0:
                    fps, frames, t0 = frames / (time.time() - t0), 0, time.time()

                process(frame)
                if show_fps:
                    draw_text(frame, f"FPS: {fps:4.1f}", (frame.shape[1] - 110, 24),
                              color=(180, 180, 180), scale=0.5, thickness=1)
                draw_text(frame, get("ui.controls_hint", ""),
                          (10, frame.shape[0] - 12), color=(200, 200, 200),
                          scale=0.5, thickness=1)
                cv2.imshow(window, frame)

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == exit_key:
                break
            elif key == ord(" "):
                paused = not paused
            elif key in (ord("r"), ord("R")) and not cap.is_webcam:
                cap.seek_start()
            elif on_key and key != 255:
                on_key(key)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def grab_frame(source=None):
    """Grab a single frame (for gradio snapshot-style use). Returns BGR frame or None."""
    source = get("vision.webcam_index", 0) if source is None else source
    cap = VideoSource(source)
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()
