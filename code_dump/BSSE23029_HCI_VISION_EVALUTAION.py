import cv2
import mediapipe as mp
import math
import queue
import gc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Immutable mapping of gestures to emojis
GESTURE_EMOJI_MAP = {
    "Fist": "✊",
    "One": "☝️",
    "Peace": "✌️",
    "Open Hand": "🖐",
    "Thumbs Up": "👍",
    "High Five": "🙌"
}

# ---------------------------------------------------------------------------
# Utilities & Core Pipeline
# ---------------------------------------------------------------------------

def draw_ui_text(frame, text, position, color=(0, 255, 0), font_scale=0.7, thickness=2):
    """Draw text with a solid black background for visibility.

    Args:
        frame (numpy.ndarray): The image frame.
        text (str): The text to draw.
        position (tuple): (x, y) coordinates.
        color (tuple): BGR color tuple.
        font_scale (float): Scale of the font.
        thickness (int): Thickness of the font.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x, y - th - 5), (x + tw, y + baseline + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points.

    Args:
        p1 (tuple): Point 1 (x, y).
        p2 (tuple): Point 2 (x, y).

    Returns:
        float: Euclidean distance.
    """
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def drain_queue(q):
    """Discard every item currently sitting in a Queue without blocking.

    Args:
        q (queue.Queue): The queue to drain.
    """
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break

class VideoSource:
    """Wrapper around cv2.VideoCapture that guarantees resource release."""

    def __init__(self, source):
        """Initialize the video source.
        
        Args:
            source (int or str): Device index or file path.
        """
        self._source = source
        self._cap = cv2.VideoCapture(source)
        self._last_frame = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    @property
    def is_webcam(self):
        return self._source == 0

    def read(self):
        """Read the next frame.
        
        Returns:
            tuple: (success_boolean, frame_array)
        """
        ret, frame = self._cap.read()
        if ret:
            if self.is_webcam:
                frame = cv2.flip(frame, 1)
            del self._last_frame
            self._last_frame = frame
            return True, frame
        else:
            if self._last_frame is not None:
                return False, self._last_frame.copy()
            return False, None

    def seek_start(self):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self._cap.isOpened():
            self._cap.release()
        del self._last_frame
        self._last_frame = None
        gc.collect()

def run_detection_pipeline(source, is_image, window_name, process_callback, reset_callback=None):
    """Standardized video/image processing loop to eliminate boilerplate across modules.

    Args:
        source (int or str): Device index or file path.
        is_image (bool): True if source is an image.
        window_name (str): Title of the OpenCV window.
        process_callback (callable): Function taking (frame, rgb_frame, paused, is_image).
        reset_callback (callable, optional): Function to reset state on rewind.
    """
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print(f"Cannot read image: {source}")
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        process_callback(frame, rgb_frame, False, True)
        cv2.imshow(window_name, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)
        gc.collect()
        return

    paused = False
    with VideoSource(source) as vs:
        try:
            while True:
                if not paused:
                    ok, frame = vs.read()
                    if frame is None or (not ok and vs.is_webcam):
                        break
                    if not ok:
                        paused = True
                else:
                    _, frame = vs.read()

                if frame is None:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Execute module-specific detections
                process_callback(frame, rgb_frame, paused, False)

                if not vs.is_webcam:
                    draw_ui_text(frame, "[SPACE] Pause/Play  [R] Replay  [ESC] Exit",
                                 (10, frame.shape[0] - 20), color=(255, 255, 255))

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(100 if paused else 30) & 0xFF
                
                if key == 27:
                    break
                elif key == ord(' ') and not vs.is_webcam:
                    paused = not paused
                elif key == ord('r') and not vs.is_webcam:
                    vs.seek_start()
                    paused = False
                    if reset_callback:
                        reset_callback()
        finally:
            cv2.destroyAllWindows()
            for _ in range(5): cv2.waitKey(1)
            gc.collect()

# ---------------------------------------------------------------------------
# MODULE 1 – EYES DETECTION
# ---------------------------------------------------------------------------

def run_eyes_detection(source, is_image=False):
    """Run eyes detection on the provided video or image source.

    Args:
        source: Device index or file path.
        is_image (bool): True if source is an image.
    """
    state_closure = {'blinks': 0, 'eye_closed': False}

    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        def process(frame, rgb_frame, paused, is_img):
            results = face_mesh.process(rgb_frame)
            h, w = frame.shape[:2]
            left_ear = right_ear = avg_ear = 0.0

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                def pt(idx): return lm[idx].x * w, lm[idx].y * h
                
                p1l, p2l, p3l, p4l, p5l, p6l = pt(33), pt(160), pt(158), pt(133), pt(153), pt(144)
                p1r, p2r, p3r, p4r, p5r, p6r = pt(362), pt(385), pt(387), pt(263), pt(373), pt(380)

                hl = calculate_distance(p1l, p4l)
                left_ear  = (calculate_distance(p2l, p6l) + calculate_distance(p3l, p5l)) / (2 * hl) if hl else 0.0
                hr = calculate_distance(p1r, p4r)
                right_ear = (calculate_distance(p2r, p6r) + calculate_distance(p3r, p5r)) / (2 * hr) if hr else 0.0
                avg_ear = (left_ear + right_ear) / 2.0

                for pt_ in [p1l,p2l,p3l,p4l,p5l,p6l, p1r,p2r,p3r,p4r,p5r,p6r]:
                    cv2.circle(frame, (int(pt_[0]), int(pt_[1])), 2, (0, 255, 0), -1)

            if not is_img and not paused and avg_ear > 0:
                if avg_ear < 0.25:
                    if not state_closure['eye_closed']:
                        state_closure['eye_closed'] = True
                # > 0.25 case
                else:
                    if state_closure['eye_closed']:
                        state_closure['eye_closed'] = False
                        state_closure['blinks'] += 1
                    
            draw_ui_text(frame, f"Blinks: {state_closure['blinks']}", (10, 30))
            draw_ui_text(frame, f"L-EAR: {left_ear:.3f} | R-EAR: {right_ear:.3f}", (10, 70))

        def reset():
            state_closure['eye_closed'] = False
            
        run_detection_pipeline(source, is_image, "Eyes Detection", process, reset)

# ---------------------------------------------------------------------------
# MODULE 2 – HAND DETECTION
# ---------------------------------------------------------------------------

def run_hand_detection(source, is_image=False):
    """Run interactive hand gesture detection.

    Args:
        source: Device index or file path.
        is_image (bool): True if source is an image.
    """
    possible_gestures = list(GESTURE_EMOJI_MAP.keys())
    state = {
        'match_start': 0.0,
        'is_matching': False
    }

    with mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
        def process(frame, rgb_frame, paused, is_img):
            res = hands.process(rgb_frame)
            h, w = frame.shape[:2]

            if res.multi_hand_landmarks:
                for hand_lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_lm, mp.solutions.hands.HAND_CONNECTIONS)
                    label = handedness.classification[0].label
                    
                    cnt = sum(1 for (tip,pip) in [(8,6),(12,10),(16,14),(20,18)]
                              if hand_lm.landmark[tip].y < hand_lm.landmark[pip].y)
                    
                    if label == "Right" and hand_lm.landmark[4].x < hand_lm.landmark[2].x: cnt += 1
                    elif label == "Left"  and hand_lm.landmark[4].x > hand_lm.landmark[2].x: cnt += 1
                    
                    gesture = {0:"Fist", 1:"One", 2:"Peace", 5:"Open Hand"}.get(cnt, "Unknown")
                    if cnt == 1 and hand_lm.landmark[4].y < hand_lm.landmark[3].y and hand_lm.landmark[8].y > hand_lm.landmark[6].y:
                        gesture = "Thumbs Up"
                    if cnt == 5 and hand_lm.landmark[0].y * h < h / 1.6:
                        gesture = "High Five"
                        
                    gesture_display = f"{gesture} {GESTURE_EMOJI_MAP.get(gesture)}"
                    
                    draw_ui_text(frame, f"Fingers: {cnt} | {gesture_display}",
                                 (int(hand_lm.landmark[0].x*w), int(hand_lm.landmark[0].y*h)))

        run_detection_pipeline(source, is_image, "Hand Detection", process)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main_menu():
    """Display main menu and launch requested detection module."""
    while True:
        print("\n=== Interactive Detection System ===")
        print("1. Eyes Detection")
        print("2. Hand Detection")
        print("0. Exit")
        choice = input("Select mode (0-2): ").strip()

        if choice == '0':
            break
        if choice not in ('1','2'):
            print("Invalid choice. Try again.")
            continue

        source  = 0
        is_img  = False

        if    choice == '1': run_eyes_detection(source, is_img)
        elif  choice == '2': run_hand_detection(source, is_img)

        gc.collect()

if __name__ == "__main__":
    main_menu()
