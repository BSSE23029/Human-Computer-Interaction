import cv2
import mediapipe as mp
# import numpy as np
# from deepface import DeepFace
# import time
import math
# import collections
# import random
# import threading
import queue
# import os
import gc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Immutable mapping of gestures to emojis
GESTURE_EMOJI_MAP = {
    "Fist": "✊",
    "One": "☝️",
    "Peace": "✌️",
    # "Three": "3️⃣",
    # "Four": "4️⃣",
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
# MODULE 1 – LIPS DETECTION
# ---------------------------------------------------------------------------

# def run_lips_detection(source, is_image=False):
#     """Run lips detection on the provided video or image source.

#     Args:
#         source: Device index or file path.
#         is_image (bool): True if source is an image.
#     """
#     state_closure = {'lip_sync_count': 0, 'mouth_open': False}

#     with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
#         def process(frame, rgb_frame, paused, is_img):
#             results = face_mesh.process(rgb_frame)
#             h, w = frame.shape[:2]
#             mar = 0.0
#             is_smiling = False

#             if results.multi_face_landmarks:
#                 lm = results.multi_face_landmarks[0].landmark
#                 p78  = (int(lm[78].x  * w), int(lm[78].y  * h))
#                 p308 = (int(lm[308].x * w), int(lm[308].y * h))
#                 p13  = (int(lm[13].x  * w), int(lm[13].y  * h))
#                 p14  = (int(lm[14].x  * w), int(lm[14].y  * h))

#                 horiz = calculate_distance(p78, p308)
#                 vert  = calculate_distance(p13, p14)
#                 if horiz > 0:
#                     mar = vert / horiz
                    
#                 xs = [int(p.x * w) for p in lm]
#                 face_w = max(xs) - min(xs)
                
#                 # Smile logic dynamically relative to face width
#                 if p78[1] < p14[1] and p308[1] < p14[1] and horiz > face_w * 0.35:
#                     is_smiling = True

#                 for pt, col in [(p78, (0,255,0)), (p308, (0,255,0)), (p13, (255,0,0)), (p14, (255,0,0))]:
#                     cv2.circle(frame, pt, 2, col, -1)

#             # Rendering UI and checking states
#             draw_ui_text(frame, f"Expression: {'Smiling 😊' if is_smiling else 'Neutral 😐'}", (10, 30))
#             border_color = (0, 0, 255) if mar > 0.55 else (0, 255, 0)
            
#             if mar > 0.55:
#                 cv2.rectangle(frame, (0, 0), (w, h), border_color, 10)
#             draw_ui_text(frame, f"MAR: {mar:.2f}", (10, 70), color=border_color)

#             if not is_img and not paused:
#                 if mar > 0.55 and not state_closure['mouth_open']:
#                     state_closure['mouth_open'] = True
#                 elif mar < 0.35 and state_closure['mouth_open']:
#                     state_closure['mouth_open'] = False
#                     state_closure['lip_sync_count'] += 1

#             if not is_img:
#                 draw_ui_text(frame, f"Lip-Sync Cycles: {state_closure['lip_sync_count']}", (10, 110))

#         run_detection_pipeline(source, is_image, "Lips Detection", process)

# ---------------------------------------------------------------------------
# MODULE 2 – EYES DETECTION
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
# MODULE 3 – FACE DETECTION
# ---------------------------------------------------------------------------

# def deepface_worker(frame_queue, result_queue, stop_event):
#     """Worker daemon thread for background processing of DeepFace emotion analysis.

#     Args:
#         frame_queue (queue.Queue): Enters BGR frames to process.
#         result_queue (queue.Queue): Returns (dominant_emotion, confidence(float)).
#         stop_event (threading.Event): Signals the daemon to shut down.
#     """
#     while not stop_event.is_set():
#         try:
#             frame = frame_queue.get(timeout=0.5)
#         except queue.Empty:
#             continue

#         if frame is None:
#             break

#         try:
#             res = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
#             if isinstance(res, list): res = res[0]
#             dominant = res['dominant_emotion']
#             conf     = res['emotion'][dominant]
            
#             drain_queue(result_queue)
#             result_queue.put((dominant, conf))
#         except Exception:
#             drain_queue(result_queue)
#             result_queue.put(("Unknown", 0.0))
#         finally:
#             del frame

# def run_face_detection(source, is_image=False):
#     """Run face emotion and head pose detection.

#     Args:
#         source: Device index or file path.
#         is_image (bool): True if source is an image.
#     """
#     frame_q  = queue.Queue(maxsize=1)
#     result_q = queue.Queue(maxsize=1)
#     stop_evt = threading.Event()

#     worker = threading.Thread(target=deepface_worker, args=(frame_q, result_q, stop_evt), daemon=True)
#     worker.start()

#     state = {
#         'emotion_hist': collections.deque(maxlen=150),
#         'current_emotion': "Loading...",
#         'conf': 0.0,
#         'frame_count': 0
#     }

#     with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
#         try:
#             def process(frame, rgb_frame, paused, is_img):
#                 h, w = frame.shape[:2]

#                 if is_img:
#                     drain_queue(frame_q)
#                     frame_q.put(frame.copy())
#                     try:
#                         state['current_emotion'], state['conf'] = result_q.get(timeout=5)
#                     except queue.Empty:
#                         state['current_emotion'], state['conf'] = "Unknown", 0.0
#                 else:
#                     state['frame_count'] += 1
#                     if not paused and state['frame_count'] % 5 == 0 and frame_q.empty():
#                         try:
#                             frame_q.put_nowait(frame.copy())
#                         except queue.Full:
#                             pass

#                     if not result_q.empty():
#                         try:
#                             state['current_emotion'], state['conf'] = result_q.get_nowait()
#                             state['emotion_hist'].append((time.time(), state['current_emotion']))
#                         except queue.Empty:
#                             pass

#                     cutoff = time.time() - 5.0
#                     while state['emotion_hist'] and state['emotion_hist'][0][0] < cutoff:
#                         state['emotion_hist'].popleft()

#                 recent_mood = (collections.Counter(e for _, e in state['emotion_hist']).most_common(1)[0][0] 
#                                if state['emotion_hist'] else "Unknown")

#                 # Head Pose
#                 results = face_mesh.process(rgb_frame)
#                 head_pose = "Forward"
#                 if results.multi_face_landmarks:
#                     lm = results.multi_face_landmarks[0].landmark
#                     nose, l_eye, r_eye = lm[1], lm[33], lm[263]
#                     eye_mid_y = (l_eye.y + r_eye.y) / 2.0
                    
#                     xs = [int(p.x * w) for p in lm]
#                     ys = [int(p.y * h) for p in lm]
#                     x_min, x_max = min(xs), max(xs)
#                     y_min, y_max = min(ys), max(ys)
#                     face_h = y_max - y_min
                    
#                     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    
#                     if   nose.x < l_eye.x:            head_pose = "Left"
#                     elif nose.x > r_eye.x:            head_pose = "Right"
#                     elif nose.y < eye_mid_y - (face_h * 0.05 / h):   head_pose = "Up"
#                     elif nose.y > eye_mid_y + (face_h * 0.05 / h):   head_pose = "Down"

#                     cv2.circle(frame, (int(nose.x*w),  int(nose.y*h)),  4, (0, 0, 255), -1)
#                     cv2.circle(frame, (int(l_eye.x*w), int(l_eye.y*h)), 2, (255, 0, 0), -1)
#                     cv2.circle(frame, (int(r_eye.x*w), int(r_eye.y*h)), 2, (255, 0, 0), -1)

#                 draw_ui_text(frame, f"Emotion: {state['current_emotion']} ({state['conf']:.1f}%)", (10, 30))
#                 if not is_img:
#                     draw_ui_text(frame, f"Head Pose: {head_pose}", (10, 70), color=(0,255,255))
#                     draw_ui_text(frame, f"Recent Mood (5s): {recent_mood}", (10, 110), color=(255,0,255))

#             run_detection_pipeline(source, is_image, "Face Detection", process)
#         finally:
#             stop_evt.set()
#             drain_queue(frame_q)
#             frame_q.put(None)
#             worker.join(timeout=3.0)
#             drain_queue(result_q)

# ---------------------------------------------------------------------------
# MODULE 4 – HAND DETECTION
# ---------------------------------------------------------------------------

def run_hand_detection(source, is_image=False):
    """Run interactive hand gesture detection.

    Args:
        source: Device index or file path.
        is_image (bool): True if source is an image.
    """
    possible_gestures = list(GESTURE_EMOJI_MAP.keys())
    state = {
        # 'target_gesture': random.choice(possible_gestures),
        # 'score': 0,
        'match_start': 0.0,
        'is_matching': False
    }

    with mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
        def process(frame, rgb_frame, paused, is_img):
            res = hands.process(rgb_frame)
            h, w = frame.shape[:2]
            
            # current_gesture = "None"
            
            # if not is_img:
            #     draw_ui_text(frame, f"Target: {state['target_gesture']} | Score: {state['score']}",
            #                  (10, 30), color=(255, 255, 0))

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
                        
                    # current_gesture = gesture
                    gesture_display = f"{gesture} {GESTURE_EMOJI_MAP.get(gesture)}"
                    
                    
                    draw_ui_text(frame, f"Fingers: {cnt} | {gesture_display}",
                                 (int(hand_lm.landmark[0].x*w), int(hand_lm.landmark[0].y*h)))

            # if not is_img and not paused:
            #     if current_gesture == state['target_gesture']:
            #         if not state['is_matching']:
            #             state['is_matching'] = True
            #             state['match_start'] = time.time()
                    
            #         held = time.time() - state['match_start']
            #         bar_w = min(int(held * 200), 200)
            #         cv2.rectangle(frame, (10, 50), (10 + bar_w, 70), (0, 255, 0), -1)
                    
            #         rem_sec = max(0.0, 1.0 - held)
            #         draw_ui_text(frame, f"{rem_sec:.1f}s", (10 + bar_w + 5, 65), font_scale=0.5)
                    
            #         if held >= 1.0:
            #             state['score'] += 1
            #             state['target_gesture'] = random.choice(possible_gestures)
            #             state['is_matching'] = False
            #     else:
            #         state['is_matching'] = False

        run_detection_pipeline(source, is_image, "Hand Detection", process)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main_menu():
    """Display main menu and launch requested detection module."""
    while True:
        print("\n=== Interactive Detection System ===")
        # print("1. Lips Detection")
        print("1. Eyes Detection")
        # print("3. Face Detection")
        print("2. Hand Detection")
        print("0. Exit")
        choice = input("Select mode (0-2): ").strip()

        if choice == '0':
            break
        if choice not in ('1','2'):
            print("Invalid choice. Try again.")
            continue

        # print("\nInput Source:")
        # print("1. Webcam Realtime")
        # print("2. Video File Upload")
        # print("3. Image File Upload")
        # src_choice = input("Select source (1-3): ").strip()

        source  = 0
        is_img  = False

        # if src_choice == '2':
        #     source = input("Enter video path: ").strip(' "\'')
        #     if not os.path.exists(source):
        #         print(f"Error: not found → {source}")
        #         continue
        # elif src_choice == '3':
        #     source = input("Enter image path: ").strip(' "\'')
        #     is_img = True
        #     if not os.path.exists(source):
        #         print(f"Error: not found → {source}")
        #         continue
        # elif src_choice != '1':
        #     print("Invalid choice.")
        #     continue

        # if   choice == '1': run_lips_detection(source, is_img)
        if choice == '1': run_eyes_detection(source, is_img)
        # elif choice == '3': run_face_detection(source, is_img)
        elif choice == '2': run_hand_detection(source, is_img)

        gc.collect()

if __name__ == "__main__":
    main_menu()
