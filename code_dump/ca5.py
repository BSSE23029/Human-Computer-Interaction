# Muhammad Mukarram Raza - BSSE-23029
# HCI Activity: Lip Detection & Vowel Classification

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 61]
LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61]

def get_dist(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    h, w, _ = frame.shape
    blank_screen = np.zeros((h, w, 3), dtype=np.uint8)
    
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            m = face_landmarks.landmark
            
            m_height = get_dist(m[13], m[14])
            m_width = get_dist(m[78], m[308])
            f_width = get_dist(m[234], m[454])
            mar = m_height / m_width if m_width != 0 else 0
            width_to_face = m_width / f_width

            vowel = "Unknown"
            v_color = (255, 255, 255)
            if mar > 0.5:
                vowel = "A (Open)"
                v_color = (0, 255, 0)
            elif mar > 0.22 and width_to_face > 0.42:
                vowel = "E (Wide)"
                v_color = (255, 0, 255)
            elif mar < 0.22 and width_to_face > 0.42:
                vowel = "I (Narrow)"
                v_color = (255, 100, 0)
            elif m_width < (f_width * 0.38) and mar > 0.25:
                vowel = "O / U (Round)"
                v_color = (0, 255, 255)

            up_pts = np.array([(int(m[i].x * w), int(m[i].y * h)) for i in UPPER_LIP], np.int32)
            cv2.polylines(blank_screen, [up_pts], True, (0, 0, 255), 2)
            
            low_pts = np.array([(int(m[i].x * w), int(m[i].y * h)) for i in LOWER_LIP], np.int32)
            cv2.polylines(blank_screen, [low_pts], True, (255, 255, 0), 2)

            cv2.putText(blank_screen, f"Vowel: {vowel}", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, v_color, 2)
            cv2.putText(blank_screen, "PRIVACY MODE: Face Hidden", (w-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cv2.imshow('Lip Detection System (Landmarks Only)', blank_screen)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()