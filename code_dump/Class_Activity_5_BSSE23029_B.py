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
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
            color = (255, 255, 255)

            if mar > 0.5:
                vowel = "A (Open)"
                color = (0, 255, 0)
            elif mar > 0.25 and width_to_face > 0.45:
                vowel = "E (Wide)"
                color = (255, 0, 255)
            elif mar < 0.25 and width_to_face > 0.45:
                vowel = "I (Narrow)"
                color = (255, 0, 0)
            elif m_width < (f_width * 0.35) and mar > 0.3:
                vowel = "O / U (Round)"
                color = (0, 255, 255)

            up_pts = np.array([(int(m[i].x * w), int(m[i].y * h)) for i in UPPER_LIP], np.int32)
            cv2.polylines(image, [up_pts], True, (255, 255, 0), 2)
            
            low_pts = np.array([(int(m[i].x * w), int(m[i].y * h)) for i in LOWER_LIP], np.int32)
            cv2.polylines(image, [low_pts], True, (255, 255, 0), 2)

            cv2.putText(image, f"Vowel: {vowel}", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
            cv2.putText(image, f"MAR: {round(mar, 2)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, f"Width Ratio: {round(width_to_face, 2)}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Lip Reading Activity', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()