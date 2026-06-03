import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh( max_num_faces=3 )

# Lip landmark indices
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Smoothing buffer
vowel_buffer = deque(maxlen=5)

# Distance function
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            upper_points = []
            lower_points = []

            # Extract lip points
            for idx in UPPER_LIP:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                upper_points.append([x, y])
            for idx in LOWER_LIP:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                lower_points.append([x, y])

            upper_points = np.array(upper_points, np.int32)
            lower_points = np.array(lower_points, np.int32)

            # Draw lip contours
            cv2.polylines(frame, [upper_points], True, (255, 255, 255), 2)
            cv2.polylines(frame, [lower_points], True, (255, 255, 255), 2)

            # Lip key points
            left = upper_points[0]
            right = upper_points[-1]
            top = upper_points[len(upper_points)//2]
            bottom = lower_points[len(lower_points)//2]

            lip_width = euclidean(left, right)
            lip_height = euclidean(top, bottom)

            # Face bounding box
            x_min = int(min([lm.x for lm in face_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in face_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in face_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in face_landmarks.landmark]) * h)
            face_w = x_max - x_min
            face_h = y_max - y_min

            # Normalized ratios
            norm_lip_width = lip_width / (face_w + 1e-6)
            norm_lip_height = lip_height / (face_h + 1e-6)
            ratio = norm_lip_height / (norm_lip_width + 1e-6)

            # ----------- VOWEL CLASSIFICATION -----------

            vowel = "Unknown"

            if norm_lip_height > 0.25 and ratio > 0.6:
                vowel = "A"
            elif norm_lip_width > 0.45 and ratio < 0.4:
                vowel = "E"   # stretched lips detected as E
            elif 0.35 <= ratio < 0.5:
                vowel = "I"
            elif 0.5 <= ratio <= 0.65:
                vowel = "O"
            elif 0.3 < ratio < 0.5 and norm_lip_width < 0.4:
                vowel = "U"

            # Smoothing
            vowel_buffer.append(vowel)
            final_vowel = max(set(vowel_buffer), key=vowel_buffer.count)

            # Bounding box
            all_points = np.vstack((upper_points, lower_points))
            x_box, y_box, w_box, h_box = cv2.boundingRect(all_points)
            cv2.rectangle(frame, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 255, 0), 2)
            cv2.putText(frame, f"Vowel: {final_vowel}", (x_box, y_box - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Lip ROI
            lip_crop = frame[y_box:y_box+h_box, x_box:x_box+w_box]
            if lip_crop.size != 0:
                cv2.imshow("Lip Region", lip_crop)

    cv2.imshow("Lip Detection + Vowel Reading", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()