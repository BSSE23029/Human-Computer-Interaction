import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Lip landmark indices (better full lips)
LIPS = [61,146,91,181,84,17,314,405,321,375,291,
        61,185,40,39,37,0,267,269,270,409,291]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    mask = np.zeros_like(frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lip_points = []

            for idx in LIPS:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                lip_points.append([x, y])

            lip_points = np.array(lip_points, np.int32)

            # 1. Draw contour
            cv2.polylines(frame, [lip_points], True, (255,225, 225), 2)

            # 2. Fill lips (masking effect)
            cv2.fillPoly(mask, [lip_points], (255,225, 225))

            # 3. Extract lip region
            x, y, w_box, h_box = cv2.boundingRect(lip_points)
            lip_crop = frame[y:y+h_box, x:x+w_box]

            # Show cropped lips
            if lip_crop.size != 0:
                cv2.imshow("Lip Region", lip_crop)

    # Combine mask with original
    output = cv2.addWeighted(frame, 1, mask, 0.4, 0)

    cv2.imshow("Lip Detection", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()