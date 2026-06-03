import cv2
import mediapipe as mp
import os

# Suppress the "objc" duplicate class warnings on macOS
os.environ['QT_SILENCE_ERRORS'] = '1'
os.environ['GRPC_VERBOSITY'] = 'ERROR'

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Webcam
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces = 9,
    refine_landmarks = True,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7,
    static_image_mode = True
) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # 1. Face Mesh Tessellation
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # 2. Face Contours
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                
                # 3. IRIS
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()