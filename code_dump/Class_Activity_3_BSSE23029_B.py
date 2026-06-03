import cv2
import mediapipe as mp

def count_fingers(hand_landmarks, handedness):
    """
    Returns a list of 5 integers (1 for UP, 0 for DOWN) representing:
    [Thumb, Index, Middle, Ring, Pinky]
    """
    fingers = []
    
    # Coordinates have (0,0) at the top-left of the image.
    
    # 1. Thumb: Check if it's extended. 
    # The logic depends on whether it's a Left or Right hand because the thumb points 
    # in opposite directions along the X-axis for each hand.
    # Note: MediaPipe's handedness is from the camera's perspective, so if we flip 
    # the image for a selfie view, Left and Right might be visually swapped unless accounted for.
    is_right_hand = handedness.classification[0].label == 'Right'
    
    thumb_tip_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x
    thumb_ip_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP].x
    
    # If the image is flipped (selfie view), the apparent "Right" hand (from camera's perspective)
    # is physically the user's Left hand. We usually check if tip is further away from the center
    # than the inner joint.
    if is_right_hand:
        # For a hand labeled "Right", the thumb tip is to the left (smaller X) of the inner joint
        if thumb_tip_x < thumb_ip_x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        # For a hand labeled "Left", the thumb tip is to the right (larger X) of the inner joint
        if thumb_tip_x > thumb_ip_x:
            fingers.append(1)
        else:
            fingers.append(0)
        
    # 2. Other four fingers: Compare tip y-coordinate to the PIP joint y-coordinate
    tip_ids = [
        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mp.solutions.hands.HandLandmark.PINKY_TIP
    ]
    
    pip_ids = [
        mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
        mp.solutions.hands.HandLandmark.PINKY_PIP
    ]
    
    for tip, pip in zip(tip_ids, pip_ids):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)  # Finger is extending upwards
        else:
            fingers.append(0)  # Finger is curled downwards
            
    return fingers

def main():
    # 1. Setup MediaPipe components
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize the webcam (Video capture)
    cap = cv2.VideoCapture(0)
    
    # Configure hands module
    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
                
            # Flip image horizontally for a selfie-view display and convert color scheme to RGB
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and find hands
            results = hands.process(image_rgb)
            
            gesture = "Searching..."
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # 2. Detection: Draw 21 landmarks on the detected hand
                    mp_draw.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # 3. Feature Extraction: Determine which fingers are up, passing handedness
                    fingers = count_fingers(hand_landmarks, handedness)
                    
                    # 4. Rule Engine: Map finger states to specific gesture names
                    # fingers array = [Thumb, Index, Middle, Ring, Pinky]
                    
                    if fingers == [0, 1, 1, 0, 0]:
                        gesture = "Peace"
                    elif fingers == [1, 0, 0, 0, 0]:
                        gesture = "Thumbs Up"
                    elif fingers == [1, 1, 1, 1, 1] or fingers == [0, 1, 1, 1, 1]: 
                        # Sometimes thumb is ignored or considered down when palm is open depending on angle
                        gesture = "Stop"
            
            # 5. Display: Overlay the recognized gesture name on the video feed
            cv2.putText(image, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(image, '(Press ESC to exit)', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show the image
            cv2.imshow('Hand Gesture Recognition', image)
            
            # Press 'ESC' key to exit
            if cv2.waitKey(5) & 0xFF == 27: 
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
