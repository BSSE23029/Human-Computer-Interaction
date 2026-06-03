import cv2
import mediapipe as mp
import time
import random

def run_hand_detection():
    """
    Module 4: Hand Detection
    Detects hands, counts fingers, classifies gestures, and runs a gesture game.
    Returns to the main menu when the ESC key is pressed.
    """
    
    # 1. Setup MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize the hand detector (max 2 hands, as per requirements)
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # 2. Setup WebCam
    cap = cv2.VideoCapture(0)

    # 3. Setup Game Variables (No global state used!)
    # We define all game variables inside the function
    possible_gestures = ["Fist", "One", "Peace", "Three", "Four", "Open Hand", "Thumbs Up"]
    target_gesture = random.choice(possible_gestures)
    score = 0
    match_start_time = 0.0  # Keeps track of when the user matched the gesture
    is_matching = False     # Is the user currently holding the right gesture?

    print("Starting Hand Detection Mode. Press 'ESC' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from webcam.")
            break
            
        # Flip frame horizontally for a mirror effect (easier for users)
        frame = cv2.flip(frame, 1)
        
        # Convert BGR image (OpenCV default) to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find hands
        results = hands.process(rgb_frame)

        # Draw Game UI at the top
        cv2.putText(frame, f"Game Target: {target_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Score: {score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Default gesture if no hand is detected
        current_gesture = "None"

        # Check if we found any hands
        if results.multi_hand_landmarks:
            
            # Loop through each hand found (up to 2)
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                # Draw the dots and lines on the hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get whether it is a Left or Right hand
                hand_label = results.multi_handedness[hand_index].classification[0].label
                
                # --- TASK 1: FINGER COUNTER ---
                # We will use simple 1s and 0s. 1 means UP, 0 means DOWN.
                thumb_up = 0
                index_up = 0
                middle_up = 0
                ring_up = 0
                pinky_up = 0

                # Index Finger (Tip is 8, PIP is 6)
                if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                    index_up = 1
                    
                # Middle Finger (Tip is 12, PIP is 10)
                if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
                    middle_up = 1
                    
                # Ring Finger (Tip is 16, PIP is 14)
                if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
                    ring_up = 1
                    
                # Pinky Finger (Tip is 20, PIP is 18)
                if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
                    pinky_up = 1

                # Thumb Logic (Using X axis as requested)
                # Because the webcam is mirrored, "Right" hand means thumb is on the left side.
                # Tip is 4, Base (MCP) is 2.
                if hand_label == "Right":
                    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x:
                        thumb_up = 1
                else: # Left Hand
                    if hand_landmarks.landmark[4].x > hand_landmarks.landmark[2].x:
                        thumb_up = 1

                # Calculate Total Fingers
                total_fingers = thumb_up + index_up + middle_up + ring_up + pinky_up

                # Get the position of the wrist (landmark 0) to display text next to the hand
                h, w, c = frame.shape
                wrist_x = int(hand_landmarks.landmark[0].x * w)
                wrist_y = int(hand_landmarks.landmark[0].y * h)

                # Display total finger count next to the hand
                cv2.putText(frame, f"Fingers: {total_fingers}", (wrist_x, wrist_y + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                # --- TASK 2: GESTURE LABEL ---
                # Figure out the gesture based on which specific fingers are up
                emoji = ""
                
                if total_fingers == 0:
                    current_gesture = "Fist"
                    emoji = "(Fist)"
                elif total_fingers == 1 and index_up == 1:
                    current_gesture = "One"
                    emoji = "(Up)"
                elif total_fingers == 1 and thumb_up == 1:
                    current_gesture = "Thumbs Up"
                    emoji = "(Y)"
                elif total_fingers == 2 and index_up == 1 and middle_up == 1:
                    current_gesture = "Peace"
                    emoji = "(V)"
                elif total_fingers == 3 and index_up == 1 and middle_up == 1 and ring_up == 1:
                    current_gesture = "Three"
                    emoji = "(3)"
                elif total_fingers == 4 and index_up == 1 and middle_up == 1 and ring_up == 1 and pinky_up == 1:
                    current_gesture = "Four"
                    emoji = "(4)"
                elif total_fingers == 5:
                    # We have 5 fingers up. Let's check the distance between Index(8) and Middle(12) tips
                    index_tip_x = hand_landmarks.landmark[8].x
                    index_tip_y = hand_landmarks.landmark[8].y
                    
                    middle_tip_x = hand_landmarks.landmark[12].x
                    middle_tip_y = hand_landmarks.landmark[12].y
                    
                    # Basic distance formula: square root of ((x2 - x1)^2 + (y2 - y1)^2)
                    x_diff = index_tip_x - middle_tip_x
                    y_diff = index_tip_y - middle_tip_y
                    distance = ((x_diff * x_diff) + (y_diff * y_diff)) ** 0.5
                    
                    if distance > 0.03:
                        current_gesture = "High Five"
                        emoji = "(High5!)"
                    else:
                        current_gesture = "Open Hand"
                        emoji = "(Flat Hand)"
                else:
                    current_gesture = "Unknown"

                # Display Gesture and Emoji
                cv2.putText(frame, f"Gesture: {current_gesture} {emoji}", (wrist_x, wrist_y + 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        # --- TASK 3: GESTURE GAME ---
        if current_gesture == target_gesture:
            # If they just started holding the correct gesture
            if is_matching == False:
                is_matching = True
                match_start_time = time.time() # Record the exact time they started

            # Calculate how long they have been holding it
            time_held = time.time() - match_start_time
            
            # Draw Progress Bar
            # Bar width goes from 0 to 200 pixels based on time held (up to 1.0 seconds)
            bar_width = int((time_held / 1.0) * 200)
            if bar_width > 200: 
                bar_width = 200 # Cap it at 200 pixels
                
            # Draw the background of the bar (gray)
            cv2.rectangle(frame, (10, 80), (210, 100), (150, 150, 150), -1)
            # Draw the filling part of the bar (green)
            cv2.rectangle(frame, (10, 80), (10 + bar_width, 100), (0, 255, 0), -1)

            # If they held it for 1 full second
            if time_held >= 1.0:
                score += 1                     # Award 1 point
                is_matching = False            # Reset matching state
                # Pick a new random target
                target_gesture = random.choice(possible_gestures) 
                
        else:
            # If they break the gesture or have the wrong one, reset the timer
            is_matching = False


        # --- DISPLAY AND EXIT LOGIC ---
        cv2.imshow("Multi-Modal Detection System", frame)

        # Wait 1 millisecond for a key press. If it's 27 (ESC key), break out.
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    # Clean up when done
    cap.release()
    cv2.destroyAllWindows()

# ---------------------------------------------------------
# Simple Menu wrapper to test the module
# (This fulfills the requirement of having a console menu)
# ---------------------------------------------------------
if __name__ == "__main__":
    while True:
        print()
        print("--- Main Menu ---")
        print("1. Lips Detection (Not Implemented Here)")
        print("2. Eyes Detection (Not Implemented Here)")
        print("3. Face Detection (Not Implemented Here)")
        print("4. Hand Detection (Module 4)")
        print("0. Exit")
        print()
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '4':
            run_hand_detection()
        elif choice == '0':
            print("Goodbye!")
            break
        else:
            print("Please select 4 to test your module, or 0 to exit.")