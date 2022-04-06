import cv2
import numpy as np
import mediapipe as mp
from helper import *

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
count = 0

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        # if not success:
        #   print("Ignoring empty camera frame.")
        #   # If loading a video, use 'break' instead of 'continue'.
        #   continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            right_hand_landmarks = results.right_hand_landmarks.landmark
            pose_landmarks = results.pose_landmarks.landmark

            ## Get coordinates
            # # Get wrist
            # wrist = [right_hand_landmarks[mp_holistic.HandLandmark.WRIST.value].x,
            #          right_hand_landmarks[mp_holistic.HandLandmark.WRIST.value].y]
            # # Get Tips
            # thumb_tip = [right_hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP.value].x,
            #              right_hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP.value].y]
            # index_tip = [right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP.value].x,
            #              right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP.value].y]
            # middle_tip = [right_hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].x,
            #               right_hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].y]
            # ring_tip = [right_hand_landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP.value].x,
            #             right_hand_landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP.value].y]
            # pinky_tip = [right_hand_landmarks[mp_holistic.HandLandmark.PINKY_TIP.value].x,
            #              right_hand_landmarks[mp_holistic.HandLandmark.PINKY_TIP.value].y]
            # # Get MCP points
            # thumb_mcp = [right_hand_landmarks[mp_holistic.HandLandmark.THUMB_MCP.value].x,
            #              right_hand_landmarks[mp_holistic.HandLandmark.THUMB_MCP.value].y]
            # index_mcp = [right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].x,
            #              right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].y]
            # middle_mcp = [right_hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP.value].x,
            #               right_hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP.value].y]
            # ring_mcp = [right_hand_landmarks[mp_holistic.HandLandmark.RING_FINGER_MCP.value].x,
            #             right_hand_landmarks[mp_holistic.HandLandmark.RING_FINGER_MCP.value].y]
            # pinky_mcp = [right_hand_landmarks[mp_holistic.HandLandmark.PINKY_MCP.value].x,
            #              right_hand_landmarks[mp_holistic.HandLandmark.PINKY_MCP.value].y]
            # right_hand = [wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip, thumb_mcp, index_mcp,
            #               middle_mcp, ring_mcp, pinky_mcp]
            # Get face
            nose = [pose_landmarks[mp_holistic.PoseLandmark.NOSE].x,
                    pose_landmarks[mp_holistic.PoseLandmark.NOSE].y]
            left_shoulder = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
                             pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].y]
            right_shoulder = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x,
                              pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y]
            left_elbow = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].x,
                          pose_landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].y]
            right_elbow = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW].x,
                           pose_landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW].y]
            left_wrist = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].x,
                          pose_landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].y]
            right_wrist = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST].x,
                           pose_landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST].y]
            left_hip = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_HIP].x,
                        pose_landmarks[mp_holistic.PoseLandmark.LEFT_HIP].y]
            right_hip = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_HIP].x,
                         pose_landmarks[mp_holistic.PoseLandmark.RIGHT_HIP].y]
            left_knee = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_KNEE].x,
                         pose_landmarks[mp_holistic.PoseLandmark.LEFT_KNEE].y]
            right_knee = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE].x,
                          pose_landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE].y]
            left_ankle = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE].x,
                          pose_landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE].y]
            right_ankle = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE].x,
                           pose_landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE].y]
            pose = [nose, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip,
                    right_hip, left_knee, right_knee, left_ankle, right_ankle]

            # Identify sign
            hand_sign, text_origin = calculate_exercise(pose, count)
            # Visualize sign
            cv2.putText(image, str(hand_sign), (int(text_origin[0] * 640) + 40, int(text_origin[1] * 480) - 40),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        except:
            pass
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
        cv2.imshow('MediaPipe Holistic', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()