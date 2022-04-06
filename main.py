import cv2
import numpy as np
import mediapipe as mp
from helper import *

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
hands = []
count = 0
print("Choose:-\n1.Basic Hand \n2.Basic Exercises")
menu = int(input())
print("Your choice:- " + str(menu))

if menu == 1:
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
                landmarks = results.right_hand_landmarks.landmark

                ## Get coordinates
                # Get wrist
                wrist = [landmarks[mp_holistic.HandLandmark.WRIST.value].x,
                         landmarks[mp_holistic.HandLandmark.WRIST.value].y]
                # Get Tips
                thumb_tip = [landmarks[mp_holistic.HandLandmark.THUMB_TIP.value].x,
                             landmarks[mp_holistic.HandLandmark.THUMB_TIP.value].y]
                index_tip = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP.value].x,
                             landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP.value].y]
                middle_tip = [landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].x,
                              landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].y]
                ring_tip = [landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP.value].x,
                            landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP.value].y]
                pinky_tip = [landmarks[mp_holistic.HandLandmark.PINKY_TIP.value].x,
                             landmarks[mp_holistic.HandLandmark.PINKY_TIP.value].y]
                # Get MCP points
                thumb_mcp = [landmarks[mp_holistic.HandLandmark.THUMB_MCP.value].x,
                             landmarks[mp_holistic.HandLandmark.THUMB_MCP.value].y]
                index_mcp = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].x,
                             landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].y]
                middle_mcp = [landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP.value].x,
                              landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP.value].y]
                ring_mcp = [landmarks[mp_holistic.HandLandmark.RING_FINGER_MCP.value].x,
                            landmarks[mp_holistic.HandLandmark.RING_FINGER_MCP.value].y]
                pinky_mcp = [landmarks[mp_holistic.HandLandmark.PINKY_MCP.value].x,
                             landmarks[mp_holistic.HandLandmark.PINKY_MCP.value].y]

                # Calculate hand signs
                def calculate_hand_sign(wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip, thumb_mcp,
                                        index_mcp, middle_mcp,
                                        ring_mcp, pinky_mcp):
                    """
                    SIGNS--> 'YO' 'thumbs up' 'thumbs down' 'call' 'rock' 'paper' 'scissor'
                    BASELINE --> the MCP point is the base line for each index
                    :param wrist: x, y coordinates
                    :param thumb_tip: x, y coordinates
                    :param index_tip: x, y coordinates
                    :param middle_tip: x, y coordinates
                    :param ring_tip: x, y coordinates
                    :param pinky_tip: x, y coordinates
                    :param thumb_mcp: x, y coordinates
                    :param index_mcp: x, y coordinates
                    :param middle_mcp: x, y coordinates
                    :param ring_mcp: x, y coordinates
                    :param pinky_mcp: x, y coordinates
                    :return: whatever sign is identified
                    """
                    ## STATE ~ SIGN
                    states = ['YO', 'thumbs up', 'thumbs down', 'call', 'Perfect', 'rock', 'paper', 'scissor']
                    state = ''
                    dist_middle_finger = ((((middle_tip[0] - middle_mcp[0]) ** 2) + (
                                (middle_tip[1] - middle_mcp[1]) ** 2)) ** 0.5)
                    dist_thumb_to_pinky = (
                                (((thumb_tip[0] - pinky_mcp[0]) ** 2) + ((thumb_tip[1] - pinky_mcp[1]) ** 2)) ** 0.5)
                    ## For 'YO' sign:
                    # index and middle tip are above baseline; ring and pinky tips are below the baseline;
                    if index_tip[1] < index_mcp[1] and middle_tip[1] < middle_mcp[1]:
                        if ring_tip[1] > ring_mcp[1] and pinky_tip[1] > pinky_mcp[1]:
                            if thumb_tip[0] < thumb_mcp[0]:
                                state = states[0]
                                return state, middle_tip

                    ## For 'Thumbs UP' sign
                    if thumb_tip[1] < thumb_mcp[1]:
                        if thumb_mcp[1] < middle_mcp[1] < ring_mcp[1] < pinky_mcp[1]:
                            if index_tip[1] < middle_tip[1] < ring_tip[1] < pinky_tip[1]:
                                if index_mcp[1] < middle_mcp[1] < ring_mcp[1] < pinky_mcp[1]:
                                    state = states[1]
                                    return state, middle_tip

                    ## For 'Thumbs DOWN' sign
                    if thumb_tip[1] > thumb_mcp[1]:
                        if index_tip[1] > middle_tip[1] > ring_tip[1] > pinky_tip[1]:
                            if index_mcp[1] > middle_mcp[1] > ring_mcp[1] > pinky_mcp[1]:
                                state = states[2]
                                return state, middle_tip

                    # ## For 'CALL' sign
                    # if dist_thumb_to_pinky > 2*dist_middle_finger:
                    #     if pinky_tip[1]>pinky_mcp[1]:
                    #         state = states[3]
                    #         return state

                    ## For 'Perfect' sign
                    if middle_tip[1] < middle_mcp[1] and ring_tip[1] < ring_mcp[1] and pinky_tip[1] < pinky_mcp[1]:
                        if index_tip[1] > index_mcp[1]:
                            if (abs(index_tip[0] - thumb_tip[0]) < 5) and (abs(index_tip[1] - thumb_tip[1]) < 5):
                                state = states[4]
                                return state, middle_tip
                    return '', middle_tip

                # Identify sign
                hand_sign, text_origin = calculate_hand_sign(wrist, thumb_tip, index_tip, middle_tip, ring_tip,
                                                             pinky_tip, thumb_mcp, index_mcp, middle_mcp, ring_mcp,
                                                             pinky_mcp)
                # Visualize sign
                cv2.putText(image, str(hand_sign), (int(text_origin[0] * 640) + 40, int(text_origin[1] * 480) - 40),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            except:
                pass
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

            cv2.imshow('MediaPipe Holistic', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

if menu == 2:
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
                hand_sign, text_origin, count = calculate_exercise(pose, count)
                # Visualize sign
                cv2.putText(image, str(hand_sign), (int(text_origin[0] * 640) + 40, int(text_origin[1] * 480) - 40),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                # Render curl counter & Setup status box
                cv2.rectangle(image, (0, 0), (270, 73), (203, 61, 170), -1)

                # Display repetition data
                cv2.putText(image, 'Repetitions:', (15, 12),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(count), (10, 60),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Display position data
                cv2.putText(image, 'Position:', (120, 12),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, hand_sign, (100, 60),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass

            # Render detections & draw image
            # results.pose_landmarks   ---> shows the x,y,z & visibility of each landmark
            # mp_pose.POSE_CONNECTIONS ---> shows each landmark connection, i.e. NOSE, RIGHT_SHOULDER, etc.
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            cv2.imshow('MediaPipe Holistic', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
