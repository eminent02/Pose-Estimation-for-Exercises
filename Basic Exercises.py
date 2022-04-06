import cv2
import numpy as np
import mediapipe as mp
from helper import *

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
counter = 0
# Pushup positions
up_pos = None
down_pos = None
pushup_pos = None
display_pos = None

# calculate angle between joints
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    angle = round(angle, 2)
    return angle


# def calculate_exercise(pose, counter):
#     # Calculate Angles
#     left_angle = calculate_angle(pose[1], pose[3], pose[5])
#     right_angle = calculate_angle(pose[2], pose[4], pose[6])
#
#     # Visualize the elbow as landmark
#     # Visualize elbow/landmark angle
#     # Angle will render right next to elbow --> multiplies the elbow position by webcam screen size
#     cv2.putText(image, str(left_angle), tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.putText(image, str(right_angle), tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#     # Push up counter logic
#     if left_angle > 160:
#         up_pos = "up"
#         display_pos = 'up'
#     if left_angle < 110 and up_pos == "up":
#         down_pos = "down"
#         display_pos = "down"
#     if left_angle > 160 and down_pos == "down":
#         pushup_pos = "up"
#         display_pos = "up"
#         counter += 1
#
#         # Reset positions after a push up is complete to stop multiple pushup registers
#         up_pos = None
#         down_pos = None
#         pushup_pos = None
#
#         # print(counter)

# For webcam input:
cap = cv2.VideoCapture(0)
result = cv2.VideoWriter('pushup.avi', cv2.VideoWriter_fourcc(*'MJPG'),60, (int(cap.get(3)), int(cap.get(4))))

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
            pose_landmarks = results.pose_landmarks.landmark

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
            # Calculate Left/Right Angles
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Visualize elbow/landmark angle
            cv2.putText(
                image, str(left_angle),
                # Angle will render right next to elbow --> multiplies the elbow position by webcam screen size
                tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.putText(
                image, str(right_angle),
                tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            )

            # Push up counter logic
            if left_angle > 160:
                up_pos = "up"
                display_pos = 'up'
            if left_angle < 110 and up_pos == "up":
                down_pos = "down"
                display_pos = "down"
            if left_angle > 160 and down_pos == "down":
                pushup_pos = "up"
                display_pos = "up"
                counter += 1

                # Reset positions after a push up is complete to stop multiple pushup registers
                up_pos = None
                down_pos = None
                pushup_pos = None

                # print(counter)

        except:
            pass

            # Render curl counter & Setup status box
        cv2.rectangle(image, (0, 0), (270, 73), (203, 61, 170), -1)

        # Display repetition data
        cv2.putText(image, 'Repetitions:', (15, 12),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60),
                   cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Display position data
        cv2.putText(image, 'Position:', (120, 12),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, display_pos, (100, 60),
                   cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections & draw image
        # results.pose_landmarks   ---> shows the x,y,z & visibility of each landmark
        # mp_pose.POSE_CONNECTIONS ---> shows each landmark connection, i.e. NOSE, RIGHT_SHOULDER, etc.
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # Display opencv to the monitor
        result.write(image)
        cv2.imshow('Mediapipe Feed', image)


        # Exit program logic
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()


