import time
import cv2
import mediapipe as mp
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mediapipe variables to generate & draw pose on the image
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    # convert image (frame) from PGR to RGB, because mediapipe use RGB images
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # send the image to the model
    results = pose.process(imgRGB)

    # to see landmarks info in the result -> results.pose_landmarks
    # print(results.pose_landmarks)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            print(id, lm)
            # lm.x & lm.y -> is the ratio of the image
            # convert from image ratio to real x & y values
            x_value, y_value = int(lm.x * width), int(lm.y * height)
            # cv2.circle(img, (x_value, y_value), 5, (255, 0, 0), cv2.FILLED)


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # print(f"FPS: {fps}")

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) == ord('q'):
        break
