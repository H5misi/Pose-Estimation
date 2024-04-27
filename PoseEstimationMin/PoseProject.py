import time
import cv2
import PoseModule

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

prev_frame_time = 0
new_frame_time = 0

detector = PoseModule.PoseDetector()

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    img = detector.find_pose(img)

    landmarks_list = detector.find_landmarks_position(img, draw=False)

    print(landmarks_list)

    # if len(landmarks_list) > 0:
    #     print(landmarks_list[3])
    #     cv2.circle(img, (landmarks_list[3][1], landmarks_list[3][2]), 10, (0, 255, 0), cv2.FILLED)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # print(f"FPS: {fps}")

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) == ord('q'):
        break
