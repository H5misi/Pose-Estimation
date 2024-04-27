import time
import cv2
import mediapipe as mp
import torch


class PoseDetector:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # mediapipe variables to generate & draw pose on the image
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                      self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence,
                                      self.min_tracking_confidence)

    def find_pose(self, img, draw=True):
        # convert image (frame) from PGR to RGB, because mediapipe use RGB images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # send the image to the model
        self.results = self.pose.process(imgRGB)

        # to see landmarks info in the result -> results.pose_landmarks
        # print(self.results.pose_landmarks)

        # draw the pose landmarks on the image
        if self.results.pose_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img

    def find_landmarks_position(self, img, draw=True):
        landmarks_list = []
        for id, landmarks in enumerate(self.results.pose_landmarks.landmark):
            height, width, channel = img.shape
            # print(id, landmarks)
            # lm.x & lm.y -> is the ratio of the image
            # convert from image ratio to real x & y values
            x_value, y_value = int(landmarks.x * width), int(landmarks.y * height)

            landmarks_list.append([id, x_value, y_value])
            if draw:
                cv2.circle(img, (x_value, y_value), 5, (255, 0, 0), cv2.FILLED)

        return landmarks_list


def main():

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    prev_frame_time = 0
    new_frame_time = 0

    detector = PoseDetector()

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


if __name__ == '__main__':
    main()
