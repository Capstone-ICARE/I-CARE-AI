import os
import json
import numpy as np
import cv2
import mediapipe as mp
import torch
from tensorflow.keras.models import load_model
import joblib
import threading
import time

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) 

# YOLOv5 모델 로드
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 

desired_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

my_w, my_h = 640, 640

def process_image(current_frame):
    image = current_frame
    original_h, original_w, _ = image.shape
    image_resized = cv2.resize(image, (my_w, my_h))
    detections = yolo_model(image_resized).pred[0]

    for det in detections:
        if det[5] == 0:
            x1, y1, x2, y2 = map(int, det[:4])

            original_x1 = int(((x1)/my_w) * original_w)
            original_y1 = int(((y1)/my_h) * original_h)
            original_x2 = int(((x2)/my_w) * original_w)
            original_y2 = int(((y2)/my_h) * original_h)
            cv2.rectangle(image, (original_x1, original_y1), (original_x2, original_y2), (0, 0, 255), 4)

            person_img = image_resized[y1:y2, x1:x2]
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(person_rgb)
            if pose_result.pose_landmarks:
                for idx in desired_landmarks:
                    landmark = pose_result.pose_landmarks.landmark[idx]
                    h, w, _ = person_img.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    original_x = int(((x1 + x)/my_w) * original_w)
                    original_y = int(((y1 + y)/my_h) * original_h)
                    cv2.circle(image, (original_x, original_y), 4, (0, 255, 0), -1)
    return image


def open_cam():
    cap = cv2.VideoCapture(1) # 0
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        new_frame = process_image(frame)
        cv2.imshow('Camera', new_frame)
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_cam()