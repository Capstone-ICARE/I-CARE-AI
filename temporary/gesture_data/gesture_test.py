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

my_w, my_h = 640, 480

def process_image(current_frame):
    keypoints = []
    image = current_frame
    image_resized = cv2.resize(image, (my_w, my_h))

    detections = yolo_model(image_resized).pred[0]
    people_detected = 0
    people_with_pose = 0

    for det in detections:
        if det[5] == 0:
            people_detected += 1
            x1, y1, x2, y2 = map(int, det[:4])

            person_img = image_resized[y1:y2, x1:x2]
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            pose_result = pose.process(person_rgb)

            if pose_result.pose_landmarks:
                people_with_pose += 1

                for idx in desired_landmarks:
                    landmark = pose_result.pose_landmarks.landmark[idx]
                    h, w, _ = person_img.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    keypoints.append((x1 + x, y1 + y))

    if people_detected == 2 and people_with_pose == 2:
        return keypoints
    else:
        print(f"X, {people_detected}, {people_with_pose}")
        return []

def open_cam(running, shared_data):
    cap = cv2.VideoCapture(1) # 0
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    while running[0]:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        cv2.imshow('Camera', frame)
        shared_data['current_frame'] = frame
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running[0] = False
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    running = [True]
    shared_data = {'current_frame': None}
    loaded_model = load_model('./gesture_data/data_npz/main/gesture_model_1.h5')
    loaded_label_encoder = joblib.load('./gesture_data/data_npz/main/label_encoder_1.pkl')
    thread = threading.Thread(target=open_cam, args=(running, shared_data))
    thread.start()
    while running[0]:
        time.sleep(2)
        current_frame = shared_data['current_frame'] 
        if current_frame is not None:
            new_keypoints = process_image(current_frame)
            if new_keypoints == []:
                print('Predicted Label: Neutral')
            else:
                X_new = np.array([np.array(new_keypoints).flatten()])
                # 예측
                predictions = loaded_model.predict(X_new)
                predicted_label = loaded_label_encoder.inverse_transform([np.argmax(predictions)])
                print(f'Predicted Label: {predicted_label[0]}')


if __name__ == "__main__":
    main()