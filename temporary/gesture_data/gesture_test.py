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
    keypoints = []
    image = current_frame
    original_h, original_w, _ = image.shape
    image_resized = cv2.resize(image, (my_w, my_h))

    detections = yolo_model(image_resized).pred[0]
    people_detected = 0
    people_with_pose = 0

    for det in detections:
        if det[5] == 0:
            people_detected += 1
            x1, y1, x2, y2 = map(int, det[:4])

            original_x1 = int(((x1)/my_w) * original_w)
            original_y1 = int(((y1)/my_h) * original_h)
            original_x2 = int(((x2)/my_w) * original_w)
            original_y2 = int(((y2)/my_h) * original_h)
            cv2.rectangle(image, (original_x1, original_y1), (original_x2, original_y2), (0, 0, 255), 2)

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
                    
                    original_x = int(((x1 + x)/my_w) * original_w)
                    original_y = int(((y1 + y)/my_h) * original_h)
                    cv2.circle(image, (original_x, original_y), 4, (0, 255, 0) if people_with_pose == 1 else (255, 0, 0), -1)

    if people_detected == 2 and people_with_pose == 2:
        cv2.imshow('Test', image)
        cv2.waitKey(1)
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
        shared_data['current_frame'] = frame
        cv2.imshow('Camera', frame)
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running[0] = False
            break
    cap.release()
    cv2.destroyAllWindows()

def load_normalization_params():
    # 평균과 표준편차를 저장해놓은 파일이 있으면 로드
    return np.load('./gesture_data/data_npz/normalization_params.npz')

def main():
    running = [True]
    shared_data = {'current_frame': None}
    loaded_model = load_model('./gesture_data/data_npz/main/gesture_new_model1.keras')
    loaded_label_encoder = joblib.load('./gesture_data/data_npz/main/new_label_encoder1.pkl')
    
    threshold = 0.9
    normalization_params = load_normalization_params()
    mean = normalization_params['mean']
    std = normalization_params['std']

    thread = threading.Thread(target=open_cam, args=(running, shared_data))
    thread.start()
    while running[0]:
        time.sleep(2)
        current_frame = shared_data['current_frame'] 
        if current_frame is not None:
            new_keypoints = process_image(current_frame)
            if len(new_keypoints) == 0:
                print('Predicted Label: Neutral')
            else:
                X_new = np.array([new_keypoints])
                X_new_normalized = (X_new - mean) / std
                # 예측
                predictions = loaded_model.predict(X_new_normalized)
                max_prob = np.max(predictions)

                if max_prob < threshold:
                    predicted_label = 'Neutral'
                else:
                    predicted_label = loaded_label_encoder.inverse_transform([np.argmax(predictions)])[0]

                predicted_label = loaded_label_encoder.inverse_transform([np.argmax(predictions)])
                print(f'Predicted Label: {predicted_label} (Confidence: {round(max_prob, 3)})')

if __name__ == "__main__":
    main()

'''
loaded_model = load_model('./gesture_data/data_npz/main/gesture_model_2.h5')
loaded_label_encoder = joblib.load('./gesture_data/data_npz/main/label_encoder_2.pkl')
data = np.load('./gesture_data/data_npz/data_gesture_neutral_handmade.npz', allow_pickle=True)
X = data['data']
X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

#X_new = np.expand_dims(X_normalized, axis=0)
predictions = loaded_model.predict(X_normalized)


threshold = 0.9
# 최대 확률 및 예측 클래스 결정
predicted_classes = []
max_probs = []
for pred in predictions:
    max_prob = np.max(pred)
    max_probs.append(max_prob)
    if max_prob < threshold:
        predicted_classes.append(8)  # neutral 클래스 인덱스 (예: 8)
    else:
        predicted_classes.append(np.argmax(pred))

predicted_labels = []
for cls in predicted_classes:
    if cls == 8:
        predicted_labels.append("Neutral")
    else:
        predicted_labels.append(loaded_label_encoder.inverse_transform([cls])[0])

for i, label in enumerate(predicted_labels):
    print(f'Sample {i}: {max_probs[i]}: {label}')

#predicted_labels = loaded_label_encoder.inverse_transform(np.argmax(predictions, axis=1))
'''