import cv2
import mediapipe as mp
import numpy as np
import pickle
import torch

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 모델 로드
with open('content/pose_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)

# 실시간 포즈 예측 함수
def predict_pose(landmarks):
    flattened_landmarks = []
    for lm in landmarks:
        flattened_landmarks.extend([lm.x, lm.y, lm.z])
    return trained_model.predict([flattened_landmarks])[0]

# 비디오 파일 경로
video_path = 'content/input_data/hf_test.mov'

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # 포즈 추정
    results = pose.process(image)

    # YOLOv5로 사람 감지
    yolo_results = model(frame)
    detections = yolo_results.pred[0]  # 감지된 객체 리스트

    landmarks_list = []

    for det in detections:
        if det[5] == 0:  # person class
            x1, y1, x2, y2 = map(int, det[:4])  # bounding box 좌표 추출
            person_img = frame[y1:y2, x1:x2]
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            # MediaPipe BlazePose로 포즈 추정
            result = pose.process(person_rgb)

            # 랜드마크 추출
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                if len(landmarks) == 33:  # 33개의 랜드마크가 있는지 확인
                    landmarks_list.append(landmarks)
                    mp_drawing.draw_landmarks(person_img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # 포즈 랜드마크 그리기
                    for lm in landmarks:
                        cx, cy = int(lm.x * person_img.shape[1]), int(lm.y * person_img.shape[0])
                        cv2.circle(person_img, (cx, cy), 5, (0, 255, 0), -1)

            # 감지된 사람을 원래 프레임에 표시
            frame[y1:y2, x1:x2] = person_img

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if landmarks_list:
        # 모든 사람의 랜드마크를 하나의 리스트로 평탄화
        all_landmarks = []
        for landmarks in landmarks_list:
            for lm in landmarks:
                all_landmarks.extend([lm.x, lm.y, lm.z])
                
        if len(all_landmarks) == 99:
            # 예측 수행
            predicted_pose = predict_pose(all_landmarks)
            print(f"Predicted pose: {predicted_pose}")
            cv2.putText(frame, f"Pose: {predicted_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            print(f"Expected 99 landmarks, but got {len(all_landmarks)}")

    # 결과 프레임 표시
    cv2.imshow('Frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
