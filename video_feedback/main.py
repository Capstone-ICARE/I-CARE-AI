import cv2
import mediapipe as mp
import torch
import numpy as np
import pickle
import time

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 데이터 수집 함수
landmarks_data = []
labels = []

def add_data(landmarks, label):
    flattened_landmarks = []
    for lm in landmarks:
        flattened_landmarks.extend([lm.x, lm.y, lm.z])
    landmarks_data.append(flattened_landmarks)
    labels.append(label)

# 비디오 파일 경로 (colab에 업로드한 비디오 파일 경로)
#video_path = ['content/input_data/hf.mov', 'content/input_data/hf2.mov']
video_path = 'content/input_data/hf.mov'

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)
print("Press 's' to save data and 'q' to quit.")

frame_count = 0
show_every_n_frames = 3  # 매 5번째 프레임만 표시

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # YOLOv5로 사람 감지
    if frame_count % show_every_n_frames == 0:
      results = model(frame)
      detections = results.pred[0] # 감지된 객체 리스트
      people = [] # key point 저장할 리스트 초기화

      for det in detections:
          if det[5] == 0:  # person class
              x1, y1, x2, y2 = map(int, det[:4]) # bounding box 좌표 추출
              person_img = frame[y1:y2, x1:x2]
              person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

              # MediaPipe BlazePose로 포즈 추정
              result = pose.process(person_rgb)

              # 랜드마크 추출
              if result.pose_landmarks:
                  landmarks = result.pose_landmarks.landmark
                  #mp_drawing.draw_landmarks(person_img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                  # 랜드마크 데이터 저장
                  key = cv2.waitKey(10) & 0xFF
                  if key == ord('s'):
                      label = input("Enter label for this pose: ")
                      add_data(result.pose_landmarks.landmark, label)
                      print(f"Data saved for label: {label}")


                  # 포즈 랜드마크 그리기
                  for lm in landmarks:
                      cx, cy = int(lm.x * person_img.shape[1]), int(lm.y * person_img.shape[0])
                      cv2.circle(person_img, (cx, cy), 5, (0, 255, 0), -1)

              # 감지된 사람을 원래 프레임에 표시
              frame[y1:y2, x1:x2] = person_img

              # 바운딩 박스 그리기
              cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

          # 원래 프레임에 감지된 사람과 랜드마크가 그려진 이미지 추가
          frame[y1:y2, x1:x2] = person_img

      # 결과 프레임 표시
      cv2.imshow('Frame', frame)

      if cv2.waitKey(10) & 0xFF == ord('q'):
          break

      # 키 입력 처리 후 잠시 대기
      time.sleep(0.1)  # 0.1초 대기

cap.release()
cv2.destroyAllWindows()

# 데이터 저장
with open('content/pose_data.pkl', 'wb') as f:
    pickle.dump((landmarks_data, labels), f)