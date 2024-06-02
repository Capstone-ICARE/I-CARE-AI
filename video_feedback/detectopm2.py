import cv2
import torch
import mediapipe as mp
import numpy as np

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# MediaPipe BlazePose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 손목 사이 거리 계산 함수
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5로 사람 감지
    results = model(frame)
    detections = results.pred[0]
    people = []
    
    for det in detections:
        if det[5] == 0:  # person class
            x1, y1, x2, y2 = map(int, det[:4])
            person_img = frame[y1:y2, x1:x2]            
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            # MediaPipe BlazePose로 포즈 추정
            result = pose.process(person_rgb)
            
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * (x2 - x1), 
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * (y2 - y1)]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * (x2 - x1), 
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * (y2 - y1)]
                
                people.append((left_wrist, right_wrist))

                # 포즈 랜드마크 그리기
                for lm in landmarks:
                    cx, cy = int(lm.x * person_img.shape[1]), int(lm.y * person_img.shape[0])
                    cv2.circle(person_img, (cx, cy), 5, (0, 255, 0), -1)

            # 감지된 사람을 원래 프레임에 표시
            frame[y1:y2, x1:x2] = person_img

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 하이파이브 동작 인식
    threshold = 50  # 픽셀 단위 거리 임계값
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            dist_left = calculate_distance(people[i][0], people[j][0])
            dist_right = calculate_distance(people[i][1], people[j][1])
            if dist_left < threshold or dist_right < threshold:
                cv2.putText(frame, "High Five Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


    # 결과 프레임을 화면에 표시
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
