import cv2
import torch
import mediapipe as mp

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# MediaPipe BlazePose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5로 사람 감지
    results = model(frame)
    detections = results.pred[0]
    
    for det in detections:
        if det[5] == 0:  # person class
            x1, y1, x2, y2 = map(int, det[:4])
            person_img = frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # MediaPipe BlazePose로 포즈 추정
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            result = pose.process(person_rgb)
            
            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    cx, cy = int(lm.x * person_img.shape[1]), int(lm.y * person_img.shape[0])
                    cv2.circle(person_img, (cx, cy), 5, (0, 255, 0), -1)

            # 감지된 사람을 원래 프레임에 표시
            frame[y1:y2, x1:x2] = person_img

    # 결과 프레임을 화면에 표시
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
