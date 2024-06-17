import cv2
import numpy as np
import mediapipe as mp

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# YOLOv3 Tiny 초기화 "D:\ICARE\I-CARE-AI\video_feedback\weights\yolov3.weights"
net = cv2.dnn.readNet("video_feedback/weights/yolov3-tiny.weights", "video_feedback/config/yolov3-tiny.cfg")
with open("video_feedback\data\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

def detect_people(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # person class
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
    return boxes

# MediaPipe Pose 초기화
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_people(frame)
    print(f"Detected {len(detections)} people")
    for box in detections:
        x, y, w, h = box
        person_img = frame[y:y+h, x:x+w]
        if person_img.shape[0] == 0 or person_img.shape[1] == 0:
            continue

        # BGR 이미지를 RGB로 변환
        image_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # 포즈 추정 수행
        results = pose.process(image_rgb)

        # BGR 이미지를 다시 쓰기 가능 상태로 변경
        image_rgb.flags.writeable = True
        person_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 랜드마크 그리기
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
            person_img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)
        print("Pose detected")

        # 원본 프레임에 포즈 결과 그리기
        frame[y:y+h, x:x+w] = person_img

    # 결과 프레임 표시
    cv2.imshow('Multi-Person Pose Estimation with YOLOv3 Tiny and MediaPipe Pose', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()