import tarfile
import zipfile
import os
import json
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import torch
import concurrent.futures
import time

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) 

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 


# 키포인트 매핑 정의
keypoint_mapping = {
    0: 'Nose', # 3: 0,   
    11: 'LeftArm', # 12: 11,  
    13: 'LeftForeArm', # 13: 13, 
    15: 'LeftHand', # 14: 15, 
    12: 'RightArm', # 17: 12, 
    14: 'RightForeArm', # 18: 14, 
    16: 'RightHand', # 19: 16,
    23: 'LeftUpLeg', # 26: 23, 
    25: 'LeftLeg', # 27: 25, 
    27: 'LeftFoot',# 28: 27, 
    24: 'RightUpLeg', # 21: 24, 
    26: 'RightLeg', # 22: 26, 
    28: 'RightFoot' # 23: 28  
}

# old_keypoints:mediapipe, new_keypoints에 새로운 인덱스(ex.Nose)로 key, value 형태로 저장
def map_keypoints(old_keypoints, keypoint_mapping):
    new_keypoints = {}

    if isinstance(old_keypoints, list):
        for idx, (x, y) in enumerate(old_keypoints):
            if idx in keypoint_mapping:  # MediaPipe 랜드마크 번호 확인
                new_idx = keypoint_mapping[idx]
                new_keypoints[new_idx] = [x, y]

    else:
        print("Unsupported keypoints format")
    return new_keypoints


# extracting and mapping keypoints 
def process_video(video_path, keypoint_mapping):
    cap = cv2.VideoCapture(video_path)
    #cap = cv2.VideoCapture(0)

    keypoints_list = []  # 각 프레임에서 추출된 딕(셔너리 형태로) 매핑된 키포인트 리스트로 저장
    frame_buffer = []
    frame_count = 0
    total_processed_frames = 0

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return keypoints_list
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # 비디오의 초당 프레임 수 (120)
    interval = int(fps)  # 1초에 하나의 프레임 처리

    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 매 'process_interval' 프레임마다 처리
        if frame_count % interval == 0:

            # YOLOv5로 사람 감지
            frame_resized = cv2.resize(frame, (640, 480))
            results = model(frame_resized)
            detections = results.pred[0]  # YOLOv5 결과 (bounding box 및 클래스 정보)

            # 탐지된 사람 수 카운트
            people_detected = 0
            people_with_pose = 0

            # 탐지된 각 사람에 대해 MediaPipe Pose 적용
            for det in detections:
                if int(det[5]) == 0:  # person class (det[5]가 class index로 감지)
                    people_detected += 1
                    print("Person detected!")

                    # Bounding box 좌표 추출
                    x1, y1, x2, y2 = map(int, det[:4])

                    # 프레임 원래 크기
                    h_original, w_original = frame.shape[:2]

                    # YOLO bounding box 좌표를 원래 프레임 크기로 변환
                    x1 = int(x1 * (w_original / 640))
                    x2 = int(x2 * (w_original / 640))
                    y1 = int(y1 * (h_original / 480))
                    y2 = int(y2 * (h_original / 480))

                    
                    person_img = frame[y1:y2, x1:x2]
                    person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

                    # MediaPipe BlazePose로 포즈 추정
                    result = pose.process(person_rgb)

                    # landmark 성공적으로 추정한 경우
                    if result.pose_landmarks:
                        people_with_pose += 1
                        print("Pose landmarks DETECTED!")

                        # 포즈 랜드마크 추출 및 키포인트 변환
                        #h, w, _ = person_rgb.shape
                        landmarks = result.pose_landmarks.landmark
                        old_keypoints = [(lm.x, lm.y) for lm in landmarks]
                        new_keypoints = map_keypoints(old_keypoints, keypoint_mapping)
                        keypoints_list.append(new_keypoints)

                        # new_keypoints를 시각화
                        for key, (x, y) in new_keypoints.items():
                            cx = int(x * person_img.shape[1])
                            cy = int(y * person_img.shape[0])
                            cv2.circle(person_img, (cx, cy), 5, (0, 255, 0), -1)
                        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        print("Pose landmarks NOT detected!")

                    total_processed_frames += 1  # 처리된 프레임 수 증가

                cv2.imshow("Frame with Detected People", frame)
                cv2.waitKey(1)
                    

            # 현재 프레임에서 탐지된 사람 수와 포즈가 추정된 사람 수 출력
            print(f"Frame {frame_count}: {people_detected} people detected, {people_with_pose} with pose landmarks")
        frame_count += 1

    # 현재까지 처리된 프레임 수 출력
    print(f"Processed {total_processed_frames} frames.")
    print(keypoints_list)

    cap.release()
    return keypoints_list


if __name__ == "__main__":    
    process_video('./action_recognition/data/videos/M099_F105_39_01-02.mp4', keypoint_mapping)
    # 사용 예시
    # video_path = './action_recognition/data/videos/M099_F105_39_01-02.mp4'
    # keypoints_list = process_video(video_path, keypoint_mapping)  # 이미 처리된 keypoints 리스트
    # #visualize_keypoints_on_video(video_path, keypoints_list)