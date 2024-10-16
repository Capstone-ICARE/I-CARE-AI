import os
import json
import numpy as np
import cv2
import mediapipe as mp
import torch

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) 

# YOLOv5 모델 로드
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 

desired_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

my_w, my_h = 640, 640

def get_augment_data(keypoints, w, h):
    transformations = [
        #(rotate_keypoints, {"angle": 30, "width": w, "height": h}),
        (scale_keypoints, {"scale_x": 1.2, "scale_y": 1.2}),
        (translate_keypoints, {"tx": 10, "ty": -10}),
        (flip_keypoints_horizontal, {"width": w})
    ]
    augmented_data = []
    augmented_data.append(keypoints)
    for transform, params in transformations:
        new_keypoints = []
        for k in keypoints:
            new_keypoints.append(transform(k, **params))
        augmented_data.append(new_keypoints)
    return augmented_data # 5개의 keypoints

def rotate_point(x, y, angle, cx, cy):
    radians = np.deg2rad(angle)
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)
    
    x -= cx
    y -= cy
    
    x_new = x * cos_theta - y * sin_theta + cx
    y_new = x * sin_theta + y * cos_theta + cy
    
    return (x_new, y_new)

def rotate_keypoints(keypoints, angle, width, height): # 회전(-180~180) : 15, 30, 45, -30, -45
    cx, cy = width // 2, height // 2
    return rotate_point(keypoints[0], keypoints[1], angle, cx, cy)

def scale_keypoints(keypoints, scale_x, scale_y): # 이미지 크기(0.5~2.0) : 0.8, 1.0(원본), 1.2, 1.5
    scaled_x = min(max(keypoints[0] * scale_x, 0), my_w)
    scaled_y = min(max(keypoints[1] * scale_y, 0), my_h)
    return (scaled_x, scaled_y)

def translate_keypoints(keypoints, tx, ty): # 이동(-20~20)
    trans_x = min(max(keypoints[0] + tx, 0), my_w)
    trans_y = min(max(keypoints[1] + ty, 0), my_h)
    return (trans_x, trans_y)

def flip_keypoints_horizontal(keypoints, width): # 좌우 반전
    return (width - keypoints[0], keypoints[1])

def process_image(image_path):
    keypoints = []
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (my_w, my_h))
    image_resized_original = image_resized.copy()

    detections = yolo_model(image_resized).pred[0]
    people_detected = 0
    people_with_pose = 0

    for det in detections:
        if det[5] == 0:
            people_detected += 1
            x1, y1, x2, y2 = map(int, det[:4])

            # 프레임 원래 크기
            #h_original, w_original = image.shape[:2]

            # YOLO bounding box 좌표를 원래 프레임 크기로 변환
            #x1 = int(x1 * (w_original / 640))
            #x2 = int(x2 * (w_original / 640))
            #y1 = int(y1 * (h_original / 480))
            #y2 = int(y2 * (h_original / 480))

            new_x1, new_y1, new_x2, new_y2 = max(0, x1 - 25), max(0, y1 - 15), min(my_w, x2 + 25), min(my_h, y2 + 15)
            #new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

            person_img = image_resized_original[new_y1:new_y2, new_x1:new_x2]
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            cv2.rectangle(image_resized, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), 2)
            #cv2.imshow('personrgb', person_rgb)
            pose_result = pose.process(person_rgb) 

            if pose_result.pose_landmarks:
                people_with_pose += 1
                for idx in desired_landmarks:
                    landmark = pose_result.pose_landmarks.landmark[idx]
                    w, h = new_x2 - new_x1, new_y2 - new_y1
                    #if landmark.x < 0 or landmark.y < 0 or landmark.x > 1 or landmark.y > 1:
                    #    print(f"X: {image_path}")
                    #    return []
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    keypoints.append((new_x1 + x, new_y1 + y))
                    cv2.circle(image_resized, (new_x1 + x, new_y1 + y), 4, (0, 255, 0) if people_with_pose == 1 else (255, 0, 0), -1)

    if people_detected == 2 and people_with_pose == 2:
        print(f'주의: {image_path}')
        cv2.imshow('Detected Pose', image_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return keypoints
    else:
        #print(f"X: {image_path}, {people_detected}, {people_with_pose}")
        return []

def process_image_list(directory, label):
    all_keypoints = []
    all_labels = []

    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')) :
            image_path = os.path.join(directory, filename)
            keypoints = process_image(image_path)

            if keypoints != []:
                #label = image_labels.get(filename, 'Neutral')
                augmented_keypoints = get_augment_data(keypoints, my_w, my_h)
                for kp in augmented_keypoints:
                    all_keypoints.append(kp) # [(x, y), (x, y), (x, y), (x, y), ...]
                    all_labels.append(label)

    return np.array([np.array(kp).flatten() for kp in all_keypoints]), np.array(all_labels)

if __name__ == "__main__":
    #directory = './images/gesture01'
    directory = './gesture_data/images/gesture04'
    label = '마주보고 손바닥 맞대고 발 뒤로 들기'
    keypoint_file_name = './gesture_data/data_npz/data_gesture_04_original.npz'
    #keypoint_file_name = './data_npz/data_gesture_01.npz'

    datas, labels = process_image_list(directory, label)
    print(f"Data shape: {datas.shape}")
    print(f"Labels shape: {labels.shape}")
    np.savez_compressed(keypoint_file_name, data=datas, labels=labels)