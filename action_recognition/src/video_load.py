import torch
import mediapipe as mp
import cv2

class VideoLoader:
    def __init__(self, video_path):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.video_keypoints = []
        self.frames = []
        self.keypoint_mapping = {
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
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps/5))
        sequence_length = 30
        keypoints_sequence = []
        frame_count = 0
        dev_frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 매 'process_interval' 프레임마다 처리
            if frame_count % interval == 0:
                frame_keypoints = self.process_frame(frame)
                if frame_keypoints:
                    keypoints_sequence.append(frame_keypoints)
                    temp_frame = frame
                    dev_frame_count +=1
                    if dev_frame_count %30 ==15:
                        self.frames.append(frame)    
            frame_count += 1
            #self.frames.append(frame)
            if len(keypoints_sequence) == sequence_length:
                self.video_keypoints.append(keypoints_sequence)
                keypoints_sequence = []
        cap.release()

        if len(keypoints_sequence) > 0:
            keypoints_sequence = self.pad_sequence(keypoints_sequence, sequence_length)
            self.video_keypoints.append(keypoints_sequence)

            if dev_frame_count % 30 <15:
                self.frames.append(temp_frame)
    
    def pad_sequence(self, sequence, target_length):
        if len(sequence) >= target_length:
            return sequence
        padding = [sequence[-1]] * (target_length - len(sequence))
        return sequence + padding

    def map_keypoints(self, old_keypoints, keypoint_mapping):
        new_keypoints = {}
        if isinstance(old_keypoints, list):
            for idx, (x, y) in enumerate(old_keypoints):
                if idx in keypoint_mapping:  # MediaPipe 랜드마크 번호 확인
                    new_idx = keypoint_mapping[idx]
                    new_keypoints[new_idx] = [x, y]
        else:
            print("Unsupported keypoints format")
        return new_keypoints
            
    def process_frame(self, frame):
        # YOLOv5로 사람 감지
        frame_resized = cv2.resize(frame, (320, 240))
        results = self.yolo_model(frame_resized)
        detections = results.pred[0]

        people_detected = 0
        people_with_pose = 0
        frame_keypoints=[] # 각 프레임에서 사람 2명 키포인트 저장

        # 탐지된 각 사람에 대해 MediaPipe Pose 적용
        for det in detections:
            if det[5] == 0:  # person class
                people_detected += 1
                x1, y1, x2, y2 = map(int, det[:4])

                # 프레임 원래 크기
                h_original, w_original = frame.shape[:2]

                # YOLO bounding box 좌표를 원래 프레임 크기로 변환
                x1 = int(x1 * (w_original / 320))
                x2 = int(x2 * (w_original / 320))
                y1 = int(y1 * (h_original / 240))
                y2 = int(y2 * (h_original / 240))

                person_img = frame[y1:y2, x1:x2]
                person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

                # MediaPipe BlazePose로 포즈 추정
                result = self.pose.process(person_rgb)

                # landmark 성공적으로 추정한 경우
                if result.pose_landmarks:
                    people_with_pose += 1

                    landmarks = result.pose_landmarks.landmark
                    old_keypoints = [(lm.x, lm.y) for lm in landmarks]
                    new_keypoints = self.map_keypoints(old_keypoints, self.keypoint_mapping)
                    frame_keypoints.append(new_keypoints)

        # Check if we have keypoints for two people
        if len(frame_keypoints) == 2:
            person1_keypoints = [x for keypoint in list(frame_keypoints[0].values()) for x in keypoint]  # 첫 번째 사람의 키포인트
            person2_keypoints = [x for keypoint in list(frame_keypoints[1].values()) for x in keypoint]  # 두 번째 사람의 키포인트
            return person1_keypoints + person2_keypoints
        else:
            return None  