import mediapipe as mp
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import queue
import threading
import time

class VideoLoader:
    def __init__(self, video_model, yolo_model):
        self.yolo_model = yolo_model
        self.video_model = video_model
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.keypoint_mapping = {
            0: 'Nose', 11: 'LeftArm', 13: 'LeftForeArm', 15: 'LeftHand',
            12: 'RightArm', 14: 'RightForeArm', 16: 'RightHand',
            23: 'LeftUpLeg', 25: 'LeftLeg', 27: 'LeftFoot',
            24: 'RightUpLeg', 26: 'RightLeg', 28: 'RightFoot'
        }
        self.predicted_labels = []
        self.video_frames = []
        self.current_frame = None
        self.running = False
        self.thread = None
        self.thread2 = None
        self.end = False
        self.end2 = False
        self.frame_queue = queue.Queue()
        self.cap = None

    def start(self):
        self.predicted_labels = []
        self.video_frames = []
        self.current_frame = None
        self.running = True
        self.thread = None
        self.thread2 = None
        self.end = False
        self.end2 = False
        self.frame_queue = queue.Queue()
        self.cap = cv2.VideoCapture(0)
        self.thread = threading.Thread(target=self.run_temp)
        self.thread.start()
        self.thread2 = threading.Thread(target=self.run)
        self.thread2.start()

    def stop(self):
        self.running = False
        while not (self.end and self.end2):
            time.sleep(0.1)
        if self.thread:
            self.thread.join()
            self.thread = None
        if self.thread2:
            self.thread2.join()
            self.thread2 = None
        self.current_frame = None
        self.cap.release()

    def get_current_frame(self):
        return self.current_frame if self.current_frame is not None else None

    def get_current_label(self):
        return self.predicted_labels[-1] if self.predicted_labels else None

    def get_label_count(self):
        return len(self.predicted_labels) if self.predicted_labels else 0

    def run_temp(self):
        #fps = self.cap.get(cv2.CAP_PROP_FPS) # 시간 오래 걸림
        fps = 30.0
        interval = max(1, int(fps/5))
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.current_frame = frame
            if frame_count % interval == 0:
                self.frame_queue.put(frame)
            frame_count += 1
        self.cap.release()
        self.end = True

    def run(self):
        self.predicted_labels = []
        self.video_frames = []
        sequence_length = 30
        keypoints_sequence = []
        dev_frame_count = 0
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                frame_keypoints = self.process_frame(frame)
                if frame_keypoints:
                    keypoints_sequence.append(frame_keypoints)
                    temp_frame = frame
                    dev_frame_count += 1
                    if dev_frame_count % 30 == 15:
                        self.video_frames.append(frame)
                if len(keypoints_sequence) == sequence_length:
                    self.predicted_labels.append(self.predict_label(keypoints_sequence))
                    keypoints_sequence = []
                self.frame_queue.task_done()
        if len(keypoints_sequence) > 0:
            keypoints_sequence = self.pad_sequence(keypoints_sequence, sequence_length)
            self.predicted_labels.append(self.predict_label(keypoints_sequence))
            if dev_frame_count % 30 < 15:
                self.video_frames.append(temp_frame)
        self.end2 = True

    def predict_label(self, keypoints):
        keypoints = np.array([keypoints])
        predictions = self.video_model.predict(keypoints)
        labels = ['격려하기', '혼내기']
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(labels)
        predicted_label = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        return predicted_label[0]

    def pad_sequence(self, sequence, target_length):
        if len(sequence) >= target_length:
            return sequence
        padding = [sequence[-1]] * (target_length - len(sequence))
        return sequence + padding

    def map_keypoints(self, old_keypoints, keypoint_mapping):
        new_keypoints = {}
        if isinstance(old_keypoints, list):
            for idx, (x, y) in enumerate(old_keypoints):
                if idx in keypoint_mapping:
                    new_idx = keypoint_mapping[idx]
                    new_keypoints[new_idx] = [x, y]
        else:
            print("Unsupported keypoints format")
        return new_keypoints
            
    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (320, 240))
        results = self.yolo_model(frame_resized)
        detections = results.pred[0]

        frame_keypoints = []

        for det in detections:
            if det[5] == 0:  # person class
                x1, y1, x2, y2 = map(int, det[:4])

                h_original, w_original = frame.shape[:2]
                x1 = int(x1 * (w_original / 320))
                x2 = int(x2 * (w_original / 320))
                y1 = int(y1 * (h_original / 240))
                y2 = int(y2 * (h_original / 240))

                person_img = frame[y1:y2, x1:x2]
                person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

                result = self.pose.process(person_rgb)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
                    old_keypoints = [(lm.x, lm.y) for lm in landmarks]
                    new_keypoints = self.map_keypoints(old_keypoints, self.keypoint_mapping)
                    frame_keypoints.append(new_keypoints)

        if len(frame_keypoints) == 2:
            person1_keypoints = [x for keypoint in list(frame_keypoints[0].values()) for x in keypoint]
            person2_keypoints = [x for keypoint in list(frame_keypoints[1].values()) for x in keypoint]
            return person1_keypoints + person2_keypoints
        else:
            return None