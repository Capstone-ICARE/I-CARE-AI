from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from video_load import VideoLoader
import cv2
import os
import random

class Video:
    def __init__(self, video_path):
        self.result = []
        loader = VideoLoader(video_path)
        video_model = load_model('./action_recognition/action_recognition_model_NEW.h5')
        video_keypoints = np.array(loader.video_keypoints)
        video_frames = loader.frames
        if video_frames is None:
            print(video_frames)
        if loader.video_keypoints and video_frames:
            predictions = video_model.predict(video_keypoints)
            labels = ['격려하기', '혼내기', 'Neutral']
            label_encoder = LabelEncoder()
            label_encoder.fit_transform(labels)
            predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
            print(f'Predicted Labels: {predicted_labels}')

            # 프레임을 파일로 저장
            directory = './action_recognition/images'
            for i, label in enumerate(predicted_labels):
                if (label != 'Neutral'):
                    frame_path = self.generate_unique_filename(directory)
                    print(f'Frame path: {frame_path}')
                    cv2.imwrite(frame_path, video_frames[i])
                    self.result.append({'filename': frame_path, 'label': str(label)})

    def generate_unique_filename(self, directory):
        while True:
            # 랜덤 숫자로 파일 이름 생성
            random_number = random.randint(100000, 999999)
            filename = f"frame_{random_number}.jpg"
            file_path = os.path.join(directory, filename)
            # 파일이 이미 존재하지 않으면 사용
            if not os.path.exists(file_path):
                return file_path