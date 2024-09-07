from video.video_load import VideoLoader
import cv2
import os
import random

class Video:
    def __init__(self, video_model, yolo_model):
        self.result = []
        self.running = False
        self.loader = None
        self.video_model = video_model
        self.loader = VideoLoader(video_model, yolo_model)

    def start_video(self):
        self.loader.start()
        self.result = []
        self.running = True

    def stop_video(self):
        if self.loader:
            self.running = False
            self.loader.stop()
            self.process_results()
        return self.result
    
    def get_current_frame(self):
        return self.loader.get_current_frame() if self.loader else None

    def get_current_label(self):
        return self.loader.get_current_label() if self.loader else None

    def get_label_count(self):
        return self.loader.get_label_count() if self.loader else 0

    def process_results(self):
        predicted_labels = self.loader.predicted_labels
        video_frames = self.loader.video_frames
        if predicted_labels and video_frames:
            print(f'Predicted Labels: {predicted_labels}')
            directory = './video/datamodel/images'
            #directory = './images/video'
            temp_label = ''
            for i, label in enumerate(predicted_labels):
                if temp_label != label:
                    if label in ['격려하기', '혼내기']:
                        frame_path = self.generate_unique_filename(directory)
                        print(f'Frame path: {frame_path}')
                        cv2.imwrite(frame_path, video_frames[i])
                        self.result.append({'fileName': frame_path, 'label': str(label)})
                        temp_label = label
        else:
            print('Video keypoints or Video frames is None')

    def generate_unique_filename(self, directory):
        while True:
            random_number = random.randint(100000, 999999)
            filename = f"frame_{random_number}.jpg"
            file_path = os.path.join(directory, filename)
            if not os.path.exists(file_path):
                return file_path