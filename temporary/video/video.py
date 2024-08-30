from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from video.video_load import VideoLoader

class Video:
    def __init__(self, video_path):
        loader = VideoLoader(video_path)
        video_model = load_model('./video/datamodel/action_recognition_model.h5')
        video_keypoints = np.array(loader.video_keypoints)
        predictions = video_model.predict(video_keypoints)
        labels = ['격려하기', '혼내기']
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(labels)
        predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        print(f'Predicted Labels: {predicted_labels}')