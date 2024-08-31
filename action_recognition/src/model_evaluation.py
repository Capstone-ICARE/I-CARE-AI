import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 모델 경로
MODEL_PATH = './action_recognition/action_recognition_model.h5'

# 데이터 로드
data = np.load('./action_recognition/preprocessed_data.npz', allow_pickle=True)
X = data['data']
y = data['labels']

X_list = X.tolist()

X_processed = []
for sample in X_list:
    X_sample = []
    for frame in sample:
        #print("Sample structure:", sample)
        person1_keypoints = [x for keypoint in list(frame[0].values()) for x in keypoint]  # 첫 번째 사람의 키포인트
        person2_keypoints = [x for keypoint in list(frame[1].values()) for x in keypoint]  # 두 번째 사람의 키포인트
        combined_keypoints = person1_keypoints + person2_keypoints
        X_sample.append(combined_keypoints)
    X_processed.append(X_sample)

X = X_processed  

# 레이블 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X = np.array(X)
y_encoded = np.array(y_encoded)

# NumPy 배열로 변환된 이후의 데이터 형태 확인
print("X shape after conversion:", X.shape)
print("y_encoded shape after conversion:", y_encoded.shape)

# 데이터 정규화 (옵션)
X = (X - np.min(X)) / (np.max(X) - np.min(X))  # 0과 1 사이로 정규화

# 모델 로드
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# 모델 평가
loss, accuracy = model.evaluate(X, y_encoded)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# 예측값 계산
y_pred = model.predict(X)
y_pred_classes = (y_pred > 0.5).astype("int32").flatten()

# 실제 값과 예측 값 비교
plt.figure(figsize=(12, 6))
plt.plot(y_encoded, label='Actual', color='b')
plt.plot(y_pred_classes, label='Predicted', color='r', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()