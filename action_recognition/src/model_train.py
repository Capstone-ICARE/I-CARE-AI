import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# 데이터 로드
data = np.load('./action_recognition/preprocessed_data.npz', allow_pickle=True)
X = data['data']  # 입력 데이터
y = data['labels']  # 라벨

X_list = X.tolist()
X_processed = [np.array(list(sample.values())) for sample in X_list]
X = np.array(X_processed)

# X 데이터 dtype 확인
print(f"X dtype: {X.dtype}")
print(f"y dtype: {y.dtype}")

# 레이블 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 데이터와 라벨 확인
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# 문자열 데이터 변환 (예: 숫자로 변환하거나 필요한 전처리 수행)
if X.dtype.kind in {'U', 'S'}:  # unicode 또는 string 타입인지 확인
    X = np.array([np.fromstring(x, dtype=np.float32, sep=',') for x in X])

# x, y 좌표 데이터가 2D 배열 형태라고 가정하고 이를 3D 텐서로 변환
# 예: X의 shape이 (num_samples, num_coords)
X = X.reshape((X.shape[0], -1, 2))  # (num_samples, num_coords // 2, 2)

# 데이터 정규화 (옵션)
X = (X - np.min(X)) / (np.max(X) - np.min(X))  # 0과 1 사이로 정규화

print(f"Labels: {y}")
print(f"Labels shape: {y.shape}")


# 데이터 분리: 훈련 데이터와 테스트 데이터
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True)

# 모델 정의
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # 클래스 수에 따라 출력층 조정
])

# 모델 컴파일
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# 모델 요약
model.summary()

# 모델 훈련
history = model.fit(X, y_encoded, epochs=10, batch_size=32, validation_split=0.2)
model_path = './action_recognition/action_recognition_model.h5'
model.save(model_path)
print(f"Model saved to {model_path}")

# 훈련 과정 시각화
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
