import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import joblib

data = np.load('./gesture_data/data_npz/main/data_gesture_main_1.npz', allow_pickle=True)
X = data['data']
y = data['labels']

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))  # 입력 차원은 키포인트 수
model.add(Dropout(0.3))  # 드롭아웃 추가
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))  # 드롭아웃 추가
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(y)), activation='softmax'))

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # 문자열 라벨을 정수로 변환
y_categorical = to_categorical(y_encoded)  # 원-핫 인코딩으로 변환

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y_categorical, epochs=50, batch_size=32)

loss, accuracy = model.evaluate(X, y_categorical)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# 모델 저장
model.save('./gesture_data/data_npz/main/gesture_model_1.h5')
joblib.dump(label_encoder, './gesture_data/data_npz/main/label_encoder_1.pkl')