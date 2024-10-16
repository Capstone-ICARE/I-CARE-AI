import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 데이터 로드
data = np.load('./action_recognition/preprocessed_data_test05.npz', allow_pickle=True)
X = data['data']  # 입력 데이터
y = data['labels']  # 라벨

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
    
    # 두 사람의 키포인트를 하나의 배열로 결합
    #combined_keypoints = np.array(person1_keypoints + person2_keypoints)
    #X_processed.append(combined_keypoints)
    #print(X_processed)

X = X_processed  
#X = np.array(X_processed)

# 레이블 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 데이터와 라벨 확인
#print(f"Features shape: {X.shape}")
#print(f"Labels shape: {y.shape}")

X = np.array(X)
y_encoded = np.array(y_encoded)

# NumPy 배열로 변환된 이후의 데이터 형태 확인
print("X shape after conversion:", X.shape)
print("y_encoded shape after conversion:", y_encoded.shape)

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
except ValueError as e:
    print(f"Error during train_test_split: {e}")


# 모델 정의
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(20, 52)),
    Dropout(0.3),  
    
    LSTM(128, return_sequences=True),
    Dropout(0.3),  

    LSTM(64, return_sequences=False),
    Dropout(0.3),  
    
    Dense(64, activation='relu'),
    Dropout(0.4), 
    Dense(32, activation='relu'),
    Dropout(0.4), 

    Dense(len(np.unique(y)), activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
# EarlyStopping 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss',  # 검증 손실을 모니터링
                               patience=5,          # 성능 향상이 없을 때 기다리는 Epoch 수
                               restore_best_weights=True)  # 가장 좋은 가중치로 복원


# 모델 요약
model.summary()
# X_train = np.array(X)
# y_train = np.array(y_encoded)
# 모델 훈련
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),  
                    epochs=30, 
                    batch_size=64,  
                    callbacks=[early_stopping]
                    )  


model_path = './action_recognition/action_recognition_model_test05.h5'
model.save(model_path)


# 훈련 과정 시각화
# import matplotlib.pyplot as plt

# plt.plot(history.history['accuracy'], label='Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()