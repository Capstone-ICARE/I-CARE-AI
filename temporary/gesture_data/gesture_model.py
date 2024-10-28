import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, Dense, Dropout, concatenate, InputLayer, Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import joblib

data = np.load('./gesture_data/data_npz/main/data_gesture_new_main12.npz', allow_pickle=True)
X = data['keypoints']
#X_angle = data['angles']
y = data['labels']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # 문자열 라벨을 정수로 변환
y_categorical = to_categorical(y_encoded)  # 원-핫 인코딩으로 변환

'''
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))  # 입력 차원은 키포인트 수
model.add(Dropout(0.3))  # 드롭아웃 추가
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))  # 드롭아웃 추가
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(y)), activation='softmax'))
'''
'''
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(52,)))  # (num_keypoints, 2) 입력
model.add(BatchNormalization())  # 배치 정규화
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())  # 배치 정규화
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())  # 배치 정규화
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())  # 배치 정규화
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())  # 배치 정규화
model.add(Dropout(0.3))

model.add(Flatten())  # 1D 데이터를 Flatten
model.add(Dense(128, activation='relu'))
model.add(Dense(len(set(y)), activation='softmax'))
'''

lambda_value = 0.025

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(52,), kernel_regularizer=l2(lambda_value)))
model.add(BatchNormalization())  # 배치 정규화
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())  # 배치 정규화
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu', kernel_regularizer=l2(lambda_value)))
model.add(BatchNormalization())  # 배치 정규화
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())  # 배치 정규화
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())  # 배치 정규화
model.add(Dropout(0.3))

model.add(Flatten())  # 1D 데이터를 Flatten
model.add(Dense(128, activation='relu'))
model.add(Dense(len(set(y)), activation='softmax'))

'''
lambda_value = 0.025

X_input = Input(shape=(52,))
dense1 = Dense(64, activation='relu', kernel_regularizer=l2(lambda_value))(X_input)
dense1 = BatchNormalization()(dense1)
dense1 = Dropout(0.3)(dense1)

X_angle_input = Input(shape=(4,))
dense2 = Dense(64, activation='relu')(X_angle_input)
dense2 = BatchNormalization()(dense2)
dense2 = Dropout(0.3)(dense2)

concat = concatenate([dense1, dense2])
concat = Flatten()(concat)

concat = Dense(128, activation='relu', kernel_regularizer=l2(lambda_value))(concat)
concat = BatchNormalization()(concat)
concat = Dropout(0.3)(concat)

concat = Dense(256, activation='relu')(concat)
concat = BatchNormalization()(concat)
concat = Dropout(0.3)(concat)

concat = Dense(256, activation='relu')(concat)
concat = BatchNormalization()(concat)
concat = Dropout(0.3)(concat)

concat = Dense(128, activation='relu')(concat)

# 최종 출력 layer
output = Dense(len(set(y)), activation='softmax')(concat)

# 모델 정의
model = Model(inputs=[X_input, X_angle_input], outputs=output)
'''
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#model.fit(X, y_categorical, epochs=50, batch_size=32)
#model.fit(X_train, y_train, epochs=50, validation_split=0.2)

#X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

#mean = np.mean(X, axis=0)
#std = np.std(X, axis=0)

#np.savez('./gesture_data/data_npz/normalization_params.npz', mean=mean, std=std)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, 
    y_train,
    epochs=32,
    batch_size=64,
    validation_split=0.3,
    callbacks=[early_stopping]
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# 모델 저장
model.save('./gesture_data/data_npz/main/gesture_new_model19.keras')
joblib.dump(label_encoder, './gesture_data/data_npz/main/new_label_encoder_8.pkl')


'''
model = load_model('./gesture_data/data_npz/main/gesture_model_2.h5')
label_encoder = joblib.load('./gesture_data/data_npz/main/label_encoder_2.pkl')

predictions = model.predict(X_test)
threshold = 0.9
# 최대 확률 및 예측 클래스 결정
predicted_classes = []
max_probs = []
for pred in predictions:
    max_prob = np.max(pred)
    max_probs.append(max_prob)
    if max_prob < threshold:
        predicted_classes.append(8)  # neutral 클래스 인덱스 (예: 8)
    else:
        predicted_classes.append(np.argmax(pred))

y_test_classes = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_classes, predicted_classes)
print(f'Accuracy: {accuracy * 100:.2f}%')

predicted_labels = []
for cls in predicted_classes:
    if cls == 8:
        predicted_labels.append("Neutral")
    else:
        predicted_labels.append(label_encoder.inverse_transform([cls])[0])

for i, label in enumerate(predicted_labels):
    real_label = label_encoder.inverse_transform([y_test_classes[i]])[0]
    print(f'Sample {i}: {round(max_probs[i], 5)}: {label}, {real_label}')
'''