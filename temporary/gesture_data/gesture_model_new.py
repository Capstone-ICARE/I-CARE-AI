import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Conv1D, Conv2D, Conv3D, Flatten, Dense, Dropout, concatenate, LSTM, Reshape
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import BatchNormalization
import joblib

data = np.load('./gesture_data/data_npz/main/data_gesture_new_main5.npz', allow_pickle=True)
X_keypoints = data['keypoints']
X_distances = data['distances']
y = data['labels']

'''
#(sample)
keypoints_input = Input(shape=(2, 5, 3, 2))
conv2 = Conv2D(64, kernel_size=(3, 3), activation='linear')(keypoints_input)
distances_input = Input(shape=(7,))
dense1 = Dense(64, activation='linear')(distances_input)
concat = concatenate([conv2, dense1])
output = Dense(len(set(y)), activation='softmax')(concat)
'''
'''
#(main1)
keypoints_input = Input(shape=(2, 5, 3, 2))
reshaped_keypoints = Reshape((2, 5 * 3 * 2))(keypoints_input)  # (2, 30)으로 Reshape
lstm = LSTM(64, activation='linear')(reshaped_keypoints)
distances_input = Input(shape=(7,))
dense1 = Dense(64, activation='linear')(distances_input)
concat = concatenate([lstm, dense1])
output = Dense(len(set(y)), activation='softmax')(concat)
'''
'''
#(main2)
keypoints_input = Input(shape=(2, 5, 3, 2))
conv3d = Conv3D(64, kernel_size=(2, 2, 2), activation='relu')(keypoints_input)  # 3D convolution 적용
distances_input = Input(shape=(7,))
dense1 = Dense(64, activation='relu')(distances_input)
flattened_conv = Flatten()(conv3d)  # Conv3D 결과를 Flatten으로 펼침
concat = concatenate([flattened_conv, dense1])
output = Dense(len(set(y)), activation='softmax')(concat)
'''
'''
#(main2:relu)
keypoints_input = Input(shape=(2, 5, 3, 2))
conv3d = Conv3D(64, kernel_size=(2, 2, 2), activation='relu')(keypoints_input)  # 3D convolution 적용
distances_input = Input(shape=(2, 5))
conv1d = Conv1D(64, kernel_size=2, activation='relu')(distances_input)
flattened_conv = Flatten()(conv3d)  # Conv3D 결과를 Flatten으로 펼침
flattened_distances = Flatten()(conv1d)
concat = concatenate([flattened_conv, flattened_distances])
output = Dense(len(set(y)), activation='softmax')(concat)
'''
'''
#(main3)
keypoints_input = Input(shape=(2, 5, 3, 2))
split_keypoints = [Dense(64, activation='linear')(keypoints_input[:, i]) for i in range(2)]  # 각 subarray를 처리
dense_keypoints = concatenate(split_keypoints)
flattened_keypoints = Flatten()(dense_keypoints)
distances_input = Input(shape=(7,))
dense1 = Dense(64, activation='linear')(distances_input)
concat = concatenate([flattened_keypoints, dense1])
output = Dense(len(set(y)), activation='softmax')(concat)
'''

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # 문자열 라벨을 정수로 변환
y_categorical = to_categorical(y_encoded)  # 원-핫 인코딩으로 변환

model = Model(inputs=[keypoints_input, distances_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_keypoints_train, X_keypoints_test, X_distances_train, X_distances_test, y_train, y_test = train_test_split(X_keypoints, X_distances, y_categorical, test_size=0.2, random_state=42)

history = model.fit(
  x=[X_keypoints_train, X_distances_train], 
  y=y_train, 
  epochs=50,
  batch_size=32,
  validation_split=0.2
)

loss, accuracy = model.evaluate([X_keypoints_test, X_distances_test], y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# 모델 저장
model.save('./gesture_data/data_npz/main/gesture_new_model6.keras')
#joblib.dump(label_encoder, './gesture_data/data_npz/main/new_label_encoder2.pkl')
