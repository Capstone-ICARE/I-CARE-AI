import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import os

# 데이터 로드
with open('content/pose_data.pkl', 'rb') as f:
    landmarks_data, labels = pickle.load(f)

# 랜드마크 데이터와 라벨을 배열로 변환
X = np.array(landmarks_data)
y = np.array(labels)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train)


# SVM 분류기 학습
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# 모델 저장
with open('content/pose_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
# 모델 로드
with open('content/pose_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
# 모델이 잘 로드되었는지 확인
if model == loaded_model:
    print("모델이 성공적으로 저장되고 로드되었습니다.")
else:
    print("모델 저장 또는 로드에 문제가 발생했습니다.")

try:
    pickle.dumps(model)
    print("모델은 직렬화 가능합니다.")
except pickle.PicklingError as e:
    print(f"모델 직렬화 중 오류 발생: {e}")


# # 모델 저장
# try:
#     with open('content/pose_model.pkl', 'wb') as f:
#         pickle.dump(model, f)
#     print("모델이 성공적으로 저장되었습니다.")
# except Exception as e:
#     print(f"모델 저장 중 오류 발생: {e}")

# # 모델 로드
# try:
#     with open('content/pose_model.pkl', 'rb') as f:
#         loaded_model = pickle.load(f)
#     print("모델이 성공적으로 로드되었습니다.")
# except Exception as e:
#     print(f"모델 로드 중 오류 발생: {e}")

# # 모델이 잘 로드되었는지 확인
# try:
#     if model == loaded_model:
#         print("모델이 성공적으로 저장되고 로드되었습니다.")
#     else:
#         print("모델이 저장 또는 로드 과정에서 변경되었습니다.")
# except Exception as e:
#     print(f"모델 비교 중 오류 발생: {e}")

# 현재 작업 디렉토리 확인
print("현재 작업 디렉토리:", os.getcwd())

# 파일 경로 확인 및 권한 검사
file_path = 'pose_model.pkl'
print("쓰기 권한 확인:", os.access(file_path, os.W_OK))
print("읽기 권한 확인:", os.access(file_path, os.R_OK))


file_path = 'content/pose_model.pkl'

# 파일 크기 확인
if os.path.exists(file_path):
    print("파일 크기:", os.path.getsize(file_path))
else:
    print("파일이 존재하지 않습니다.")

# 모델이 잘 로드되었는지 확인하는 방법 개선
try:
    if isinstance(model, type(loaded_model)):
        print("모델 타입이 일치합니다.")
        # 모델의 주요 속성 비교 (예: 파라미터, 구조 등)
        if model.get_params() == loaded_model.get_params():
            print("모델의 주요 속성이 일치합니다.")
        else:
            print("모델의 주요 속성이 일치하지 않습니다.")
    else:
        print("로드된 모델의 타입이 다릅니다.")
except Exception as e:
    print(f"모델 비교 중 오류 발생: {e}")