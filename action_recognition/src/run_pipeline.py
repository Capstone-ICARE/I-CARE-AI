import subprocess

# 데이터 전처리
subprocess.run(['python', './action_recognition/src/data_preprocessing.py'])

# 모델 학습
subprocess.run(['python', './action_recognition/src/model_training.py'])

# 모델 평가
subprocess.run(['python', './action_recognition/src/model_evaluation.py'])
