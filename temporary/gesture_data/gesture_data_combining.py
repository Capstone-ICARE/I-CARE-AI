import numpy as np

# 데이터 로드
data01 = np.load('./gesture_data/data_npz/use/data_gesture_01_combined.npz', allow_pickle=True)
data02 = np.load('./gesture_data/data_npz/use/data_gesture_02_combined.npz', allow_pickle=True)
data03 = np.load('./gesture_data/data_npz/use/data_gesture_03_combined.npz', allow_pickle=True)
data04 = np.load('./gesture_data/data_npz/data_gesture_04_handmade.npz', allow_pickle=True)
data05 = np.load('./gesture_data/data_npz/data_gesture_05_handmade.npz', allow_pickle=True)
data06 = np.load('./gesture_data/data_npz/data_gesture_06_handmade.npz', allow_pickle=True)
data07 = np.load('./gesture_data/data_npz/data_gesture_07_handmade.npz', allow_pickle=True)
data08 = np.load('./gesture_data/data_npz/data_gesture_08_handmade.npz', allow_pickle=True)
data09 = np.load('./gesture_data/data_npz/data_gesture_neutral_handmade.npz', allow_pickle=True)

# 데이터 추출
datas1 = data01['data']
labels1 = data01['labels']
datas2 = data02['data']
labels2 = data02['labels']
datas3 = data03['data']
labels3 = data03['labels']
datas4 = data04['data']
labels4 = data04['labels']
datas5 = data05['data']
labels5 = data05['labels']
datas6 = data06['data']
labels6 = data06['labels']
datas7 = data07['data']
labels7 = data07['labels']
datas8 = data08['data']
labels8 = data08['labels']
datas9 = data09['data']
labels9 = data09['labels']

combined_datas = np.concatenate((datas1, datas2, datas3, datas4, datas5, datas6, datas7, datas8, datas9), axis=0)
combined_labels = np.concatenate((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9), axis=0)

print(f"Data shape: {combined_datas.shape}")
print(f"Labels shape: {combined_labels.shape}")
# 통합된 데이터를 저장
np.savez_compressed('./gesture_data/data_npz/main/data_gesture_main_1.npz', data=combined_datas, labels=combined_labels)