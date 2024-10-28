import numpy as np

# 데이터 로드
data01 = np.load('./gesture_data/data_npz/data_gesture_all_handmade.npz', allow_pickle=True)
data02 = np.load('./gesture_data/data_npz/data_gesture_all_original.npz', allow_pickle=True)

# 데이터 추출
datas1 = data01['data']
labels1 = data01['labels']
datas2 = data02['data']
labels2 = data02['labels']

combined_datas = np.concatenate((datas1, datas2), axis=0)
combined_labels = np.concatenate((labels1, labels2), axis=0)

print(f"Data shape: {combined_datas.shape}")
print(f"Labels shape: {combined_labels.shape}")
# 통합된 데이터를 저장
np.savez_compressed('./gesture_data/data_npz/main/data_gesture_all_main.npz', data=combined_datas, labels=combined_labels)

'''
data01 = np.load('./gesture_data/data_npz/data_gesture_03_original.npz', allow_pickle=True)
data02 = np.load('./gesture_data/data_npz/data_gesture_03_handmade.npz', allow_pickle=True)
datas1 = data01['data']
labels1 = data01['labels']
datas2 = data02['data']
labels2 = data02['labels']
combined_datas = np.concatenate((datas1, datas2), axis=0)
combined_labels = np.concatenate((labels1, labels2), axis=0)
print(f"Data shape: {combined_datas.shape}")
print(f"Labels shape: {combined_labels.shape}")
# 통합된 데이터를 저장
np.savez_compressed('./gesture_data/data_npz/use/data_gesture_03_combined.npz', data=combined_datas, labels=combined_labels)
'''