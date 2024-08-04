import tarfile
import zipfile
import os
import json
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 원본 데이터를 저장할 디렉토리 경로
RAW_DIR = os.path.normpath('./action_recognition/data/raw')
TAR_DIR = os.path.join(RAW_DIR, 'tar_files')
EXTRACTED_DIR = os.path.join(RAW_DIR, 'extracted')

# 디렉토리 생성 함수
def create_directories():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(EXTRACTED_DIR, exist_ok=True)

# 경로에서 유효하지 않은 문자 제거 함수
def clean_path(path):
    return path.replace('\\', '/').replace('\x01', '_').replace('\x02', '_')

# tar 파일 추출 함수
def extract_tar(tar_file_path, extract_dir):
    with tarfile.open(tar_file_path, 'r') as tar:
        for member in tar.getmembers():
            member.name = clean_path(member.name)  # 경로 정리
            try:
                tar.extract(member, path=os.path.normpath(extract_dir))
            except tarfile.TarError:
                print(f"Skipping corrupted file: {member.name}")
    print(f"Extracted tar file to {extract_dir}")

# zip 파일 추출 함수
def extract_zip(zip_file_path, extract_dir):
    print(f"Extracting zip file from {zip_file_path} to {extract_dir}...")
    os.makedirs(os.path.normpath(extract_dir), exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            cleaned_member = clean_path(member)  # 경로 정리
            try:
                zip_ref.extract(member, extract_dir)
                os.rename(os.path.join(extract_dir, member), os.path.join(extract_dir, cleaned_member))
            except Exception as e:
                print(f"Error extracting {member}: {e}")
    print(f"Extracted zip file to {extract_dir}")

# 키포인트 매핑 정의

keypoint_mapping = {
    0: 'Nose', # 3: 0,   
    11: 'LeftArm', # 12: 11,  
    13: 'LeftForeArm', # 13: 13, 
    15: 'LeftHand', # 14: 15, 
    12: 'RightArm', # 17: 12, 
    14: 'RightForeArm', # 18: 14, 
    16: 'RightHand', # 19: 16,
    23: 'LeftUpLeg', # 26: 23, 
    25: 'LeftLeg', # 27: 25, 
    27: 'LeftFoot',# 28: 27, 
    24: 'RightUpLeg', # 21: 24, 
    26: 'RightLeg', # 22: 26, 
    28: 'RightFoot' # 23: 28  
}
# keypoint_mapping = {
#     'Nose': 0, # 3: 0,   
#     'LeftArm': 11, # 12: 11,  
#     'LeftForeArm': 13, # 13: 13, 
#     'LeftHand': 15, # 14: 15, 
#     'RightArm': 12, # 17: 12, 
#     'RightForeArm': 14, # 18: 14, 
#     'RightHand': 16, # 19: 16,
#     'LeftUpLeg': 23, # 26: 23, 
#     'LeftLeg': 25, # 27: 25, 
#     'LeftFoot': 27,# 28: 27, 
#     'RightUpLeg': 24, # 21: 24, 
#     'RightLeg': 26, # 22: 26, 
#     'RightFoot': 28 # 23: 28  
# }

# keypoint_mapping = {
#     'Nose': 'nose', # 3: 0,   
#     'LeftArm': 'left_shoulder', # 12: 11,  
#     'LeftForeArm': 'left_elbow', # 13: 13, 
#     'LeftHand': 'left_wrist', # 14: 15, 
#     'RightArm': 'right_shoulder', # 17: 12, 
#     'RightForeArm': 'right_elbow', # 18: 14, 
#     'RightHand': 'right_wrist', # 19: 16,
#     'LeftUpLeg': 'left_hip', # 26: 23, 
#     'LeftLeg': 'left_knee', # 27: 25, 
#     'LeftFoot': 'left_ankle' ,# 28: 27, 
#     'RightUpLeg': 'right_hip', # 21: 24, 
#     'RightLeg': 'right_knee', # 22: 26, 
#     'RightFoot': 'right_ankle' # 23: 28  
# }

# def map_keypoints(old_keypoints, keypoint_mapping, num_keypoints=33):
#     new_keypoints = np.full((num_keypoints, 2), np.nan)  # NaN으로 초기화
#     if isinstance(old_keypoints, list):
#         for idx, (x, y) in enumerate(old_keypoints):
#             if idx in keypoint_mapping.values():
#                 new_keypoints[idx] = [x, y]
#     elif isinstance(old_keypoints, dict):
#         for old_key, coords in old_keypoints.items():
#             if old_key in keypoint_mapping:
#                 new_idx = keypoint_mapping[old_key]
#                 new_keypoints[new_idx] = [coords['x'], coords['y']]
#     else:
#         print("Unsupported keypoints format")
#     return new_keypoints

def map_keypoints(old_keypoints, keypoint_mapping, num_keypoints=33):
    #new_keypoints = np.full((num_keypoints, 2), np.nan)  # NaN으로 초기화
    new_keypoints = {}

    if isinstance(old_keypoints, list):
        for idx, (x, y) in enumerate(old_keypoints):
            if idx in keypoint_mapping:  # MediaPipe 랜드마크 번호 확인
                new_idx = keypoint_mapping[idx]
                new_keypoints[new_idx] = [x, y]

    else:
        print("Unsupported keypoints format")
    return new_keypoints


# extracting and mapping keypoints 
def process_video(video_path, keypoint_mapping):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = [] # 각 프레임에서 추출된 매핑된 키포인트 저장

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe를 사용하여 키포인트 추출
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            old_keypoints = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
            new_keypoints = map_keypoints(old_keypoints, keypoint_mapping)
            keypoints_list.append(new_keypoints)

    cap.release()
    return keypoints_list


video_labels = {
    'M099_F105_46_01-02.mp4': '위로하기',
    'M099_F105_46_01-04.mp4': '위로하기',
    'M099_F105_46_01-05.mp4': '위로하기',
    'M099_F105_46_01-06.mp4': '위로하기',
    'M099_F105_46_01-07.mp4': '위로하기',
    '(1)M099_F105_39_04-01.mp4': '혼내기',
    '(1)M099_F105_39_04-03.mp4': '혼내기',
    '(1)M099_F105_39_04-05.mp4': '혼내기',
    '(1)M099_F105_39_04-06.mp4': '혼내기',
    '(1)M099_F105_39_04-07.mp4': '혼내기',
    '(1)M099_F105_39_04-08.mp4': '혼내기',
}

    # 'M099_F105_37_01-02.mp4': '때리고 맞기',
    # 'M099_F105_37_01-03.mp4': '때리고 맞기',
    # 'M099_F105_37_01-04.mp4': '때리고 맞기',
    # 'M099_F105_37_01-05.mp4': '때리고 맞기',
    # 'M099_F105_37_01-06.mp4': '때리고 맞기',
    # 'M099_F105_37_01-07.mp4': '때리고 맞기',
    # 'M099_F105_37_01-08.mp4': '때리고 맞기',

def process_videos(video_dir, keypoint_mapping, video_labels):
    all_keypoints = []
    all_labels = []

    # 비디오 파일 목록
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"Processing video: {video_path}")

        # 비디오에서 키포인트 추출
        keypoints_list = process_video(video_path, keypoint_mapping)

        # 라벨 추출 (파일 이름을 라벨로 사용하거나 다른 방법으로 라벨을 정의할 수 있습니다.)
        label = video_labels.get(video_file, 'unknown')  # video_labels 딕셔너리에서 라벨을 가져옴

        # 키포인트와 라벨 저장
        for keypoints in keypoints_list:
            all_keypoints.append(keypoints)
            all_labels.append(label)

    return np.array(all_keypoints), np.array(all_labels)



def load_and_preprocess_data(json_dir, keypoint_mapping, target_files):
    data = []
    labels = []
    for file in target_files:
        file_path = os.path.join(json_dir, file)
        if os.path.exists(file_path):
            print(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    sub_category = json_data['metaData']['sub category']  # sub category 추출
                    for annotation in json_data.get('annotation', []):
                        if annotation.get('pose', {}).get('type') == 'pose':
                            old_keypoints = annotation['pose']['location']

                            # old_keypoints가 리스트 형태라면 이를 확인하고 처리
                            if isinstance(old_keypoints, list):
                                old_keypoints_list = [(point['x'], point['y']) for point in old_keypoints]
                            elif isinstance(old_keypoints, dict):
                                old_keypoints_list = [(point['x'], point['y']) for point in old_keypoints.values()]
                            else:
                                print(f"Unexpected format of keypoints in file {file_path}")
                                continue

                            new_keypoints = map_keypoints(old_keypoints_list, keypoint_mapping)
                            # 딕셔너리 값 (좌표 쌍)만 추출하여 NumPy 배열로 변환
                            keypoint_values = np.array(list(new_keypoints.values()))

                            data.append(new_keypoints)
                            labels.append(sub_category)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except KeyError as e:
                    print(f"KeyError in file {file_path}: {e}")
        else:
            print(f"File {file_path} does not exist.")
    return np.array(data), np.array(labels)

def compare_keypoints(json_keypoints, mp_keypoints):
    print("JSON Keypoints:")
    for key, (x, y) in json_keypoints.items():
        print(f'{key}: {x}, {y}')

    print("MediaPipe Keypoints:")
    for idx, (x, y) in enumerate(mp_keypoints):
        print(f'Index {idx}: {x}, {y}')



# 키포인트 시각화 함수
def visualize_keypoints(image, keypoints):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for (x, y) in keypoints:
        if not np.isnan(x) and not np.isnan(y):
            plt.scatter([x], [y], c='r', s=10)
    plt.show()

# 키포인트 값 검증 함수
def check_keypoints_range(keypoints, image_shape):
    for (x, y) in keypoints:
        if not np.isnan(x) and not np.isnan(y):
            if not (0 <= x < image_shape[1] and 0 <= y < image_shape[0]):
                return False
    return True





if __name__ == "__main__":
    # 디렉토리 생성
    create_directories()

    # 예시 파일 경로 (사용자가 필요한 경우 경로를 수정할 수 있음)
    tar_file_path = [
        os.path.join(TAR_DIR, 'train.tar'),
        os.path.join(TAR_DIR, 'download.tar'), # 실제 영상 데이터
    ]         
    zip_file_path = [
        os.path.join(EXTRACTED_DIR, clean_path('121-1.3D_사람_간_상호작용_데이터(2인)/01-1.정식개방데이터/Training/02.라벨링데이터/TL.zip.part0')),
        os.path.join(EXTRACTED_DIR, 'TS.zip.part1073741824')
    ]

    # tar 파일 추출 (한 번만 실행)
    if not os.path.exists(EXTRACTED_DIR):
        extract_tar(tar_file_path, EXTRACTED_DIR)

    # zip 파일 추출 (한 번만 실행)
    if not os.path.exists(EXTRACTED_DIR):
        extract_zip(zip_file_path, EXTRACTED_DIR)

    # 필요한 JSON 파일 목록
    target_files = [
        #격려하기
        # 'JSON(230728)/M099_F105/M099_F105_38_01/M099_F105_38_01-01/M099_F105_38_01-01_frame300.json'
        # 'JSON(230728)\M099_F105\M099_F105_38_03\M099_F105_38_03-01\M099_F105_38_03-01_frame450.json',
        # 'JSON(230728)\M099_F105\M099_F105_38_03\M099_F105_38_03-01\M099_F105_38_03-01_frame600.json',
        # 'JSON(230728)\M099_F105\M099_F105_38_03\M099_F105_38_03-01\M099_F105_38_03-01_frame900.json',
        # 'JSON(230728)\M099_F105\M099_F105_38_03\M099_F105_38_03-01\M099_F105_38_03-01_frame1020.json',
        # 'JSON(230728)\M099_F105\M099_F105_38_03\M099_F105_38_03-01\M099_F105_38_03-01_frame1440.json',
        # 'JSON(230728)\M099_F105\M099_F105_38_03\M099_F105_38_03-01\M099_F105_38_03-01_frame1860.json',
        # 'JSON(230728)\M099_F105\M099_F105_38_03\M099_F105_38_03-01\M099_F105_38_03-01_frame1890.json',

        #혼내기
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame660.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame1140.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame1170.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame1380.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame1620.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame2100.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame2130.json',

        #위로하기
        'JSON(230728)\M099_F105\M099_F105_46_01\M099_F105_46_01-01\M099_F105_46_01-01_frame180.json',
        'JSON(230728)\M099_F105\M099_F105_46_01\M099_F105_46_01-01\M099_F105_46_01-01_frame630.json',
        'JSON(230728)\M099_F105\M099_F105_46_01\M099_F105_46_01-01\M099_F105_46_01-01_frame1080.json',
        'JSON(230728)\M099_F105\M099_F105_46_01\M099_F105_46_01-01\M099_F105_46_01-01_frame1320.json',
        'JSON(230728)\M099_F105\M099_F105_46_01\M099_F105_46_01-01\M099_F105_46_01-01_frame1530.json',
        'JSON(230728)\M099_F105\M099_F105_46_01\M099_F105_46_01-01\M099_F105_46_01-01_frame1980.json',
        'JSON(230728)\M099_F105\M099_F105_46_01\M099_F105_46_01-01\M099_F105_46_01-01_frame2040.json',
    
    ]

    # 비디오 파일 디렉토리 경로
    video_dir = './action_recognition/data/videos'  # 비디오 파일이 저장된 디렉토리

    # 비디오 데이터 처리
    video_data, video_labels = process_videos(video_dir, keypoint_mapping, video_labels)
    print(f"Video Data shape: {video_data.shape}")
    print(f"Video Labels shape: {video_labels.shape}")

    # JSON 데이터 전처리
    json_data, json_labels = load_and_preprocess_data(EXTRACTED_DIR, keypoint_mapping, target_files)
    print(f"JSON Data shape: {json_data.shape}")
    print(f"JSON Labels shape: {json_labels.shape}")


    # 데이터 통합
    combined_data = np.concatenate((json_data, video_data), axis=0)
    combined_labels = np.concatenate((json_labels, video_labels), axis=0)

    # 데이터 저장
    np.savez_compressed('./action_recognition/preprocessed_data.npz', data=combined_data, labels=combined_labels)
    print("Combined data saved to ./action_recognition/preprocessed_data.npz")




    # # 예제 이미지 경로 (사용자가 필요한 경우 경로를 수정할 수 있음)
    # example_image_path = 'D:\standing.jpg'
    # example_image = cv2.imread(example_image_path)
    
    # # # 첫 번째 샘플의 키포인트 시각화
    # # example_keypoints = data[0]  # 첫 번째 샘플의 키포인트
    # # visualize_keypoints(example_image, example_keypoints)
    
    # # 미디어파이프로 이미지에서 키포인트 추출
    # results = pose.process(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))
    # if results.pose_landmarks:
    #     mp_keypoints = np.array([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])

    #     # 미디어파이프로 인식한 키포인트 시각화
    #     visualize_keypoints(example_image, mp_keypoints)

    #     # 키포인트 값 검증
    #     if check_keypoints_range(mp_keypoints, example_image.shape):
    #         print("All keypoints are within the image bounds.")
    #     else:
    #         print("Some keypoints are out of the image bounds.")
    # else:
    #     print("No pose landmarks detected.")
    

def visualize_keypoints_on_video(video_path, keypoints_list):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 현재 프레임에 대한 키포인트를 가져옴
        if frame_idx < len(keypoints_list):
            keypoints = keypoints_list[frame_idx]

            # 딕셔너리 형태로 키포인트 가져오기
            for key, (x, y) in keypoints.items():  
                if not np.isnan(x) and not np.isnan(y):
                    cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 0, 255), -1)
                
        # 결과를 화면에 표시
        cv2.imshow('Keypoints Visualization', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# 이미 처리된 비디오 목록 저장
processed_videos = []

def process_videos(video_dir, keypoint_mapping, video_labels):
    all_keypoints = []
    all_labels = []

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)

        # 이미 처리된 비디오인지 확인
        if video_path in processed_videos:
            print(f"Skipping already processed video: {video_path}")
            continue  # 다음 비디오로 넘어감

        print(f"Processing video: {video_path}")
        keypoints_list = process_video(video_path, keypoint_mapping)
        label = video_labels.get(video_file, 'unknown')

        for keypoints in keypoints_list:
            all_keypoints.append(keypoints)
            all_labels.append(label)

        # 처리된 비디오 목록에 추가
        processed_videos.append(video_path)

    return np.array(all_keypoints), np.array(all_labels)


# 사용 예시
video_path = './action_recognition/data/videos/M099_F105_46_01-02.mp4'
keypoints_list = process_video(video_path, keypoint_mapping)  # 이미 처리된 keypoints 리스트
visualize_keypoints_on_video(video_path, keypoints_list)
