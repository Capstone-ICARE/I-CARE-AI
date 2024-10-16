import tarfile
import zipfile
import os
import json
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import torch
import concurrent.futures


# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) 

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 

# 원본 데이터를 저장할 디렉토리 경로
RAW_DIR = os.path.normpath('./action_recognition/data/raw')
TAR_DIR = os.path.join(RAW_DIR, 'tar_files')
EXTRACTED_DIR = os.path.join(RAW_DIR, 'extracted')

def main():
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
        #혼내기
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame660.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame1140.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame1170.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame1380.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame1620.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame2100.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-01\M099_F105_39_01-01_frame2130.json',

        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-02\M099_F105_39_01-02_frame660.json'
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-02\M099_F105_39_01-02_frame1140.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-02\M099_F105_39_01-02_frame1170.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-02\M099_F105_39_01-02_frame1380.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-02\M099_F105_39_01-02_frame1620.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-02\M099_F105_39_01-02_frame2100.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-02\M099_F105_39_01-02_frame2130.json',

        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-03\M099_F105_39_01-03_frame660.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-03\M099_F105_39_01-03_frame1140.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-03\M099_F105_39_01-03_frame1170.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-03\M099_F105_39_01-03_frame1380.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-03\M099_F105_39_01-03_frame1620.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-03\M099_F105_39_01-03_frame2100.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-03\M099_F105_39_01-03_frame2130.json',

        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-04\M099_F105_39_01-04_frame660.json'
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-04\M099_F105_39_01-04_frame1140.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-04\M099_F105_39_01-04_frame1170.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-04\M099_F105_39_01-04_frame1380.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-04\M099_F105_39_01-04_frame1620.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-04\M099_F105_39_01-04_frame2100.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-04\M099_F105_39_01-04_frame2130.json',

        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-05\M099_F105_39_01-05_frame660.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-05\M099_F105_39_01-05_frame1140.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-05\M099_F105_39_01-05_frame1170.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-05\M099_F105_39_01-05_frame1380.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-05\M099_F105_39_01-05_frame1620.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-05\M099_F105_39_01-05_frame2100.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-05\M099_F105_39_01-05_frame2130.json',

        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-06\M099_F105_39_01-06_frame660.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-06\M099_F105_39_01-06_frame1140.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-06\M099_F105_39_01-06_frame1170.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-06\M099_F105_39_01-06_frame1380.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-06\M099_F105_39_01-06_frame1620.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-06\M099_F105_39_01-06_frame2100.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-06\M099_F105_39_01-06_frame2130.json',

        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-07\M099_F105_39_01-07_frame660.json'
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-07\M099_F105_39_01-07_frame1140.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-07\M099_F105_39_01-07_frame1170.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-07\M099_F105_39_01-07_frame1380.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-07\M099_F105_39_01-07_frame1620.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-07\M099_F105_39_01-07_frame2100.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-07\M099_F105_39_01-07_frame2130.json',
        
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-08\M099_F105_39_01-08_frame660.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-08\M099_F105_39_01-08_frame1140.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-08\M099_F105_39_01-08_frame1170.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-08\M099_F105_39_01-08_frame1380.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-08\M099_F105_39_01-08_frame1620.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-08\M099_F105_39_01-08_frame2100.json',
        'JSON(230728)\M099_F105\M099_F105_39_01\M099_F105_39_01-08\M099_F105_39_01-08_frame2130.json',

        # 격려하기 #38
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-01\M099_F105_38_01-01_frame180.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-01\M099_F105_38_01-01_frame300.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-01\M099_F105_38_01-01_frame630.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-01\M099_F105_38_01-01_frame1200.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-01\M099_F105_38_01-01_frame1350.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-01\M099_F105_38_01-01_frame1530.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-01\M099_F105_38_01-01_frame2070.json',
        
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-02\M099_F105_38_01-02_frame180.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-02\M099_F105_38_01-02_frame300.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-02\M099_F105_38_01-02_frame630.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-02\M099_F105_38_01-02_frame1200.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-02\M099_F105_38_01-02_frame1350.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-02\M099_F105_38_01-02_frame1530.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-02\M099_F105_38_01-02_frame2070.json',

        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-03\M099_F105_38_01-03_frame180.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-03\M099_F105_38_01-03_frame300.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-03\M099_F105_38_01-03_frame630.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-03\M099_F105_38_01-03_frame1200.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-03\M099_F105_38_01-03_frame1350.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-03\M099_F105_38_01-03_frame1530.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-03\M099_F105_38_01-03_frame2070.json',

        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-04\M099_F105_38_01-04_frame180.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-04\M099_F105_38_01-04_frame300.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-04\M099_F105_38_01-04_frame630.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-04\M099_F105_38_01-04_frame1200.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-04\M099_F105_38_01-04_frame1350.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-04\M099_F105_38_01-04_frame1530.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-04\M099_F105_38_01-04_frame2070.json',

        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-05\M099_F105_38_01-05_frame180.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-05\M099_F105_38_01-05_frame300.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-05\M099_F105_38_01-05_frame630.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-05\M099_F105_38_01-05_frame1200.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-05\M099_F105_38_01-05_frame1350.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-05\M099_F105_38_01-05_frame1530.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-05\M099_F105_38_01-05_frame2070.json',

        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-06\M099_F105_38_01-06_frame180.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-06\M099_F105_38_01-06_frame300.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-06\M099_F105_38_01-06_frame630.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-06\M099_F105_38_01-06_frame1200.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-06\M099_F105_38_01-06_frame1350.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-06\M099_F105_38_01-06_frame1530.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-06\M099_F105_38_01-06_frame2070.json',

        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-07\M099_F105_38_01-07_frame180.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-07\M099_F105_38_01-07_frame300.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-07\M099_F105_38_01-07_frame630.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-07\M099_F105_38_01-07_frame1200.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-07\M099_F105_38_01-07_frame1350.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-07\M099_F105_38_01-07_frame1530.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-07\M099_F105_38_01-07_frame2070.json',

        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-08\M099_F105_38_01-08_frame180.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-08\M099_F105_38_01-08_frame300.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-08\M099_F105_38_01-08_frame630.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-08\M099_F105_38_01-08_frame1200.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-08\M099_F105_38_01-08_frame1350.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-08\M099_F105_38_01-08_frame1530.json',
        'JSON(230728)\M099_F105\M099_F105_38_01\M099_F105_38_01-08\M099_F105_38_01-08_frame2070.json',

        # Neutral

    ]

    video_labels = {

    'M099_F105_38_03-01.mp4' : '격려하기',
    'M099_F105_38_03-02.mp4' : '격려하기',
    'M099_F105_38_03-03.mp4' : '격려하기',
    'M099_F105_38_03-04.mp4' : '격려하기',
    'M099_F105_38_03-05.mp4' : '격려하기',
    'M099_F105_38_03-06.mp4' : '격려하기',
    'M099_F105_38_03-07.mp4' : '격려하기',
    'M099_F105_38_03-08.mp4' : '격려하기',
    'M099_F105_38_01-01.mp4' : '격려하기',
    'M099_F105_38_01-02.mp4' : '격려하기',
    'M099_F105_38_01-03.mp4' : '격려하기',
    'M099_F105_38_01-04.mp4' : '격려하기',
    'M099_F105_38_01-05.mp4' : '격려하기',
    'M099_F105_38_01-06.mp4' : '격려하기',
    'M099_F105_38_01-07.mp4' : '격려하기',
    'M099_F105_38_01-08.mp4' : '격려하기',
    'M099_F105_38_02-01.mp4' : '격려하기',
    'M099_F105_38_02-02.mp4' : '격려하기',
    'M099_F105_38_02-03.mp4' : '격려하기',
    'M099_F105_38_02-04.mp4' : '격려하기',
    'M099_F105_38_02-05.mp4' : '격려하기',
    'M099_F105_38_02-06.mp4' : '격려하기',
    'M099_F105_38_02-07.mp4' : '격려하기',
    'M099_F105_38_02-08.mp4' : '격려하기',
    'pat1.mov' : '격려하기',
    'pat2.mov' : '격려하기',
    'pat3.mov' : '격려하기',
    'pat4.mov' : '격려하기',
    'pat5.mov' : '격려하기',
    'pat6.mov' : '격려하기',
    'pat7.mov' : '격려하기',
    'pat8.mov' : '격려하기',
    'pat9.mov' : '격려하기',
    'pat10.mov' : '격려하기',
    'pat11.mov' : '격려하기',

    '(1)M099_F105_39_04-01.mp4': '혼내기',
    '(1)M099_F105_39_04-03.mp4': '혼내기',
    '(1)M099_F105_39_04-05.mp4': '혼내기',
    '(1)M099_F105_39_04-06.mp4': '혼내기',
    '(1)M099_F105_39_04-07.mp4': '혼내기',
    '(1)M099_F105_39_04-08.mp4': '혼내기',
    'M099_F105_39_01-01.mp4':'혼내기',
    'M099_F105_39_01-02.mp4':'혼내기',
    'M099_F105_39_01-03.mp4':'혼내기',
    'M099_F105_39_01-04.mp4':'혼내기',
    'M099_F105_39_01-05.mp4':'혼내기',
    'M099_F105_39_01-06.mp4':'혼내기',
    'M099_F105_39_01-07.mp4':'혼내기',
    'M099_F105_39_01-08.mp4':'혼내기',
    'M099_F105_39_02-01.mp4':'혼내기',
    'M099_F105_39_02-02.mp4':'혼내기',
    'M099_F105_39_02-03.mp4':'혼내기',
    'M099_F105_39_02-04.mp4':'혼내기',
    'M099_F105_39_02-05.mp4':'혼내기',
    'M099_F105_39_02-06.mp4':'혼내기',
    'M099_F105_39_02-07.mp4':'혼내기',
    'M099_F105_39_02-08.mp4':'혼내기',
    'M099_F105_39_02-08.mp4':'혼내기',
    'scold1.mov':'혼내기',
    'scold2.mov':'혼내기',
    'scold3.mov':'혼내기',
    'scold4.mov':'혼내기',
    'scold5.mov':'혼내기',
    'scold6.mov':'혼내기',

    '(1)M099_F105_36_02-01.mp4':'Neutral',
    '(1)M099_F105_36_02-02.mp4':'Neutral',
    '(1)M099_F105_36_02-03.mp4':'Neutral',
    '(1)M099_F105_36_02-04.mp4':'Neutral',
    '(1)M099_F105_36_02-05.mp4':'Neutral',
    '(1)M099_F105_36_02-06.mp4':'Neutral',
    '(1)M099_F105_36_02-07.mp4':'Neutral',
    '(1)M099_F105_36_02-08.mp4':'Neutral',
    '(2)M099_F105_19_02-01.mp4':'Neutral',
    '(2)M099_F105_19_02-02.mp4':'Neutral',
    '(2)M099_F105_19_02-03.mp4':'Neutral',
    '(2)M099_F105_19_02-04.mp4':'Neutral',
    '(2)M099_F105_19_02-05.mp4':'Neutral',
    '(2)M099_F105_19_02-06.mp4':'Neutral',
    '(2)M099_F105_19_02-07.mp4':'Neutral',
    '(2)M099_F105_19_02-08.mp4':'Neutral',
    '(2)M099_F105_20_02-01.mp4':'Neutral',
    '(2)M099_F105_20_02-02.mp4':'Neutral',
    '(2)M099_F105_20_02-03.mp4':'Neutral',
    '(2)M099_F105_20_02-04.mp4':'Neutral',
    '(2)M099_F105_20_02-05.mp4':'Neutral',
    '(2)M099_F105_20_02-06.mp4':'Neutral',
    '(2)M099_F105_20_02-07.mp4':'Neutral',
    '(2)M099_F105_20_02-08.mp4':'Neutral',
    'M099_F105_28_03-01.mp4':'Neutral',
    'M099_F105_28_03-02.mp4':'Neutral',
    'M099_F105_28_03-03.mp4':'Neutral',
    'M099_F105_28_03-04.mp4':'Neutral',
    'M099_F105_28_03-05.mp4':'Neutral',
    'M099_F105_28_03-06.mp4':'Neutral',
    'M099_F105_28_03-07.mp4':'Neutral',
    'M099_F105_28_03-08.mp4':'Neutral',
    'neutral1.mov':'Neutral',
    'neutral2.mov':'Neutral',

    }                        

    #sequence_length = 30                       

    # 비디오 파일 디렉토리 경로
    video_dir = './action_recognition/data/videos'  

    # 비디오 데이터 처리
    video_data, video_labels = process_videos(video_dir, keypoint_mapping, video_labels)
    print(f"Video Data shape: {video_data.shape}")
    print(f"Video Labels shape: {video_labels.shape}")

    # JSON 데이터 전처리
    json_data, json_labels = load_and_preprocess_data(EXTRACTED_DIR, keypoint_mapping, target_files)
    print(f"JSON Data shape: {json_data.shape}")
    print(f"JSON Labels shape: {json_labels.shape}")

    # 데이터 통합
    #json_data_expanded = np.expand_dims(json_data, axis=1)  # 시퀀스 차원 추가
    # json_data = json_data.tolist()
    # json_data = json_data * 30
    #json_data = np.array(json_data)
    # json_data = np.repeat(json_data[:, np.newaxis, :], 30, axis=1)
    # print(f"JSON Data shape: {json_data.shape}")

    # combined_data = np.concatenate((json_data, video_data), axis=0)
    # combined_labels = np.concatenate((json_labels, video_labels), axis=0)

    # 데이터 저장
    np.savez_compressed('./action_recognition/preprocessed_data_test05.npz', data=video_data, labels=video_labels)
    print("Combined data saved to ./action_recognition/preprocessed_data_test05.npz")

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

# old_keypoints:mediapipe, new_keypoints에 새로운 인덱스(ex.Nose)로 key, value 형태로 저장
def map_keypoints(old_keypoints, keypoint_mapping):
    new_keypoints = {}

    if isinstance(old_keypoints, list):
        for idx, (x, y) in enumerate(old_keypoints):
            if idx in keypoint_mapping:  # MediaPipe 랜드마크 번호 확인
                new_idx = keypoint_mapping[idx]
                new_keypoints[new_idx] = [x, y]

    else:
        print("Unsupported keypoints format")
    return new_keypoints

def get_augment_data(keypoints_sequence, w, h):
    transformations = [
        # (rotate_keypoints, {"angle": 30, "width": w, "height": h}),
        (scale_keypoints, {"scale_x": 1.2, "scale_y": 1.2}),
        (translate_keypoints, {"tx": 10, "ty": -10}),
        (flip_keypoints_horizontal, {"width": w})
    ]
    augmented_data = []
    augmented_data.append(keypoints_sequence)
    for transform, params in transformations:
        keypoints_seq = []
        for (p1, p2) in keypoints_sequence: # [({'Nose': [x, y], 'LeftArm': [x, y], ...}, {}), ({}, {}), ({}, {})...]
            person1 = {}
            person2 = {}
            for key, keypoints in p1.items():
                person1[key] = transform(keypoints, **params)
            for key, keypoints in p2.items():
                person2[key] = transform(keypoints, **params)
            added_keypoints = (person1, person2)
            keypoints_seq.append(added_keypoints)
        augmented_data.append(keypoints_seq)
    return augmented_data # 5개의 keypoints_sequence

def rotate_point(x, y, angle, cx, cy):
    radians = np.deg2rad(angle)
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)
    
    x -= cx
    y -= cy
    
    x_new = x * cos_theta - y * sin_theta + cx
    y_new = x * sin_theta + y * cos_theta + cy
    
    return [x_new, y_new]

def rotate_keypoints(keypoints, angle, width, height): # 회전(-180~180) : 15, 30, 45, -30, -45
    cx, cy = width // 2, height // 2
    return rotate_point(keypoints[0], keypoints[1], angle, cx, cy)

def scale_keypoints(keypoints, scale_x, scale_y): # 이미지 크기(0.5~2.0) : 0.8, 1.0(원본), 1.2, 1.5
    return [keypoints[0] * scale_x, keypoints[1] * scale_y]

def translate_keypoints(keypoints, tx, ty): # 이동(-20~20)
    return [keypoints[0] + tx, keypoints[1] + ty]

def flip_keypoints_horizontal(keypoints, width): # 좌우 반전
    return [width - keypoints[0], keypoints[1]]

# 데이터 증강 설정 (imgaug)
# augmenter = iaa.Sequential([
#     iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0))),
#     iaa.Fliplr(0.5),  # 좌우 반전
#     iaa.Affine(rotate=(-20, 20)),
#     iaa.Multiply((0.8, 1.2), per_channel=0.2)  # 밝기 조절
# ])

# extracting and mapping keypoints 
def process_video(video_path, keypoint_mapping, sequence_length=20):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 비디오의 초당 프레임 수 (120)
    interval = max(1, int(fps/5)) # 초당 5프레임 처리

    keypoints_sequence = [] # 각 프레임에서 추출된 매핑된 키포인트 저장
    frame_count = 0

    while cap.isOpened(): #and len(keypoints_sequence) < sequence_length
        ret, frame = cap.read()
        if not ret:
            break
        temp_frame = frame
        # 매 'process_interval' 프레임마다 처리
        if frame_count % interval == 0:
            frame_keypoints = process_frame(frame, keypoint_mapping)
            if frame_keypoints:
                keypoints_sequence.append(frame_keypoints)

        frame_count += 1

    cap.release()

    

    h, w = temp_frame.shape[:2]
    augment_sequence_list = get_augment_data(keypoints_sequence, w, h)
    new_sequence_list = []

    for sequence in augment_sequence_list:
        if len(sequence) == 0:
            return 0
        # 시퀀스 길이 표준화
        for i in range(0, len(sequence), sequence_length):
            chunk = sequence[i:i + sequence_length]
            if len(chunk) < sequence_length:
                for i in range(sequence_length - len(chunk)):
                    chunk.append(chunk[-(i + 1)])
            new_sequence_list.append(chunk)
    return new_sequence_list


def process_frame(frame, keypoint_mapping):
    # data_augmentation
    #frame_aug = augmenter.augment_image(frame)

    # YOLOv5로 사람 감지
    frame_resized = cv2.resize(frame, (320, 240))
    results = model(frame_resized)
    detections = results.pred[0]

    people_detected = 0
    people_with_pose = 0
    frame_keypoints=[] # 각 프레임에서 사람 2명 키포인트 저장
    keypoints_tuple = ()
    keypoints_sequence = []

    # 탐지된 각 사람에 대해 MediaPipe Pose 적용
    for det in detections:
        if det[5] == 0:  # person class
            people_detected += 1
            x1, y1, x2, y2 = map(int, det[:4])

            # 프레임 원래 크기
            h_original, w_original = frame.shape[:2]

            # YOLO bounding box 좌표를 원래 프레임 크기로 변환
            x1 = int(x1 * (w_original / 320))
            x2 = int(x2 * (w_original / 320))
            y1 = int(y1 * (h_original / 240))
            y2 = int(y2 * (h_original / 240))

            person_img = frame[y1:y2, x1:x2]
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            # MediaPipe BlazePose로 포즈 추정
            result = pose.process(person_rgb)

            # landmark 성공적으로 추정한 경우
            if result.pose_landmarks:
                people_with_pose += 1

                landmarks = result.pose_landmarks.landmark
                old_keypoints = [(lm.x, lm.y) for lm in landmarks]
                new_keypoints = map_keypoints(old_keypoints, keypoint_mapping)
                frame_keypoints.append(new_keypoints)

                # new_keypoints를 시각화
                for key, (x, y) in new_keypoints.items():
                    cx = int(x * person_img.shape[1])
                    cy = int(y * person_img.shape[0])
                    cv2.circle(person_img, (cx, cy), 5, (0, 255, 0), -1)

        # Display results
        # cv2.imshow('Keypoints Visualization', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Check if we have keypoints for two people
    if len(frame_keypoints) == 2:
        keypoints_tuple = (frame_keypoints[0], frame_keypoints[1])  # Tuple of two people's keypoints
        return keypoints_tuple

    return None    
            
    
def pad_sequence(sequence, target_length):
    if len(sequence) >= target_length:
        return sequence
    
    padding = [sequence[-1]] * (target_length - len(sequence))
    return sequence + padding

def process_videos(video_dir, keypoint_mapping, video_labels):
    all_sequences = []
    all_labels = []

    # target_videos = [

    #     'M099_F105_38_03-01.mp4',
    #     'M099_F105_38_03-02.mp4',
    #     # 'M099_F105_39_02-01.mp4',
    #     # 'M099_F105_39_02-02.mp4'
    # ]

    # 비디오 파일 목록
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]#if f.endswith('.mp4', '.mov')
    #video_files = [f for f in os.listdir(video_dir) if f in target_videos]


    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"Processing video: {video_path}")

        # 비디오에서 키포인트 추출
        sequence_list = process_video(video_path, keypoint_mapping, sequence_length=20)
        if sequence_list == 0:
            continue
        # 라벨 추출
        label = video_labels.get(video_file, 'unknown')  # video_labels 딕셔너리에서 라벨을 가져옴
        if isinstance(sequence_list, list):
            print("리스트")
        else:
            print("sequence_list가 리스트가 아닙니다:", type(sequence_list))
        for sequence in sequence_list:
            if len(sequence) == 0:
                continue
            # 키포인트와 라벨 저장
            all_sequences.append(sequence)
            all_labels.append(label)

    return np.array(all_sequences), np.array(all_labels)



def load_and_preprocess_data(json_dir, keypoint_mapping, target_files, sequence_length=20):
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
                    
                    data_tuple = ()
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

                            for key, (x, y) in new_keypoints.items():
                                normalized_x = x / 1920
                                normalized_y = y / 1080
                                new_keypoints[key] = [normalized_x, normalized_y]
                            
                            # 딕셔너리 값 (좌표 쌍)만 추출하여 NumPy 배열로 변환
                            # keypoint_values = np.array(list(new_keypoints.values()))
                            data_tuple = data_tuple + (new_keypoints, )
                    if len(data_tuple) == 2:
                        sequence_list = get_augment_data([data_tuple], 1920, 1080)
                        for sequence in sequence_list:
                            tup = sequence[0]
                            data.append(tup)
                            labels.append(sub_category)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except KeyError as e:
                    print(f"KeyError in file {file_path}: {e}")
        else:
            print(f"File {file_path} does not exist.")

    return np.array(data), np.array(labels)

if __name__ == "__main__":
    main()
    # 디렉토리 생성
    #create_directories()

    






