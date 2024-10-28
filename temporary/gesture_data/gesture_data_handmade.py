import os
import json
import numpy as np
import cv2

my_w, my_h = 640, 640

def get_augment_data(keypoints, w, h):
  transformations = [
    (scale_keypoints, {"scale_x": 1.2, "scale_y": 1.2}),
    (translate_keypoints, {"tx": 10, "ty": -10})
  ]
  augmented_data = []
  augmented_data.append(keypoints)
  for transform, params in transformations:
    new_keypoints = []
    for k in keypoints:
      new_keypoints.append(transform(k, **params))
    augmented_data.append(new_keypoints)

  flipped_keypoints = [flip_keypoints_horizontal(k, width=w) for k in keypoints]
  augmented_data.append(flipped_keypoints)
  for transform, params in transformations:
    new_keypoints = []
    for k in flipped_keypoints:
      new_keypoints.append(transform(k, **params))
    augmented_data.append(new_keypoints)
  return augmented_data # 6개의 keypoints

def scale_keypoints(keypoints, scale_x, scale_y): # 이미지 크기(0.5~2.0) : 0.8, 1.0(원본), 1.2, 1.5
  scaled_x = min(max(keypoints[0] * scale_x, 0), my_w)
  scaled_y = min(max(keypoints[1] * scale_y, 0), my_h)
  return (scaled_x, scaled_y)

def translate_keypoints(keypoints, tx, ty): # 이동(-20~20)
  trans_x = min(max(keypoints[0] + tx, 0), my_w)
  trans_y = min(max(keypoints[1] + ty, 0), my_h)
  return (trans_x, trans_y)

def flip_keypoints_horizontal(keypoints, width): # 좌우 반전
  return (width - keypoints[0], keypoints[1])

# JSON 파일을 읽는 함수
def load_json_file(filename):
  with open(filename, 'r', encoding='utf-8') as file:
    return json.load(file)
  
def fix_json_to_image(keypoints, jsonname, image_directory):
  scaled_keypoints = []
  json_name = os.path.splitext(jsonname)[0]
  image_extensions = ['.png', '.jpg', '.jpeg']
  for filename in os.listdir(image_directory):
    file_name, file_extension = os.path.splitext(filename)
    if file_name == json_name and file_extension.lower() in image_extensions:
      image_path = os.path.join(image_directory, filename)
      image = cv2.imread(image_path)
      if image is None:
        #raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
        return fix_json(keypoints)
      original_h, original_w, _ = image.shape
      image_resized = cv2.resize(image, (my_w, my_h))
      scale_w = my_w / original_w
      scale_h = my_h / original_h
      for x, y in keypoints:
        new_x = int(x * scale_w)
        new_y = int(y * scale_h)
        scaled_keypoints.append((new_x, new_y))
        #cv2.circle(image_resized, (new_x, new_y), 4, (0, 255, 0) if len(scaled_keypoints) <= 13 else (255, 0, 0), -1)
      #cv2.imshow('Image', image_resized)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()
      return scaled_keypoints
  return fix_json(keypoints)

def fix_json(keypoints):
  scaled_keypoints = []
  original_h, original_w = 4032, 3024
  scale_w = my_w / original_w
  scale_h = my_h / original_h
  for x, y in keypoints:
    new_x = int(x * scale_w)
    new_y = int(y * scale_h)
    scaled_keypoints.append((new_x, new_y))
  return scaled_keypoints

def process_image(json_path, jsonname, image_directory):
  data = load_json_file(json_path)
  keypoints = []

  if len(data['shapes']) == 2:
    shape1, shape2 = data['shapes']
    #if shape1['label'] == 'person01' and shape2['label'] == 'person02':
    if len(shape1['points']) == 13 and len(shape2['points']) == 13:
      keypoints = [(x, y) for x, y in shape1['points']] + [(x, y) for x, y in shape2['points']]
      if not os.path.exists(image_directory):
        return fix_json(keypoints)
      else:
        return fix_json_to_image(keypoints, jsonname, image_directory)
    else:
      print(f'각 points != 13:{json_path}')
    #else:
    #  print(f'label이 person01, person02 아님:{json_path}')
  else:
    print(f'shapes 배열 길이 != 2:{json_path}')
  return []

def process_image_list(image_directory, json_directory, label):
  all_keypoints = []
  all_labels = []
  a = 0
  for jsonname in os.listdir(json_directory):
    if jsonname.endswith(('.json')):
      json_path = os.path.join(json_directory, jsonname)
      keypoints = process_image(json_path, jsonname, image_directory)
      if keypoints != []:
        a += 1
        augmented_keypoints = get_augment_data(keypoints, my_w, my_h)
        for kp in augmented_keypoints:
          all_keypoints.append(kp) # [[(x, y), (x, y), (x, y), (x, y), ...], []]
          all_labels.append(label)
  print(f'{json_directory} : {a}개')
  return np.array([np.array(kp) for kp in all_keypoints]), np.array(all_labels)

if __name__ == "__main__":
  image_directory = [
    './gesture_data/images/handmade/gesture01',
    './gesture_data/images/handmade/gesture02',
    './gesture_data/images/handmade/gesture03',
    './gesture_data/images/handmade/gesture04',
    './gesture_data/images/handmade/gesture05',
    './gesture_data/images/handmade/gesture06',
    './gesture_data/images/handmade/gesture07',
    #'./gesture_data/images/handmade/gesture08',
    './gesture_data/images/neutral-x'
  ]
  json_directory = [
    './gesture_data/images/handmade/gesture01_json',
    './gesture_data/images/handmade/gesture02_json',
    './gesture_data/images/handmade/gesture03_json',
    './gesture_data/images/handmade/gesture04_json',
    './gesture_data/images/handmade/gesture05_json',
    './gesture_data/images/handmade/gesture06_json',
    './gesture_data/images/handmade/gesture07_json',
    #'./gesture_data/images/handmade/gesture08_json',
    './gesture_data/images/neutral_json-x'
  ]
  label = [
    '옆구리 늘리기',
    '마주보고 두 손바닥 맞대기',
    '서로 등지고 양손잡고 잡아당기기',
    '정면보고 손잡고 발바닥 맞대기',
    '마주보고 손잡고 발목잡고 뒤로 당기기',
    '손 잡고 만세하기',
    '인사하기',
    #'준비 동작, 차렷!',
    'Neutral'
  ]
  keypoint_file_name = './gesture_data/data_npz/data_gesture_all_handmade.npz'

  all_datas = []
  all_labels = []
  for i in range(0, len(label)):
    datas, labels = process_image_list(image_directory[i], json_directory[i], label[i])
    all_datas.append(datas)
    all_labels.append(labels)
  combined_datas = np.concatenate(all_datas, axis=0)
  combined_labels = np.concatenate(all_labels, axis=0)
  print(f"Data shape: {combined_datas.shape}")
  print(f"Labels shape: {combined_labels.shape}")
  np.savez_compressed(keypoint_file_name, data=combined_datas, labels=combined_labels)

