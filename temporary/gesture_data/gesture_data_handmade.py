import os
import json
import numpy as np
import cv2

my_w, my_h = 640, 640

def get_augment_data(keypoints, w, h):
  transformations = [
    (scale_keypoints, {"scale_x": 1.2, "scale_y": 1.2}),
    (translate_keypoints, {"tx": 10, "ty": -10}),
    (flip_keypoints_horizontal, {"width": w})
  ]
  augmented_data = []
  augmented_data.append(keypoints)
  for transform, params in transformations:
    new_keypoints = []
    for k in keypoints:
      new_keypoints.append(transform(k, **params))
    augmented_data.append(new_keypoints)
  return augmented_data # 5개의 keypoints

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
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
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
  return []

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
      #return fix_json_to_image(keypoints, jsonname, image_directory)
      return fix_json(keypoints)
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

  for jsonname in os.listdir(json_directory):
    if jsonname.endswith(('.json')):
      json_path = os.path.join(json_directory, jsonname)
      keypoints = process_image(json_path, jsonname, image_directory)

      if keypoints != []:
        augmented_keypoints = get_augment_data(keypoints, my_w, my_h)
        for kp in augmented_keypoints:
          all_keypoints.append(kp) # [(x, y), (x, y), (x, y), (x, y), ...]
          all_labels.append(label)

  return np.array([np.array(kp).flatten() for kp in all_keypoints]), np.array(all_labels)

if __name__ == "__main__":
  image_directory = './gesture_data/images/handmade/gesture05'
  json_directory = './gesture_data/images/handmade/neutral_json'
  label = 'Neutral'
  keypoint_file_name = './gesture_data/data_npz/data_gesture_neutral_handmade.npz'

  datas, labels = process_image_list(image_directory, json_directory, label)
  print(f"Data shape: {datas.shape}")
  print(f"Labels shape: {labels.shape}")
  np.savez_compressed(keypoint_file_name, data=datas, labels=labels)

