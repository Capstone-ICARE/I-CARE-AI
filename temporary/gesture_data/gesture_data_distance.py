import numpy as np
import math

def calculate_two_keypoints(left_point, right_point):
  left_x, left_y = left_point
  right_x, right_y = right_point
  distance = right_x - left_x
  if distance < 0:
    signed_distance = -math.sqrt(distance**2 + (right_y - left_y)**2)
  else:
    signed_distance = math.sqrt(distance**2 + (right_y - left_y)**2)
  angle = math.degrees(math.atan2(right_y - left_y, distance))
  return signed_distance, angle

def calculate_two_keypoints2(left_point, right_point):
  left = np.array(left_point)
  right = np.array(right_point)
  vector = right - left
  distance = np.linalg.norm(vector)
  angle = np.arctan2(vector[1], vector[0])
  return distance, angle

def calculate_shoulder_and_hip(left_shoulder, right_shoulder, left_hip, right_hip):
  distance_shoulder, angle_shoulder = calculate_two_keypoints2(left_shoulder, right_shoulder)
  distance_hip, angle_hip = calculate_two_keypoints2(left_hip, right_hip)
  angle_diff = np.abs(angle_shoulder-angle_hip)
  return distance_shoulder, angle_shoulder, distance_hip, angle_hip, angle_diff

def calculate_three_keypoints(top, middle, bottom):
  vector_top_to_middle = np.array(middle) - np.array(top)
  vector_middle_to_bottom = np.array(bottom) - np.array(middle)
  magnitude_top_to_middle = np.linalg.norm(vector_top_to_middle)
  magnitude_middle_to_bottom = np.linalg.norm(vector_middle_to_bottom)
  if magnitude_top_to_middle > 0 and magnitude_middle_to_bottom > 0:
    cosine_angle = np.dot(vector_top_to_middle, vector_middle_to_bottom) / (magnitude_top_to_middle * magnitude_middle_to_bottom)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_degrees = np.degrees(angle)
  else:
    angle_degrees = None
  #return [vector_top_to_middle, vector_middle_to_bottom, angle_degrees]
  return angle_degrees

def calculate_distance_nose(persons):
  p1 = persons[0]
  p2 = persons[1]
  p1_nose_x, p1_nose_y = p1[0][0], p1[0][1]
  p2_nose_x, p2_nose_y = p2[0][0], p2[0][1]
  return math.sqrt((p1_nose_x - p2_nose_x)**2 + (p1_nose_y - p2_nose_y)**2)

def calculate_distance_wrist(persons):
  p1 = persons[0]
  p2 = persons[1]
  p1_left_x, p1_left_y = p1[5][0], p1[5][1]
  p1_right_x, p1_right_y = p1[6][0], p1[6][1]
  p2_left_x, p2_left_y = p2[5][0], p2[6][1]
  p2_right_x, p2_right_y = p2[5][0], p2[6][1]
  distance_wrist_left_to_right = math.sqrt((p1_left_x - p2_right_x)**2 + (p1_left_y - p2_right_y)**2)
  distance_wrist_right_to_left = math.sqrt((p1_right_x - p2_left_x)**2 + (p1_right_y - p2_left_y)**2)
  return distance_wrist_left_to_right, distance_wrist_right_to_left

def normalize_by_left_hip(persons):
  normalized_persons = []
  for p in persons:
    new_person = []
    left_hip_x, left_hip_y = p[7][0], p[7][1]
    for keypoint in p:
      normalized_x = keypoint[0] - left_hip_x
      normalized_y = keypoint[1] - left_hip_y
      new_person.append((normalized_x, normalized_y))
    normalized_persons.append(new_person)
  return normalized_persons

desired_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
#datas = np.load('./gesture_data/data_npz/main/data_gesture_all_main.npz', allow_pickle=True)
datas = np.load('./gesture_data/data_npz/data_gesture_all_handmade.npz', allow_pickle=True)
keypoints = datas['data']
labels = datas['labels']

new_keypoints = []
#new_angles = []
# 코 좌표, 어깨, (팔의 벡터, 각도), 손 좌표, 골반, (다리의 벡터, 각도), 발 좌표
# 코 좌표, [어깨 거리, 각도], [왼쪽 팔 좌표], [오른쪽 팔 좌표], [왼쪽 다리 좌표], [오른쪽 다리 좌표]
# 1:두 사람의 코 거리, 왼쪽과 오른쪽 손목 거리, 오른쪽과 왼쪽 손목 거리, 어깨 거리, 각도, [[코 좌표, 0, 0], [왼쪽 팔 좌표], [오른쪽 팔 좌표], [왼쪽 다리 좌표], [오른쪽 다리 좌표]]
# 2:어깨 거리, 각도, 골반 거리, 각도, 두 각도 차이, [[코 좌표, 0, 0], [왼쪽 팔 좌표], [오른쪽 팔 좌표], [왼쪽 다리 좌표], [오른쪽 다리 좌표]]
for keypoint in keypoints:
  #persons = [keypoint[:13], keypoint[13:]]
  person1_keypoints = np.array(keypoint[:13])
  person2_keypoints = np.array(keypoint[13:])
  combined_vector = np.concatenate([person1_keypoints.flatten(), person2_keypoints.flatten()])
  left_shoulder = person1_keypoints[1]
  right_shoulder = person1_keypoints[2]
  shoulder_distance = np.linalg.norm(right_shoulder - left_shoulder)
  normalized_keypoints = []
  for i in range(0, 52, 2):
    normalized_keypoints.extend(((combined_vector[i], combined_vector[i+1]) - left_shoulder) / shoulder_distance)
  new_keypoints.append(normalized_keypoints)
  #angle = []
  #for p in [person1_keypoints, person2_keypoints]:
  #  angle.append(calculate_three_keypoints(p[7], p[9], p[11]))
  #  angle.append(calculate_three_keypoints(p[8], p[10], p[12]))
  #new_angles.append(angle)

  # persons = normalize_by_left_hip(persons)
  # new_data = []
  # distance = []
  # distance.append(calculate_distance_nose(persons)) # 코 거리
  # distance.extend(calculate_distance_wrist(persons)) # 손목들 거리
  # for p in persons:
  #   new_person = []
  #   new_person.append([p[0], [0, 0], [0, 0]]) # 코, padding
  #   new_person.append([p[1], p[3], p[5]]) # 왼쪽 팔
  #   new_person.append([p[2], p[4], p[6]]) # 오른쪽 팔
  #   new_person.append([p[7], p[9], p[11]]) # 왼쪽 다리
  #   new_person.append([p[8], p[10], p[12]]) # 오른쪽 다리
  #   new_data.append(new_person)
  #   distance.append(calculate_shoulder_and_hip(p[1], p[2], p[7], p[8])) # 어깨 거리, 각도, 골반 거리, 각도, 두 각도 차이
  #   distance.extend(calculate_two_keypoints(p[1], p[2])) # 어깨 거리, 각도
  # new_keypoints.append(new_data)
  # new_distances.append(distance)

save_keypoints = np.array([np.array(kp) for kp in new_keypoints])
#save_angles = np.array([np.array(d) for d in new_angles])

print(f"Keypoints shape: {save_keypoints.shape}")
#print(f"Angles shape: {save_angles.shape}")
print(f"Labels shape: {np.array(labels).shape}")

#np.savez_compressed('./gesture_data/data_npz/main/data_gesture_new_main10.npz', keypoints=save_keypoints, angles=save_angles, labels=labels)

np.savez_compressed('./gesture_data/data_npz/main/data_gesture_new_main12.npz', keypoints=save_keypoints, labels=labels)


# 모델 학습 잘 안되면 팔, 다리 각도 추가