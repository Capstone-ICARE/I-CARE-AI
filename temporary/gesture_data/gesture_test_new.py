import os
import json
import numpy as np
import cv2
import mediapipe as mp
import torch
from tensorflow.keras.models import load_model
import joblib
import threading
import time
import math

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) 

# YOLOv5 모델 로드
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 

desired_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

my_w, my_h = 640, 640

def process_image(current_frame):
  keypoints = []
  image = current_frame
  original_h, original_w, _ = image.shape
  image_resized = cv2.resize(image, (my_w, my_h))

  detections = yolo_model(image_resized).pred[0]
  #people_detected = 0
  people_with_pose = 0

  for det in detections:
    if people_with_pose >= 2:
      break
    if det[5] == 0:
      #people_detected += 1
      x1, y1, x2, y2 = map(int, det[:4])

      original_x1 = int(((x1)/my_w) * original_w)
      original_y1 = int(((y1)/my_h) * original_h)
      original_x2 = int(((x2)/my_w) * original_w)
      original_y2 = int(((y2)/my_h) * original_h)
      #cv2.rectangle(image, (original_x1, original_y1), (original_x2, original_y2), (0, 0, 255), 2)

      person_img = image_resized[y1:y2, x1:x2]
      person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

      pose_result = pose.process(person_rgb)

      if pose_result.pose_landmarks:
        people_with_pose += 1
        for idx in desired_landmarks:
          landmark = pose_result.pose_landmarks.landmark[idx]
          if landmark.x == 0 and landmark.y == 0:
            print('(0,0)입니다.')
            return []
          h, w, _ = person_img.shape
          x, y = int(landmark.x * w), int(landmark.y * h)
          keypoints.append((x1 + x, y1 + y))
          original_x = int(((x1 + x)/my_w) * original_w)
          original_y = int(((y1 + y)/my_h) * original_h)
          #cv2.circle(image, (original_x, original_y), 4, (0, 255, 0) if people_with_pose == 1 else (255, 0, 0), -1)
  if people_with_pose == 2:
    #cv2.imshow('Test', image)
    #cv2.waitKey(1)
    return keypoints
  else:
    #print(f"X, {people_detected}, {people_with_pose}")
    return []
    
def process_image_sample(current_frame):
    image = current_frame
    original_h, original_w, _ = image.shape
    image_resized = cv2.resize(image, (my_w, my_h))
    detections = yolo_model(image_resized).pred[0]

    for det in detections:
        if det[5] == 0:
            x1, y1, x2, y2 = map(int, det[:4])

            original_x1 = int(((x1)/my_w) * original_w)
            original_y1 = int(((y1)/my_h) * original_h)
            original_x2 = int(((x2)/my_w) * original_w)
            original_y2 = int(((y2)/my_h) * original_h)
            cv2.rectangle(image, (original_x1, original_y1), (original_x2, original_y2), (0, 0, 255), 4)

            person_img = image_resized[y1:y2, x1:x2]
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(person_rgb)
            if pose_result.pose_landmarks:
                for idx in desired_landmarks:
                    landmark = pose_result.pose_landmarks.landmark[idx]
                    h, w, _ = person_img.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    original_x = int(((x1 + x)/my_w) * original_w)
                    original_y = int(((y1 + y)/my_h) * original_h)
                    cv2.circle(image, (original_x, original_y), 4, (0, 255, 0), -1)
    return image

def open_cam(running, shared_data):
  cap = cv2.VideoCapture(1) # 0
  if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    return
  while running[0]:
    ret, frame = cap.read()
    if not ret:
      print("프레임을 읽을 수 없습니다.")
      break
    shared_data['current_frame'] = frame
    new_frame = process_image_sample(frame)
    cv2.imshow('Camera', new_frame)
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
      running[0] = False
      break
  cap.release()
  cv2.destroyAllWindows()

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

def get_keypoints_and_angles(keypoints):
  normalized_keypoints = []
  new_angles = []
  person1_keypoints = np.array(keypoints[:13])
  person2_keypoints = np.array(keypoints[13:])
  combined_vector = np.concatenate([person1_keypoints.flatten(), person2_keypoints.flatten()])
  left_shoulder = person1_keypoints[1]
  right_shoulder = person1_keypoints[2]
  shoulder_distance = np.linalg.norm(right_shoulder - left_shoulder)
  for i in range(0, 52, 2):
    new_keypoints = ((combined_vector[i], combined_vector[i+1]) - left_shoulder) / shoulder_distance
    normalized_keypoints.extend([new_keypoints[0], new_keypoints[1]])
  for p in [person1_keypoints, person2_keypoints]:
    new_angles.append(calculate_three_keypoints(p[7], p[9], p[11]))
    new_angles.append(calculate_three_keypoints(p[8], p[10], p[12]))
  return np.array(normalized_keypoints), np.array([new_angles])

def get_keypoints(keypoints):
  normalized_keypoints = []
  person1_keypoints = np.array(keypoints[:13])
  person2_keypoints = np.array(keypoints[13:])
  combined_vector = np.concatenate([person1_keypoints.flatten(), person2_keypoints.flatten()])
  left_shoulder = person1_keypoints[1]
  right_shoulder = person1_keypoints[2]
  shoulder_distance = np.linalg.norm(right_shoulder - left_shoulder)
  for i in range(0, 52, 2):
    new_keypoints = ((combined_vector[i], combined_vector[i+1]) - left_shoulder) / shoulder_distance
    normalized_keypoints.extend([new_keypoints[0], new_keypoints[1]])
  return np.array(normalized_keypoints)

def main():
  running = [True]
  shared_data = {'current_frame': None}
  threshold = 0.6
  model = load_model('./gesture_data/data_npz/main/gesture_new_model19.keras')
  label_encoder = joblib.load('./gesture_data/data_npz/main/new_label_encoder_8.pkl')
  thread = threading.Thread(target=open_cam, args=(running, shared_data))
  thread.start()
  while running[0]:
    time.sleep(2)
    current_frame = shared_data['current_frame'] 
    if current_frame is not None:
      new_keypoints = process_image(current_frame)
      if len(new_keypoints) == 0:
        print('Predicted Label: Neutral')
      else:
        #X_new_keypoints, X_new_angles = get_keypoints_and_angles(new_keypoints)
        #predictions = model.predict([np.expand_dims(X_new_keypoints, axis=0), X_new_angles])
        X_new_keypoints = get_keypoints(new_keypoints)
        predictions = model.predict(np.expand_dims(X_new_keypoints, axis=0))
        max_prob = np.max(predictions)
        if max_prob < threshold:
          predicted_label = 'Neutral'
        else:
          predicted_label = label_encoder.inverse_transform([np.argmax(predictions, axis=1)])[0]
        print(f'Predicted Label: {predicted_label}, (Confidence: {round(max_prob, 3)})')

if __name__ == "__main__":
  main()