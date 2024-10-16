import random
import os
import cv2
import numpy as np

class Gesture:
  def __init__(self, pose, yolo_model, gesture_model, gesture_label_encoder):
    self.pose = pose
    self.yolo_model = yolo_model
    self.gesture_model = gesture_model
    self.gesture_label_encoder = gesture_label_encoder
    self.desired_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    self.labels = {
      '옆구리 늘리기': 'hint.jpg',
      '마주보고 두 손바닥 맞대기': 'normal.jpg',
      '서로 등지고 양손잡고 잡아당기기': 'answer.jpg',
      '마주보고 손바닥 맞대고 발 뒤로 들기': 'gesture04.jpg',
      '등 맞대고 앉기(스쿼트)': 'gesture05.jpg',
      '정면보고 손잡고 발바닥 맞대기': 'answer.jpg',
      '안마하기': 'hint.jpg',
      '마주보고 손잡고 발목잡고 뒤로 당기기': 'normal.jpg',
      #'팔하트': 'answer.jpg'
    }
    self.cor_label = random.choice(list(self.labels.keys()))
    self.check = False
    self.check_count = 0
    self.result = []

  def fix_cor_label(self):
    while True:
      temp_cor_label = random.choice(list(self.labels.keys()))
      if self.cor_label != temp_cor_label:
        self.cor_label = temp_cor_label
        break
    self.check = False
    self.check_count = 0
    return self.cor_label
  
  def get_cor_label(self):
    return self.cor_label
  
  def get_result(self):
    return self.result
  
  def get_hint_path(self):
    #directory = './gesture/images/hint'
    directory = './images/gesture/hint'
    hint_image = self.labels[self.cor_label]
    hint_path = os.path.join(directory, hint_image)
    if os.path.exists(hint_path):
      return hint_path

  def predict_label(self, current_frame):
    #directory = './gesture/images'
    directory = './images/gesture'
    frame_keypoints = self.process_frame(current_frame)
    if frame_keypoints == []:
      return self.increment_check_count()
    X_keypoints = np.array([np.array(frame_keypoints).flatten()])
    prediction = self.gesture_model.predict(X_keypoints)
    predicted_label = self.gesture_label_encoder.inverse_transform([np.argmax(prediction)])[0]
    print(predicted_label)
    if self.cor_label == predicted_label:
      if self.check:
        frame_path = self.generate_unique_filename(directory)
        cv2.imwrite(frame_path, current_frame)
        self.result.append({'fileName': frame_path, 'label': predicted_label})
        return True, True
      else:
        self.check = True
        return True, False
    else:
      return self.increment_check_count()

  def increment_check_count(self):
    if self.check and self.check_count < 5:
      self.check_count += 1
      return True, False
    else:
      self.check = False
      self.check_count = 0
      return False, False

  def generate_unique_filename(self, directory):
    while True:
      random_number = random.randint(100000, 999999)
      filename = f"frame_{random_number}.jpg"
      file_path = os.path.join(directory, filename)
      if not os.path.exists(file_path):
        return file_path
      
  def process_frame(self, current_frame):
    W, H = 640, 640
    keypoints = []
    image = current_frame
    image_resized = cv2.resize(image, (W, H))
    image_resized_original = image_resized.copy()

    detections = self.yolo_model(image_resized).pred[0]
    people_detected = 0
    people_with_pose = 0

    for det in detections:
      if det[5] == 0:
        people_detected += 1
        x1, y1, x2, y2 = map(int, det[:4])

        new_x1, new_y1, new_x2, new_y2 = max(0, x1 - 25), max(0, y1 - 15), min(W, x2 + 25), min(H, y2 + 15)
        person_img = image_resized_original[new_y1:new_y2, new_x1:new_x2]
        person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(person_rgb)

        if pose_result.pose_landmarks:
          people_with_pose += 1
          for idx in self.desired_landmarks:
            landmark = pose_result.pose_landmarks.landmark[idx]
            w, h = new_x2 - new_x1, new_y2 - new_y1
            x, y = int(landmark.x * w), int(landmark.y * h)
            keypoints.append((new_x1 + x, new_y1 + y))

    if people_detected == 2 and people_with_pose == 2:
        return keypoints
    else:
        #print(f"X, {people_detected}, {people_with_pose}")
        return []