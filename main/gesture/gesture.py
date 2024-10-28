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
      '정면보고 손잡고 발바닥 맞대기': 'gesture04.jpg',
      '마주보고 손잡고 발목잡고 뒤로 당기기': 'gesture05.jpg',
      '손 잡고 만세하기': 'answer.jpg',
      '인사하기': 'hint.jpg',
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
    directory = './static/gesture/hint'
    hint_image = self.labels[self.cor_label]
    hint_path = os.path.join(directory, hint_image)
    if os.path.exists(hint_path):
      return hint_path

  def predict_label(self, current_frame):
    #directory = './gesture/images'
    directory = './static/gesture'
    threshold = 0.6
    frame_keypoints = self.process_frame(current_frame)
    if frame_keypoints == []:
      return self.increment_check_count()
    X_new_keypoints = self.fix_keypoints(frame_keypoints)
    predictions = self.gesture_model.predict(np.expand_dims(X_new_keypoints, axis=0))
    max_prob = np.max(predictions)
    if max_prob < threshold:
      predicted_label = 'Neutral'
    else:
      predicted_label = self.gesture_label_encoder.inverse_transform([np.argmax(predictions, axis=1)])[0]
    print(f'Predicted Label: {predicted_label}, (Confidence: {round(max_prob, 3)})')
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
      
  def fix_keypoints(self, keypoints):
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
      
  def process_frame(self, current_frame):
    W, H = 640, 640
    keypoints = []
    image = current_frame
    image_resized = cv2.resize(image, (W, H))
    image_resized_original = image_resized.copy()

    detections = self.yolo_model(image_resized).pred[0]
    people_with_pose = 0

    for det in detections:
      if people_with_pose >= 2:
        break
      if det[5] == 0:
        x1, y1, x2, y2 = map(int, det[:4])

        person_img = image_resized_original[y1:y2, x1:x2]
        person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(person_rgb)

        if pose_result.pose_landmarks:
          people_with_pose += 1
          for idx in self.desired_landmarks:
            landmark = pose_result.pose_landmarks.landmark[idx]
            h, w, _ = person_img.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            keypoints.append((x1 + x, y1 + y))

    if people_with_pose == 2:
      return keypoints
    else:
      #print(f"X, {people_detected}, {people_with_pose}")
      return []