import cv2
import threading
import time
from gesture.gesture import Gesture

class Camera:
  def __init__(self, pose, yolo_model, gesture_model, gesture_label_encoder):
    self.gesture = Gesture(pose, yolo_model, gesture_model, gesture_label_encoder)
    self.running = False
    self.thread = None
    self.current_frame = None
    self.cap = None
    self.end = False

  def get_current_frame(self):
    return self.current_frame if self.current_frame is not None else None
  
  def predict_gesture(self):
    return self.gesture.predict_label(self.current_frame)
  
  def fix_cor_label(self):
    return self.gesture.fix_cor_label()
  
  def get_hint_path(self):
    return self.gesture.get_hint_path()

  def open(self):
    self.running = True
    self.end = False
    self.cap = cv2.VideoCapture(0)
    self.thread = threading.Thread(target=self.run)
    self.thread.start()
    return self.gesture.get_cor_label()

  def close(self):
    self.running = False
    while not (self.end):
      time.sleep(0.1)
    if self.thread:
      self.thread.join()
      self.thread = None
    self.current_frame = None
    self.cap.release()
    return self.gesture.get_result()

  def run(self):
    print('opening')
    while self.running:
      ret, frame = self.cap.read()
      if not ret:
        print('closing')
        break
      self.current_frame = frame
    self.cap.release()
    self.end = True