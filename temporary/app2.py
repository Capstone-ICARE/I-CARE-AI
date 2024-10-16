from flask import Flask, request, send_file, jsonify, Response
from flask_cors import CORS
import mediapipe as mp
import torch
import cv2
import os
from tensorflow.keras.models import load_model
import joblib
from gesture.camera import Camera

app = Flask(__name__)
CORS(app)

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) 

# YOLOv5 모델 로드
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

gesture_model = load_model('./gesture/model/gesture_model_01.h5')
gesture_label_encoder = joblib.load('./gesture/model/label_encoder.pkl')

child_cameras = {}

@app.route('/gesture/start', methods=['GET'])
def start_camera():
  child_id = request.args.get('childId')
  if child_id not in child_cameras:
    child_cameras[child_id] = Camera(pose, yolo_model, gesture_model, gesture_label_encoder)
  camera = child_cameras.get(child_id)
  if camera:
    cor_label = camera.open()
    return jsonify({"cor_label": cor_label}), 200

@app.route('/gesture/stop', methods=['POST'])
def stop_camera():
  child_id = request.args.get('childId')
  camera = child_cameras.get(child_id)
  if camera:
    result = camera.close()
    del child_cameras[child_id]
    return jsonify(result), 200
  
@app.route('/gesture/again', methods=['GET'])
def again_gesture_label():
  child_id = request.args.get('childId')
  camera = child_cameras.get(child_id)
  if camera:
    cor_label = camera.fix_cor_label()
    return jsonify({"cor_label": cor_label}), 200

@app.route('/gesture/hint', methods=['GET'])
def get_hint_image():
  child_id = request.args.get('childId')
  camera = child_cameras.get(child_id)
  if camera:
    hint_path = camera.get_hint_path()
    if os.path.exists(hint_path):
      print(f'Hint : {hint_path}')
      return send_file(hint_path, mimetype='image/jpeg')
    else:
      return "Hint image not found", 404

@app.route('/gesture/predict', methods=['GET'])
def get_gesture_predict():
  child_id = request.args.get('childId')
  camera = child_cameras.get(child_id)
  if camera:
    check1, check2 = camera.predict_gesture()
    return jsonify({'check1': check1, 'check2': check2}), 200

@app.route('/gesture/stream', methods=['GET'])
def get_camera_stream():
  child_id = request.args.get('childId')
  camera = child_cameras.get(child_id)
  def gen_frames(camera):
    if camera:
      while camera.running:
        frame = camera.get_current_frame()
        if frame is not None:
          ret, buffer = cv2.imencode('.jpg', frame)
          frame = buffer.tobytes()
          yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
  return Response(gen_frames(camera),
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gesture/image', methods=['GET'])
def get_gesture_image():
  file_name = request.args.get('fileName')
  if file_name:
    gesture_path = file_name
    if os.path.exists(gesture_path):
      return send_file(gesture_path, mimetype='image/jpeg')
    else:
      return "Gesture image not found", 404
  else:
    return "Gesture file name not provided", 400

if __name__ == '__main__':
    app.run(debug=True)