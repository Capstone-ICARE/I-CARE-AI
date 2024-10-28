from flask import Flask, request, send_file, jsonify, Response
from flask_cors import CORS
from tensorflow.keras.models import load_model
import mediapipe as mp
import torch
import os
import cv2
import time
import json
import joblib
from profile.profile import Profile
from icon.icon import Icon
from video.video import Video
from gesture.camera import Camera

# Flask 애플리케이션 객체 생성
app = Flask(__name__)
CORS(app)

child_videos = {}
child_cameras = {}

video_model = load_model('./video/datamodel/action_recognition_model.h5')
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

gesture_model = load_model('./gesture/model/gesture_model_01.h5')
gesture_label_encoder = joblib.load('./gesture/model/label_encoder.pkl')

# 기본 라우트 정의
@app.route('/profile', methods=['POST'])
def post_profile():
    json_data = request.json
    diary = json_data.get('diary')
    file_name = json_data.get('fileName')
    if profile.create_profile(diary, file_name):
        return "Success profile"
    else:
        return "Failure profile", 400
    
@app.route('/profile', methods=['GET'])
def get_profile():
    file_name = request.args.get('fileName')
    if file_name:
        profile_path = f'./images/profile/{file_name}'
        if os.path.exists(profile_path):
            return send_file(profile_path, mimetype='image/jpeg')
        else:
            return "Profile image not found", 404
    else:
        return "Profile file name not provided", 400
    
@app.route('/icon', methods=['POST'])
def post_icon():
    json_data = request.json
    diary = json_data.get('diary')
    icons = icon.get_icons(diary)
    if icons:
        return jsonify({"icons": icons})
    else:
        return jsonify({"icons": icons}), 400
    
@app.route('/video/running', methods=['GET'])
def get_video_running():
    child_id = request.args.get('childId')
    if child_id in child_videos:
        video = child_videos.get(child_id)
        return jsonify({"running": video.running}), 200
    else:
        return jsonify({"running": False}), 200

@app.route('/video/start', methods=['POST'])
def start_video():
    child_id = request.args.get('childId')
    if child_id not in child_videos:
        child_videos[child_id] = Video(video_model, yolo_model)
    video = child_videos.get(child_id)
    if video:
        video.start_video()
        return jsonify({"message": "Start Video"}), 200

@app.route('/video/stop', methods=['POST'])
def stop_video():
    child_id = request.args.get('childId')
    video = child_videos.get(child_id)
    if video:
        result = video.stop_video()
        return jsonify(result), 200

@app.route('/video/status', methods=['GET'])
def video_status():
    child_id = request.args.get('childId')
    video = child_videos.get(child_id)
    def generate(video):
        if video:
            original_count = 0
            limit = 0
            while video.running:
                time.sleep(3)
                current_label = video.get_current_label()
                if current_label is None:
                    current_label = '-'
                count = video.get_label_count()
                if original_count == count:
                    limit += 1
                    if limit == 4:
                        current_label = '-'
                    elif limit > 4:
                        continue
                else:
                    limit = 0
                original_count = count
                yield f"data:{json.dumps({'currentLabel': current_label})}\n\n"
    return Response(generate(video), mimetype='text/event-stream')

@app.route('/video/stream', methods=['GET'])
def video_stream():
    child_id = request.args.get('childId')
    video = child_videos.get(child_id)
    def gen_frames(video):  # 비디오 프레임을 생성하는 함수
        if video:
            while video.running:
                frame = video.get_current_frame()
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen_frames(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/image', methods=['GET'])
def get_video_image():
    file_name = request.args.get('fileName')
    if file_name:
        video_path = file_name
        if os.path.exists(video_path):
            return send_file(video_path, mimetype='image/jpeg')
        else:
            return "Video image not found", 404
    else:
        return "Video file name not provided", 400
    
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

# 애플리케이션 실행
if __name__ == '__main__':
    profile = Profile()
    icon = Icon()
    app.run(debug=True)
    
# https://huggingface.co/monologg/kobert/tree/main
