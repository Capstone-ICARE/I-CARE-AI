from flask import Flask, request, send_file, jsonify
import os
<<<<<<< HEAD
from profile.profile import Profile
=======
from profile.profile import create_profile
>>>>>>> 7156bb63743c67538305b21e56b2609cf77f465f
from icon.icon import Icon

# Flask 애플리케이션 객체 생성
app = Flask(__name__)

# 기본 라우트 정의
@app.route('/profile', methods=['POST'])
def post_profile():
    json_data = request.json
    diary = json_data.get('diary')
    file_name = json_data.get('fileName')
<<<<<<< HEAD
    if profile.create_profile(diary, file_name):
        return "Success profile"
    else:
        return "Failure profile", 400
=======
    if create_profile(diary, file_name):
        return "Success profile"
    else:
        return "Failure profile"
>>>>>>> 7156bb63743c67538305b21e56b2609cf77f465f
    
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
<<<<<<< HEAD
        return jsonify({"icons": icons}), 400
=======
        return "Icon error", 400
>>>>>>> 7156bb63743c67538305b21e56b2609cf77f465f
        

# 애플리케이션 실행
if __name__ == '__main__':
<<<<<<< HEAD
    profile = Profile()
    icon = Icon()
    app.run(debug=True)
    
# https://huggingface.co/monologg/kobert/tree/main
=======
    icon = Icon()
    app.run(debug=True)
>>>>>>> 7156bb63743c67538305b21e56b2609cf77f465f
