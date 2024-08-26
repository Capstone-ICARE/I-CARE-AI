from flask import Flask, request
from profile.profile import create_wordcloud

# Flask 애플리케이션 객체 생성
app = Flask(__name__)

# 기본 라우트 정의
@app.route('/profile', methods=['POST'])
def home():
    json_data = request.json
    diarys = json_data.get('diarys')
    file_name = json_data.get('file_name')
    if create_wordcloud(diarys, file_name):
        return "Success profile"
    else:
        return "Failure profile"

# 애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)