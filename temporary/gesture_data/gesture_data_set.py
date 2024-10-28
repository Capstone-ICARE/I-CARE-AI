import cv2
import os
import numpy as np

my_w, my_h = 640, 480

max_points = 26

def draw_points(image, points):
    for (x, y) in points:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image

def process_image(image_path):
    keypoints = []
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (my_w, my_h))

    while True:
        color = (0, 255, 0) if len(keypoints) < 13 else (255, 0, 0)
        image_display = draw_points(image_resized.copy(), keypoints)
        cv2.imshow("Image", image_display)

        key = cv2.waitKey(0)

        if key == ord('q'): # 종료
            break
        elif key == ord('u'): # 뒤로 가기
            if keypoints:
                keypoints.pop()
        elif key == ord('a'): # 점 추가
            if len(keypoints) < max_points:
                def get_mouse_click(event, x, y ,flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        keypoints.append((x, y))
                        cv2.destroyWindow("Image")
                cv2.setMouseCallback("Image", get_mouse_click)

                while len(keypoints) < max_points and not cv2.getWindowProperty("Image", 0) >= 0:
                    image_display = draw_points(image_resized.copy(), keypoints)
                    cv2.imshow("Image", image_display)
                    cv2.waitKey(1)
    print(f'keypoints: {keypoints}')
    cv2.destroyAllWindows()
    return keypoints


def process_image_list(already_set_keypoints):
    all_keypoints = []
    directory = './images/handmade'

    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')) and filename not in already_set_keypoints:
            image_path = os.path.join(directory, filename)
            keypoints = process_image(image_path)
            if keypoints != []:
                all_keypoints.append(keypoints) # [(x, y), (x, y), (x, y), (x, y), ...]
                print(f"'{filename}' : {keypoints},")

    return np.array([np.array(kp) for kp in all_keypoints])

if __name__ == "__main__":
    already_set_keypoints = [
        'KakaoTalk_20241007_140104689_26.jpg',
        'KakaoTalk_20241007_152049515_03.png',
        'KakaoTalk_20241007_152049515_05.png',
        'KakaoTalk_20241007_152049515_06.jpg'
    ]

    datas = process_image_list(already_set_keypoints)
    print(f"Data shape: {datas.shape}")