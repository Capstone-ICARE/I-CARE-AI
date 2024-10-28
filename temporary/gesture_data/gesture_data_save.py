import numpy as np

my_w, my_h = 640, 480

def get_augment_data(keypoints, w, h):
    transformations = [
        #(rotate_keypoints, {"angle": 30, "width": w, "height": h}),
        (scale_keypoints, {"scale_x": 1.2, "scale_y": 1.2}),
        (translate_keypoints, {"tx": 10, "ty": -10}),
        (flip_keypoints_horizontal, {"width": w})
    ]
    augmented_data = []
    augmented_data.append(keypoints)
    for transform, params in transformations:
        new_keypoints = []
        for k in keypoints:
            new_keypoints.append(transform(k, **params))
        augmented_data.append(new_keypoints)
    return augmented_data # 5개의 keypoints

def rotate_point(x, y, angle, cx, cy):
    radians = np.deg2rad(angle)
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)
    
    x -= cx
    y -= cy
    
    x_new = x * cos_theta - y * sin_theta + cx
    y_new = x * sin_theta + y * cos_theta + cy
    
    return (x_new, y_new)

def rotate_keypoints(keypoints, angle, width, height): # 회전(-180~180) : 15, 30, 45, -30, -45
    cx, cy = width // 2, height // 2
    return rotate_point(keypoints[0], keypoints[1], angle, cx, cy)

def scale_keypoints(keypoints, scale_x, scale_y): # 이미지 크기(0.5~2.0) : 0.8, 1.0(원본), 1.2, 1.5
    return (keypoints[0] * scale_x, keypoints[1] * scale_y)

def translate_keypoints(keypoints, tx, ty): # 이동(-20~20)
    return (keypoints[0] + tx, keypoints[1] + ty)

def flip_keypoints_horizontal(keypoints, width): # 좌우 반전
    return (width - keypoints[0], keypoints[1])

def process_image_list(image_keypoints, image_labels):
    all_keypoints = []
    all_labels = []

    for filename, keypoints in image_keypoints.items():
        label = image_labels.get(filename, 'Neutral')
        augmented_keypoints = get_augment_data(keypoints, my_w, my_h)
        for kp in augmented_keypoints:
            all_keypoints.append(kp) # [(x, y), (x, y), (x, y), (x, y), ...]
            all_labels.append(label)

    return np.array([np.array(kp) for kp in all_keypoints]), np.array(all_labels)

if __name__ == "__main__":
    image_labels = {
        'KakaoTalk_20241007_140104689_26.jpg' : '두 손으로 하이파이브',
        'KakaoTalk_20241007_152049515_03.png' : '두 손으로 하이파이브',
        'KakaoTalk_20241007_152049515_05.png' : '두 손으로 하이파이브',
        'KakaoTalk_20241007_152049515_06.jpg' : '두 손으로 하이파이브',
    }
    image_keypoints = {
        'KakaoTalk_20241007_140104689_26.jpg' : [(229, 197), (211, 227), (197, 229), (271, 219), (269, 215), (311, 183), (307, 181), (227, 292), (204, 301), (238, 372), (217, 377), (221, 419), (227, 450), (418, 190), (455, 217), (441, 216), (369, 208), (361, 212), (323, 174), (322, 182), (446, 315), (406, 309), (457, 370), (426, 366), (471, 433), (437, 417)],
        'KakaoTalk_20241007_152049515_03.png' : [(291, 109), (285, 153), (278, 153), (330, 132), (327, 128), (343, 77), (334, 77), (284, 243), (275, 247), (317, 338), (308, 355), (252, 403), (251, 411), (427, 101), (441, 125), (437, 126), (384, 122), (377, 124), (367, 68), (358, 74), (464, 246), (438, 246), (408, 362), (393, 343), (495, 329), (487, 310)],
        'KakaoTalk_20241007_152049515_05.png' : [(249, 137), (242, 170), (236, 184), (267, 130), (278, 142), (311, 97), (311, 102), (229, 258), (250, 267), (256, 356), (271, 367), (218, 354), (219, 339), (380, 123), (392, 148), (385, 151), (350, 125), (338, 131), (331, 71), (328, 74), (389, 248), (385, 246), (391, 381), (366, 355), (441, 287), (447, 295)],
        'KakaoTalk_20241007_152049515_06.jpg' : [(419, 78), (526, 110), (402, 105), (467, 126), (301, 126), (389, 95), (258, 100), (516, 212), (400, 214), (461, 333), (368, 327), (505, 434), (336, 438), (256, 126), (138, 158), (248, 153), (177, 143), (358, 135), (227, 95), (377, 92), (138, 269), (240, 271), (245, 353), (179, 367), (236, 439), (133, 450)]
    }

    datas, labels = process_image_list(image_keypoints, image_labels)
    print(f"Data shape: {datas.shape}")
    print(f"Labels shape: {labels.shape}")
    np.savez_compressed('./data_npz/data_gesture_handmade.npz', data=datas, labels=labels)