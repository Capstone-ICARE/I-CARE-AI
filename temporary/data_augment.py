import numpy as np

def augment_data(keypoints_sequence, w, h):
    transformations = [
        (rotate_keypoints, {"angle": 30, "width": w, "height": h}),
        (scale_keypoints, {"scale_x": 1.2, "scale_y": 1.2}),
        (translate_keypoints, {"tx": 10, "ty": -10}),
        (flip_keypoints_horizontal, {"width": w})
    ]
    augmented_data = []
    augmented_data.append(keypoints_sequence)
    for transform, params in transformations:
        keypoints_seq = []
        for (p1, p2) in keypoints_sequence: # [({'Nose': [x, y], 'LeftArm': [x, y], ...}, {}), ({}, {}), ({}, {})...]
            person1 = {}
            person2 = {}
            for key, keypoints in p1:
                person1[key] = transform(keypoints, **params)
            for key, keypoints in p2:
                person2[key] = transform(keypoints, **params)
            added_keypoints = (person1, person2)
            keypoints_seq.append(added_keypoints)
        augmented_data.append(keypoints_seq)
    return augmented_data # 5개의 keypoints_sequence: [[], [], [], [], []]

def rotate_point(x, y, angle, cx, cy):
    radians = np.deg2rad(angle)
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)
    
    x -= cx
    y -= cy
    
    x_new = x * cos_theta - y * sin_theta + cx
    y_new = x * sin_theta + y * cos_theta + cy
    
    return [x_new, y_new]

def rotate_keypoints(keypoints, angle, width, height): # 회전(-180~180) : 15, 30, 45, -30, -45
    cx, cy = width // 2, height // 2
    return rotate_point(keypoints[0], keypoints[1], angle, cx, cy)

def scale_keypoints(keypoints, scale_x, scale_y): # 이미지 크기(0.5~2.0) : 0.8, 1.0(원본), 1.2, 1.5
    return [keypoints[0] * scale_x, keypoints[1] * scale_y]

def translate_keypoints(keypoints, tx, ty): # 이동(-20~20)
    return [keypoints[0] + tx, keypoints[1] + ty]

def flip_keypoints_horizontal(keypoints, width): # 좌우 반전
    return [width - keypoints[0], keypoints[1]]
