import numpy as np

def augment_data(keypoints_tuple, w, h):
    keypoints_list = []
    keypoints_list.append(keypoints_tuple)
    
    a = rotate_keypoints(keypoints_tuple[0], 30, w, h)
    b = rotate_keypoints(keypoints_tuple[1], 30, w, h)
    keypoints_list.append((a, b))

def rotate_point(x, y, angle, cx, cy):
    radians = np.deg2rad(angle)
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)
    
    x -= cx
    y -= cy
    
    x_new = x * cos_theta - y * sin_theta + cx
    y_new = x * sin_theta + y * cos_theta + cy
    
    return x_new, y_new

def rotate_keypoints(keypoints, angle, width, height): # 회전(-180~180) : 15, 30, 45, -30, -45
    cx, cy = width // 2, height // 2
    return {key: [rotate_point(keypoints[0], keypoints[1], angle, cx, cy)] for key, keypoints in keypoints.items()}

def scale_keypoints(keypoints, scale_x, scale_y): # 이미지 크기(0.5~2.0) : 0.8, 1.0(원본), 1.2, 1.5
    return {key: [keypoints[0] * scale_x, keypoints[1] * scale_y] for key, keypoints in keypoints.items()}

def translate_keypoints(keypoints, tx, ty): # 이동(-20~20)
    return {key: [keypoints[0] + tx, keypoints[1] + ty] for key, keypoints in keypoints.items()}

def flip_keypoints_horizontal(keypoints, width): # 좌우 반전
    return {key: [width - keypoints[0], keypoints[1]] for key, keypoints in keypoints.items()}
