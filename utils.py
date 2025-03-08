import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """3点の座標から角度を計算"""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_joint_angles(landmarks):
    """関節角度を計算"""
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    body_angle = calculate_angle(shoulder, hip, knee)  # 股関節角度
    floor_angle = calculate_angle(shoulder, hip, right_hip)  # 体幹角度
    return body_angle, floor_angle

