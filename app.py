import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import pandas as pd
import plotly.express as px
import os
from utils import get_joint_angles

# MediaPipeのセットアップ
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Webアプリのタイトル
st.title("📏 前屈・後屈動作分析アプリ")
st.write("動画をアップロードして、関節角度を解析し、前屈・後屈の深まり度を測定します！")

# ユーザー選択: 左右側面 + 動作種類（前屈 or 後屈）
side_option = st.radio("📌 分析する側面を選択", ["左側面", "右側面"])
motion_type = st.radio("📌 分析する動作を選択", ["前屈", "後屈"])

# 動画アップロード
uploaded_file = st.file_uploader("📤 動画をアップロードしてください", type=["mp4", "mov"])

if uploaded_file:
    # 一時ファイルとして保存
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    cap = cv2.VideoCapture(temp_file.name)

    # 動画の基本情報取得
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 保存用の動画ファイル作成
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 解析用データ
    joint_data = []
    initial_hip_angle = None
    initial_trunk_angle = None
    initial_knee_angle = None
    initial_ankle_angle = None

    with mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.80) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 側面ごとに適切な関節マーカーを選択
                if side_option == "左側面":
                    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                else:
                    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                    knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
                    ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                
                # マーカーの描画
                keypoints = [shoulder, hip, knee, ankle]
                for i in range(len(keypoints) - 1):
                    p1 = keypoints[i]
                    p2 = keypoints[i + 1]
                    cv2.line(image, (int(p1.x * frame_width), int(p1.y * frame_height)),
                             (int(p2.x * frame_width), int(p2.y * frame_height)), (0, 200, 200), 4)  # 青緑の線
                    cv2.circle(image, (int(p1.x * frame_width), int(p1.y * frame_height)), 8, (255, 140, 0), -1)  # 濃いオレンジのマーカー
                cv2.circle(image, (int(ankle.x * frame_width), int(ankle.y * frame_height)), 8, (255, 140, 0), -1)  # 足関節マーカー
                
                def calculate_angle(a, b, c):
                    ba = np.array([a.x - b.x, a.y - b.y])
                    bc = np.array([c.x - b.x, c.y - b.y])
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    return np.degrees(np.arccos(cosine_angle))
                
                hip_angle = 180 - calculate_angle(shoulder, hip, knee)
                _, trunk_angle = get_joint_angles(landmarks)
                knee_angle = 180 - calculate_angle(hip, knee, ankle)
                
                if initial_hip_angle is None:
                    initial_hip_angle = hip_angle
                if initial_trunk_angle is None:
                    initial_trunk_angle = trunk_angle
                if initial_knee_angle is None:
                    initial_knee_angle = knee_angle
                
                frame_data = {
                    "Time (s)": cap.get(cv2.CAP_PROP_POS_FRAMES) / fps,
                    "股関節角度": hip_angle,
                    "体幹角度": trunk_angle,
                    "膝関節角度": knee_angle,
                }
                joint_data.append(frame_data)
                
            out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
    cap.release()
    out.release()

    df = pd.DataFrame(joint_data)
    st.write("✅ 解析完了！")
    
    if initial_trunk_angle is not None:
        trunk_range_of_motion = abs(initial_trunk_angle - max(df["体幹角度"])) if motion_type == "前屈" else abs(initial_trunk_angle - max(df["体幹角度"]))
        st.write(f"📊 深まり度（体幹角度変化）: {trunk_range_of_motion:.2f}°")
    
    if initial_hip_angle is not None:
        hip_range_of_motion = abs(initial_hip_angle - max(df["股関節角度"])) if motion_type == "前屈" else abs(initial_hip_angle - max(df["股関節角度"]))
        st.write(f"📊 股関節角度変化: {hip_range_of_motion:.2f}°")
    
    fig = px.line(df, x="Time (s)", y=["股関節角度", "体幹角度", "膝関節角度"],
                  title=f"{motion_type}の関節角度変化", labels={"value": "角度 (度)", "variable": "関節"})
    st.plotly_chart(fig)

    st.subheader("🎥 解析結果の動画")
    st.video(output_video_path)

    st.download_button("📥 解析動画をダウンロード", open(output_video_path, "rb"), "analysis.mp4", "video/mp4")
    csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    df.to_csv(csv_path, index=False)
    st.download_button("📥 CSVデータをダウンロード", open(csv_path, "rb"), "analysis.csv", "text/csv")
