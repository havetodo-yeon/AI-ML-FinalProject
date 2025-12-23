# scripts/download_and_extract.py

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pytubefix import YouTube
import mediapipe as mp
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "raw_videos")
RAW_KEYPOINT_DIR = os.path.join(ROOT, "data", "raw_keypoints")
RAW_TEST_DIR = os.path.join(ROOT, "data", "test_keypoints")

os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
os.makedirs(RAW_KEYPOINT_DIR, exist_ok=True)
os.makedirs(RAW_TEST_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def sanitize_filename(filename):
    """Windows에서 허용되지 않는 문자를 제거하거나 대체합니다."""
    # Windows에서 허용되지 않는 문자: < > : " / \ | ? *
    invalid_chars = '<>:"/\\|?*'
    
    # 비표준 따옴표 및 특수 문자를 언더스코어로 변경
    filename = filename.replace("'", "_").replace("'", "_").replace("'", "_")
    filename = filename.replace(""", "_").replace(""", "_")
    
    # 허용되지 않는 문자 제거
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    
    # 공백을 언더스코어로 변경
    filename = filename.replace(" ", "_")
    
    # 연속된 언더스코어를 하나로
    while "__" in filename:
        filename = filename.replace("__", "_")
    
    # 앞뒤 언더스코어 제거
    filename = filename.strip("_")
    
    # 빈 문자열이면 기본값 사용
    if not filename:
        filename = "video"
    
    # Windows 파일명 길이 제한 (255자)
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename

def download_youtube(url):
    yt = YouTube(url)
    name = sanitize_filename(yt.title)
    filepath = os.path.join(RAW_VIDEO_DIR, f"{name}.mp4")
    yt.streams.filter(file_extension='mp4').first().download(
        output_path=RAW_VIDEO_DIR,
        filename=f"{name}.mp4"
    )
    print(f"🎬 Downloaded → {filepath}")
    return filepath, name


def extract_3d_keypoints(video_path, dir_path =RAW_KEYPOINT_DIR, name ="Data"):
    # 파일명도 정리 (혹시 모를 경우를 대비)
    name = sanitize_filename(name)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_rows = []

    pbar = tqdm(total=total_frames, desc="Extracting BlazePose 3D",
                ascii=True,          # unicode 막대 → ASCII 막대로 변경
                dynamic_ncols=False  # 윈도우 콘솔 버그 방지
                )
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_model.process(rgb)

        if res.pose_world_landmarks:
            for i, lm in enumerate(res.pose_world_landmarks.landmark):
                pose_rows.append({
                    "frame": frame_idx,
                    "landmark": i,
                    "x": lm.x,
                    "y": -lm.y,      # flip for unity-like coords
                    "z": lm.z,
                    "visibility": lm.visibility,
                })

        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    df = pd.DataFrame(pose_rows)

    # extract local positions from pelvis

    # 경로도 os.path.join을 사용하여 안전하게 생성
    out_path = os.path.join(dir_path, f"{name}.npz")
    np.savez(out_path, data=df.to_numpy())
    print(f"📌 Saved 3D keypoints → {out_path}")

def main():
    video_path, name = download_youtube('https://www.youtube.com/shorts/jkyfropXop0')
    extract_3d_keypoints(video_path, RAW_KEYPOINT_DIR, name)
    # TODO : URL list to download various motion dataset.
    # url_list

if __name__ == "__main__":
    main()
