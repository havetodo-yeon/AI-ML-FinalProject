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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
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
    import re
    
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
    filepath = f"{RAW_VIDEO_DIR}/{name}.mp4"
    yt.streams.filter(file_extension='mp4').first().download(
        output_path=RAW_VIDEO_DIR,
        filename=f"{name}.mp4"
    )
    print(f"🎬 Downloaded → {filepath}")
    return filepath, name

def load_raw_keypoints(npz_path):
    """Load BlazePose raw data (T,33,3)."""
    data = np.load(npz_path, allow_pickle=True)["data"]
    df = pd.DataFrame(data, columns=["frame","landmark","x","y","z","visibility"])

    # 프레임 누락이 있으면(포즈 미검출) 길이가 줄어들어 시퀀스가 끊겨 보입니다.
    # 최대 프레임까지 (T,33,3) 텐서를 만들고 NaN은 시간축 선형보간으로 채웁니다.
    if df.empty:
        return np.zeros((0, 33, 3), dtype=np.float32)

    max_f = int(df["frame"].max())
    T = max_f + 1
    seq = np.full((T, 33, 3), np.nan, dtype=np.float32)

    for _, r in df.iterrows():
        f = int(r["frame"])
        lm = int(r["landmark"])
        if 0 <= lm < 33 and 0 <= f < T:
            seq[f, lm, 0] = float(r["x"])
            seq[f, lm, 1] = float(r["y"])
            seq[f, lm, 2] = float(r["z"])

    # time interpolation per landmark/coord
    t_idx = np.arange(T)
    for lm in range(33):
        for c in range(3):
            v = seq[:, lm, c]
            good = ~np.isnan(v)
            if good.sum() == 0:
                seq[:, lm, c] = 0.0
            elif good.sum() == 1:
                seq[:, lm, c] = v[good][0]
            else:
                seq[:, lm, c] = np.interp(t_idx, t_idx[good], v[good]).astype(np.float32)

    return seq   # (T,33,3)

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
    # 경로도 os.path.join을 사용하여 안전하게 생성
    out_path = os.path.join(dir_path, f"{name}.npz")
    np.savez(out_path, data=df.to_numpy())
    print(f"📌 Saved 3D keypoints → {out_path}")

    base = os.path.basename(out_path).replace(".npz", "")
    # -------------------------------
    # 1) Load raw BlazePose sequence
    # -------------------------------
    raw_seq = load_raw_keypoints(out_path)      # (T,33,3)
    
    # 복잡한 댄스 영상의 경우 튀는 모션 보간 적용 (선택적)
    # 빠르고 심한 움직임이 있는 경우 True로 설정
    APPLY_OUTLIER_INTERPOLATION = True
    if APPLY_OUTLIER_INTERPOLATION:
        try:
            from src.postprocess import interpolate_outliers
            raw_seq = interpolate_outliers(
                raw_seq,
                velocity_threshold=3.0,
                window_size=5,
                method="linear",
                strength=0.8  # raw 데이터는 조금 덜 강하게 보간
            )
            print("✅ Applied outlier interpolation to raw data")
        except ImportError:
            print("⚠️ Could not import interpolate_outliers, skipping...")
    
    raw_out_path = os.path.join(RAW_TEST_DIR, f"{base}_raw.npy")
    np.save(raw_out_path, raw_seq)
    print(f"Saved raw → {raw_out_path}")


def main():
    video_path, name = download_youtube('https://www.youtube.com/shorts/xDOe8icUNTo')
    extract_3d_keypoints(video_path, RAW_TEST_DIR, name)
    # TODO : URL list to download various motion dataset.
    # url_list

if __name__ == "__main__":
    main()
