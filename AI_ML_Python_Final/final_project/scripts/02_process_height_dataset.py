import os
import numpy as np
import pandas as pd
from glob import glob
import os, sys
from scipy.signal import medfilt
from scipy.signal import savgol_filter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.viser_test import PoseViser

RAW_KEYPOINTS_DIR = os.path.join(ROOT, "data", "raw_keypoints")
OUT_DIR = os.path.join(ROOT, "data", "processed")

os.makedirs(OUT_DIR, exist_ok=True)


def load_raw_keypoints(npz_path):
    """Load BlazePose raw data (T,33,3)."""
    data = np.load(npz_path, allow_pickle=True)["data"]
    df = pd.DataFrame(data, columns=["frame","landmark","x","y","z","visibility"])

    # 프레임 누락(포즈 미검출)이 있으면 시퀀스 길이가 줄어들어 학습/테스트가 불안정해질 수 있음
    # 최대 프레임까지 (T,33,3)을 만들고 NaN은 시간축 선형보간으로 채움
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


def create_height_corrected_target(raw_seq, smooth_kernel=0):
    """
    raw_seq: (T,33,3)
    Return target height corrected (T,33,3)
    """

    seq = raw_seq.copy()

    # BlazePose foot joints
    LEFT_FOOT = 31
    RIGHT_FOOT = 32

    # 모든 프레임에서 발의 y값 중 최소값 → ground level
    foot_y = np.minimum(seq[:, LEFT_FOOT, 1], seq[:, RIGHT_FOOT, 1])

    # 선택: 발 y가 튀는(노이즈) 경우를 줄이기 위한 median smoothing
    # smooth_kernel은 홀수 권장 (예: 5,7,9). 0이면 미사용.
    if smooth_kernel and smooth_kernel >= 3:
        k = int(smooth_kernel)
        if k % 2 == 0:
            k += 1
        foot_y = medfilt(foot_y, kernel_size=k).astype(np.float32)
    
    # reshape to (T,1) → broadcast to (T,33)
    frame_ground = foot_y[:, None]

    # 전체 시퀀스를 ground=0으로 맞춤
    seq[:, :, 1] -= frame_ground

    return seq


def temporal_smooth_sequence(seq, method="savgol", window=9, poly=2):
    """
    seq: (T,33,3)
    시간축 노이즈(프레임 간 jitter) 감소용 스무딩.
    - method: "savgol" 또는 "moving_avg"
    """
    if seq.shape[0] == 0:
        return seq

    T = seq.shape[0]
    w = int(window)
    if w < 3:
        return seq
    if w % 2 == 0:
        w += 1
    if w > T:
        # 너무 짧은 시퀀스는 window를 줄임(홀수 유지)
        w = T if (T % 2 == 1) else max(1, T - 1)
    if w < 3:
        return seq

    out = seq.copy().astype(np.float32)

    if method == "moving_avg":
        # 간단한 이동평균(가장자리 padding)
        pad = w // 2
        padded = np.pad(out, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        kernel = np.ones((w,), dtype=np.float32) / float(w)
        for j in range(33):
            for c in range(3):
                out[:, j, c] = np.convolve(padded[:, j, c], kernel, mode="valid").astype(np.float32)
        return out

    # 기본: Savitzky-Golay (부드럽게 + 형태 유지)
    p = int(poly)
    if p >= w:
        p = max(1, w - 2)
    for j in range(33):
        for c in range(3):
            out[:, j, c] = savgol_filter(out[:, j, c], window_length=w, polyorder=p, axis=0, mode="interp").astype(np.float32)
    return out


def main():
    paths = glob(f"{RAW_KEYPOINTS_DIR}/*.npz")
    print(f"Found {len(paths)} raw keypoint files")

    for path in paths:
        base = os.path.basename(path).replace(".npz", "")

        # -------------------------------
        # 1) Load raw BlazePose sequence
        # -------------------------------
        raw_seq = load_raw_keypoints(path)      # (T,33,3)
        raw_out_path = os.path.join(OUT_DIR, f"{base}_raw.npy")
        np.save(raw_out_path, raw_seq)
        print(f"Saved raw → {raw_out_path}")

        # ---------------------------------------
        # 2) Create height-corrected target (GT)
        # ---------------------------------------
        # 2-1) 지면 보정(발 기반 ground offset)
        # smooth_kernel을 5~9 정도로 주면 ground jitter를 줄이는 데 도움이 되는 경우가 많음
        target_seq = create_height_corrected_target(raw_seq, smooth_kernel=7)

        # 2-2) 시간축 스무딩(전반적 포즈 노이즈 완화)
        target_seq = temporal_smooth_sequence(target_seq, method="savgol", window=9, poly=2)
        target_out_path = os.path.join(OUT_DIR, f"{base}_target.npy")
        np.save(target_out_path, target_seq)
        print(f"Saved target → {target_out_path}")

    print("\nDone processing all keypoint files.")

    # IF you want to visualize the output
    # vis = PoseViser(fps=30)
    # raw = np.load("./data/processed/Dimitrov_Slow_Motion_Forehand_target.npy")
    # vis.play_sequence(raw)


if __name__ == "__main__":
    main()
