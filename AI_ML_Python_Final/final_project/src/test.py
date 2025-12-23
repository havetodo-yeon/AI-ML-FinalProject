import numpy as np
import tensorflow as tf
import os
import glob
import pandas as pd
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.viser_test import PoseViser
from src.postprocess import (
    apply_grounding, 
    enforce_bone_lengths, 
    constrain_joint_angles,
    constrain_ankle_rotation,
    interpolate_outliers
)

# 최근 학습된 모델을 기본값으로 사용
MODEL_PATH = os.path.join(ROOT, "experiments", "mlp_mix_mish_mish_relu_mish_relu")
# 다른 모델 옵션:
# - mlp_tanh
# - mlp_mix_tanh_tanh_relu
# - height_mlp_model

def load_model(model_path):
    """모델을 로드합니다. 여러 방법을 시도합니다."""
    try:
        # 방법 1: 기본 로딩
        return tf.keras.models.load_model(model_path)
    except Exception as e1:
        print(f"⚠️ 기본 로딩 실패: {e1}")
        try:
            # 방법 2: compile=False로 로딩 (메타데이터 문제 우회)
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e2:
            print(f"⚠️ compile=False 로딩 실패: {e2}")
            try:
                # 방법 3: SavedModel 직접 로딩
                return tf.saved_model.load(model_path)
            except Exception as e3:
                print(f"⚠️ SavedModel 로딩 실패: {e3}")
                # 사용 가능한 모델 목록 출력
                experiments_dir = os.path.join(ROOT, "experiments")
                if os.path.exists(experiments_dir):
                    available_models = [d for d in os.listdir(experiments_dir) 
                                      if os.path.isdir(os.path.join(experiments_dir, d))]
                    print(f"\n📁 사용 가능한 모델 목록:")
                    for model_name in available_models:
                        print(f"  - {model_name}")
                raise IOError(f"모델을 로드할 수 없습니다: {model_path}\n"
                            f"다른 모델을 시도해보세요: --model_path experiments/mlp_tanh")

def _make_windows(seq_2d, window=9, flatten=True):
    T, D = seq_2d.shape
    w = int(window)
    if w < 1:
        raise ValueError("window must be >= 1")
    if w == 1:
        return seq_2d.astype(np.float32)
    pad = w // 2
    padded = np.pad(seq_2d, ((pad, pad), (0, 0)), mode="edge")
    X = np.empty((T, w * D), dtype=np.float32)
    for t in range(T):
        X[t] = padded[t : t + w].reshape(-1)
    return X

def refine_sequence(raw_path, model, out_path=None):

    raw = np.load(raw_path)    # (T,33,3)
    T = raw.shape[0]

    raw2 = raw.reshape(T, -1).astype(np.float32)  # (T,99)
    window = 9

    # 단순 모델: 항상 flatten window 사용
    x = _make_windows(raw2, window=window)

    delta = model.predict(x, verbose=0).astype(np.float32)  # (T,99)

    refined2 = raw2 + delta
    refined = refined2.reshape(T, 33, 3)

    # 후처리 0) 튀는 모션 보간 (가장 먼저 적용하여 비정상적인 변화 제거)
    refined = interpolate_outliers(
        refined, 
        velocity_threshold=3.0,  # 평균 속도의 3배 이상이면 튀는 것으로 간주
        window_size=5,           # 주변 5프레임 사용
        method="linear",         # 선형 보간
        strength=1.0            # 완전 보간
    )

    # 후처리 1) 발 기준으로 재지면화(몸이 뜨는 현상 방지)
    # 발 y값 스무딩 강화 (더 큰 커널 사용)
    refined = apply_grounding(refined, clamp_y=True, smooth_foot_y=True, smooth_kernel=11)

    # 후처리 2) 뼈 길이 유지(팔/다리 말이 안 되는 길이/자세를 강하게 줄임)
    # 기준은 raw (관절 길이를 크게 바꾸지 않도록)
    refined = enforce_bone_lengths(refined, reference_seq=raw, iters=5, strength=1.0)

    # 후처리 3) 관절 각도 제약 (무릎, 팔꿈치가 뒤로 꺾이지 않도록)
    refined = constrain_joint_angles(refined, reference_seq=raw, strength=0.8)

    # 후처리 4) 발목 회전 제약 (발이 비정상적으로 꺾이지 않도록)
    refined = constrain_ankle_rotation(refined, reference_seq=raw, strength=0.9)

    # 후처리 5) 뼈 길이 재확인 (각도 제약 후 길이가 변했을 수 있으므로)
    refined = enforce_bone_lengths(refined, reference_seq=raw, iters=2, strength=0.5)

    # 후처리 6) 최종 튀는 모션 재보간 (다른 후처리 후에도 튀는 부분이 있을 수 있음)
    refined = interpolate_outliers(
        refined,
        velocity_threshold=2.0,  # 조금 더 엄격한 기준
        window_size=7,            # 더 넓은 범위 사용
        method="smooth",          # 부드러운 보간
        strength=1.0             # 부분 보간
    )

    # 최종 안전장치
    refined[:, :, 1] = np.maximum(refined[:, :, 1], 0.0)

    if out_path is not None:
        np.save(out_path, refined)
        print(f"Saved refined → {out_path}")

    return refined

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    model = load_model(args.model_path)

    TEST_DATA_PATH = os.path.join(ROOT, "data", "test_keypoints")
    for raw_path in glob.glob(f"{TEST_DATA_PATH}/*_raw.npy"):
        out_path = raw_path.replace("_raw.npy", "_refined.npy")
        refine_sequence(raw_path, model, out_path)
    
        raw = np.load(raw_path)
        refine = np.load(out_path)
        
        T, nBones,_ = refine.shape
        rows = []
        for t in range(T):
            for b in range(nBones):
                x,y,z = refine[t,b]
                rows.append([t,b,x,y,z,1.0])
        
        df = pd.DataFrame(rows, columns=["frame","landmark","x","y","z","visibility"])
        filename = os.path.basename(raw_path)      # test_raw.npy
        name_only = os.path.splitext(filename)[0]   # test_raw

        TEST_OUTPUT_PATH = os.path.join(ROOT,"data", "output")
        df.to_csv(f"{TEST_OUTPUT_PATH}/reconverted_{name_only}.csv",index=False)
        print("test CSV saved!")

    # visualize in web viewer
    vis = PoseViser(fps=30)
    vis.play_two_sequences(raw,refine,offset=0.0)

    

if __name__ == "__main__":
    main()