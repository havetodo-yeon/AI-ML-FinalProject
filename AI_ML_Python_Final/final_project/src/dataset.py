import os
import numpy as np
from glob import glob
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.viser_test import PoseViser

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

# PROCESSED_DIR = "./data/processed"

def _make_windows(seq_2d, window=9):
    """
    seq_2d: (T,D)
    return X: (T, window*D) (항상 flatten)
    """
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
        chunk = padded[t : t + w]  # (w,D)
        X[t] = chunk.reshape(-1)
    return X


def load_dataset(window=9, val_ratio=0.2, seed=42):
    """
    Temporal-MLP 학습용 데이터 로더.
    - 입력 X: centered window (T, window*99)
    - 정답 Y: delta (target_center - raw_center) (T,99)
    - split은 '파일 단위'로 나눔 (같은 영상의 프레임이 train/val에 섞이지 않게)
    """
    raw_paths = sorted(glob(f"{PROCESSED_DIR}/*_raw.npy"))

    pairs = []
    for rp in raw_paths:
        tp = rp.replace("_raw.npy", "_target.npy")
        if not os.path.exists(tp):
            print("Skip (no target):", rp)
            continue
        pairs.append((rp, tp))

    if len(pairs) == 0:
        raise RuntimeError(f"No dataset pairs found in {PROCESSED_DIR}. Run scripts/02_process_height_dataset.py first.")

    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)

    n_val = max(1, int(len(pairs) * float(val_ratio))) if len(pairs) > 1 else 0
    val_set = set(idx[:n_val].tolist())

    X_train_all, Y_train_all = [], []
    X_val_all, Y_val_all = [], []

    for i, (rp, tp) in enumerate(pairs):
        raw = np.load(rp).astype(np.float32)  # (T,33,3)
        tgt = np.load(tp).astype(np.float32)  # (T,33,3)

        # (T,33,3) -> (T,99)
        raw2 = raw.reshape(raw.shape[0], -1)
        tgt2 = tgt.reshape(tgt.shape[0], -1)

        Xw = _make_windows(raw2, window=window)  # (T, window*99)
        Yd = (tgt2 - raw2).astype(np.float32)                # (T, 99)  (delta learning)

        if i in val_set:
            X_val_all.append(Xw)
            Y_val_all.append(Yd)
        else:
            X_train_all.append(Xw)
            Y_train_all.append(Yd)

    X_train = np.concatenate(X_train_all, axis=0)
    Y_train = np.concatenate(Y_train_all, axis=0)

    if len(X_val_all) > 0:
        X_val = np.concatenate(X_val_all, axis=0)
        Y_val = np.concatenate(Y_val_all, axis=0)
    else:
        X_val = None
        Y_val = None

    print(f"Loaded dataset (window={window})")
    print(f"  train: X={X_train.shape}, Y={Y_train.shape}")
    if X_val is not None:
        print(f"  val:   X={X_val.shape}, Y={Y_val.shape}")
    else:
        print("  val:   (none)")

    return X_train, Y_train, X_val, Y_val
