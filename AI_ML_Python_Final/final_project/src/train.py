import os
import sys
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import load_dataset
from src.build_model import build_model

# MediaPipe Pose indices for weighted loss
# 중요 관절: 발(27-32), 무릎(25-26), 엉덩이(23-24), 어깨(11-12)
IMPORTANT_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # 어깨, 엉덩이, 무릎, 발목, 발꿈치, 발

def create_joint_weights(output_dim=99, important_joints=None, important_weight=2.0):
    """
    관절별 가중치 생성 (중요 관절에 더 높은 가중치)
    output_dim: 출력 차원 (33 joints * 3 coords = 99)
    """
    if important_joints is None:
        important_joints = IMPORTANT_JOINTS
    
    weights = np.ones(output_dim, dtype=np.float32)
    # 각 관절은 3개 좌표 (x, y, z)
    for joint_idx in important_joints:
        base_idx = joint_idx * 3
        if base_idx + 2 < output_dim:
            weights[base_idx:base_idx+3] = important_weight
    
    return weights

def huber_loss(y_true, y_pred, delta=0.1):
    """Huber Loss: 이상치에 robust한 손실 함수"""
    error = y_true - y_pred
    is_small_error = tf.abs(error) < delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small_error, squared_loss, linear_loss)

def weighted_huber_loss(y_true, y_pred, weights, delta=0.1):
    """Weighted Huber Loss"""
    loss = huber_loss(y_true, y_pred, delta)
    return tf.reduce_mean(loss * weights)

EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")

# =========================
# 여기만 바꿔서 실험하면 됨
# =========================
WINDOW = 9
EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3

# "디버깅"이 아니라 결과를 더 좋게 만드는 용도(베스트 시점 가중치로 자동 복원)
USE_EARLY_STOPPING = True

# 성능 개선 옵션
USE_HUBER_LOSS = True          # Huber Loss 사용 (이상치에 robust)
HUBER_DELTA = 0.1             # Huber Loss delta 값
USE_WEIGHTED_LOSS = True       # 중요 관절에 더 높은 가중치
USE_COSINE_DECAY = True        # Cosine Decay 학습률 스케줄링
USE_GRADIENT_CLIPPING = True   # Gradient Clipping (학습 안정성)
CLIP_NORM = 1.0                # Gradient clipping norm

# 한 번 실행하면 아래 여러 모델을 순서대로 학습 & 저장
MODEL_CONFIGS = [
    {
        "name": "mlp_mix_mish_mish_relu_mish_relu",
        "hidden_units": (1024, 512, 256, 256, 128),
        "activation": ("mish", "mish", "relu", "mish", "relu"),
        "dropout": 0.0,
        "use_layernorm": True,
        "use_residual": False,  # Residual connection 사용 여부
    },
    {
        "name": "mlp_mix_tanh_relu_tanh",
        "hidden_units": (512, 256, 128),
        "activation": ("tanh", "relu", "tanh"),
        "dropout": 0.0,
        "use_layernorm": True,
        "use_residual": False,
    },
    {
        "name": "mlp_mix_tanh_tanh_relu_tanh",
        "hidden_units": (512, 512, 256, 128),
        "activation": ("tanh", "tanh", "relu", "tanh"),
        "dropout": 0.0,
        "use_layernorm": True,
        "use_residual": False,
    },
    # {
    #     "name": "mlp_tanh",
    #     "hidden_units": (512, 512, 256),
    #     "activation": "tanh",
    #     "dropout": 0.0,
    #     "use_layernorm": True,
    # },
    # {
    #     "name": "mlp_mix_tanh_tanh_relu",
    #     "hidden_units": (512, 256, 128),
    #     "activation": ("tanh", "tanh", "relu"),
    #     "dropout": 0.0,
    #     "use_layernorm": True,
    # },
]

def train():
    # 1) 데이터 로딩 (temporal windows)
    X_train, Y_train, X_val, Y_val = load_dataset(window=WINDOW, val_ratio=0.2, seed=42)

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        model_path = os.path.join(EXPERIMENTS_DIR, name)

        print("\n" + "=" * 60)
        print(f"Training: {name}")
        print(f"Save to : {model_path}")
        print(f"hidden_units={cfg['hidden_units']}, activation={cfg['activation']}, dropout={cfg['dropout']}, ln={cfg['use_layernorm']}")
        print("=" * 60)

        # 2) 모델
        use_residual = cfg.get("use_residual", False)
        model = build_model(
            input_dim=X_train.shape[1],
            output_dim=Y_train.shape[1],
            hidden_units=cfg["hidden_units"],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
            use_layernorm=cfg["use_layernorm"],
            use_residual=use_residual,
        )
        
        # 손실 함수 설정
        if USE_WEIGHTED_LOSS and USE_HUBER_LOSS:
            joint_weights = create_joint_weights(
                output_dim=Y_train.shape[1],
                important_weight=2.0
            )
            loss_fn = lambda y_true, y_pred: weighted_huber_loss(
                y_true, y_pred, joint_weights, delta=HUBER_DELTA
            )
            print(f"Using Weighted Huber Loss (delta={HUBER_DELTA})")
        elif USE_HUBER_LOSS:
            loss_fn = lambda y_true, y_pred: tf.reduce_mean(huber_loss(y_true, y_pred, delta=HUBER_DELTA))
            print(f"Using Huber Loss (delta={HUBER_DELTA})")
        elif USE_WEIGHTED_LOSS:
            joint_weights = create_joint_weights(
                output_dim=Y_train.shape[1],
                important_weight=2.0
            )
            loss_fn = lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred) * joint_weights)
            print("Using Weighted MSE Loss")
        else:
            loss_fn = "mse"
            print("Using MSE Loss")
        
        # Optimizer 설정 (Gradient Clipping 및 Cosine Decay 포함)
        if USE_COSINE_DECAY:
            # Cosine Decay: 더 부드러운 학습률 감소
            total_steps = EPOCHS * (len(X_train) // BATCH_SIZE + 1)
            cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=LR,
                first_decay_steps=total_steps // 4,
                t_mul=2.0,
                m_mul=0.5,
                alpha=1e-6
            )
            lr_schedule = cosine_decay
            print("Using Cosine Decay learning rate schedule")
        else:
            lr_schedule = LR
        
        if USE_GRADIENT_CLIPPING:
            optimizer = tf.keras.optimizers.Adam(lr_schedule, clipnorm=CLIP_NORM)
            print(f"Using Gradient Clipping (norm={CLIP_NORM})")
        else:
            optimizer = tf.keras.optimizers.Adam(lr_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["mae"]
        )

        callbacks = []
        if USE_EARLY_STOPPING and X_val is not None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    restore_best_weights=True
                ),
            ]
            
            # 학습률 스케줄링: Cosine Decay가 아닐 때만 ReduceLROnPlateau 사용
            if not USE_COSINE_DECAY:
                callbacks.append(
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=3,
                        min_lr=1e-6
                    )
                )
                print("Using ReduceLROnPlateau learning rate schedule")

        # 3) 학습
        if X_val is not None:
            history = model.fit(
                X_train, Y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(X_val, Y_val),
                shuffle=True,
                callbacks=callbacks
            )
        else:
            history = model.fit(
                X_train, Y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                shuffle=True
            )

        # 4) 저장
        model.save(model_path)
        print(f"Saved → {model_path}")

if __name__ == "__main__":
    train()
