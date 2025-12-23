import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(
    input_dim: int,
    output_dim: int = 99,
    hidden_units=(512, 512, 256, 128),
    activation=("tanh", "tanh", "relu", "tanh"),
    dropout: float = 0.0,
    use_layernorm: bool = True,
    use_residual: bool = False,
):
    """
    - input_dim: 입력 벡터 차원 (예: window=9면 9*99=891)
    - hidden_units: Dense 깊이/너비를 여기서만 바꾸면 됨 (예: (256,256,256))
    - activation: 기본 tanh (요청사항)
    - use_residual: Residual connection 사용 여부 (더 깊은 네트워크에 유용)
    - 출력은 delta(=target-raw) (99차원) 를 예측
    """
    inp = layers.Input(shape=(int(input_dim),), name="x")
    x = inp
    if use_layernorm:
        x = layers.LayerNormalization(name="ln")(x)

    # activation:
    # - str: 모든 hidden layer에 동일 activation
    # - list/tuple: hidden layer마다 activation 지정 (예: ["tanh","tanh","relu"])
    if isinstance(activation, (list, tuple)):
        if len(activation) != len(hidden_units):
            raise ValueError("activation 리스트 길이는 hidden_units 길이와 같아야 합니다.")
        activations = list(activation)
    else:
        activations = [activation] * len(hidden_units)

    # Residual connection을 위한 이전 레이어 저장
    prev_x = None
    
    for i, u in enumerate(hidden_units):
        # Residual connection: 같은 차원일 때만 적용 가능
        if use_residual and prev_x is not None:
            # 이전 레이어와 같은 차원이면 skip connection 사용
            if prev_x.shape[-1] == int(u):
                # Skip connection 추가
                dense_out = layers.Dense(int(u), use_bias=False, name=f"dense_{i}_pre")(x)
                if use_layernorm:
                    dense_out = layers.LayerNormalization(name=f"ln_{i}")(dense_out)
                dense_out = layers.Activation(activations[i], name=f"act_{i}")(dense_out)
                x = layers.Add(name=f"add_{i}")([dense_out, prev_x])
            else:
                # 차원이 다르면 일반 레이어 사용
                x = layers.Dense(int(u), activation=activations[i], name=f"dense_{i}")(x)
        else:
            x = layers.Dense(int(u), activation=activations[i], name=f"dense_{i}")(x)
        
        if dropout and dropout > 0:
            x = layers.Dropout(float(dropout), name=f"dropout_{i}")(x)
        
        # Residual connection을 위해 현재 레이어 출력 저장 (dropout 전)
        if use_residual:
            prev_x = x

    out = layers.Dense(int(output_dim), name="delta")(x)
    return models.Model(inputs=inp, outputs=out, name="simple_mlp")
