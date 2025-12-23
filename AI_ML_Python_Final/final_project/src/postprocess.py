import numpy as np
from scipy.signal import medfilt

# MediaPipe Pose indices
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT = 31
RIGHT_FOOT = 32

# Upper body
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


DEFAULT_BONES = [
    # torso / hips
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_SHOULDER),
    (RIGHT_HIP, RIGHT_SHOULDER),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    # arms
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    # legs
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    # feet chain (ankle -> heel/foot)
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_HEEL, LEFT_FOOT),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_HEEL, RIGHT_FOOT),
]


def apply_grounding(seq, left_foot=LEFT_FOOT, right_foot=RIGHT_FOOT, clamp_y=True, smooth_foot_y=True, smooth_kernel=7):
    """
    seq: (T,33,3)
    각 프레임에서 발의 최소 y를 0으로 맞추어 "몸이 뜨는" 현상을 강하게 줄임.
    
    Args:
        smooth_foot_y: True면 발 y값을 시간축으로 스무딩 (프레임 간 급격한 변화로 인한 "뜨는" 현상 방지)
        smooth_kernel: median filter 커널 크기 (홀수, 기본 7)
    """
    if seq.shape[0] == 0:
        return seq
    out = seq.copy().astype(np.float32)
    
    # 발의 최소 y값 (각 프레임)
    foot_y = np.minimum(out[:, left_foot, 1], out[:, right_foot, 1])  # (T,)
    
    # 발 y값을 시간축으로 스무딩 (프레임 간 급격한 변화를 줄여 "뜨는" 현상 방지)
    if smooth_foot_y and seq.shape[0] > 1:
        kernel = int(smooth_kernel)
        if kernel > 1 and kernel % 2 == 1:
            # median filter로 발 y값 스무딩
            foot_y = medfilt(foot_y, kernel_size=kernel).astype(np.float32)
    
    # 스무딩된 발 y값을 기준으로 지면 보정
    out[:, :, 1] -= foot_y[:, None]
    
    if clamp_y:
        out[:, :, 1] = np.maximum(out[:, :, 1], 0.0)
    return out


def enforce_bone_lengths(seq, reference_seq, bones=DEFAULT_BONES, iters=2, strength=1.0, eps=1e-8):
    """
    seq: (T,33,3) - 수정 대상
    reference_seq: (T,33,3) - 기준(보통 raw). 프레임별 bone length를 기준으로 삼음.
    iters: 반복 횟수(2~5 권장)
    strength: 1.0이면 완전 맞추기, 0.5면 절반만 교정
    """
    if seq.shape[0] == 0:
        return seq
    out = seq.copy().astype(np.float32)
    ref = reference_seq.astype(np.float32)

    # 기준 bone length (T, nbones)
    ref_len = []
    for (a, b) in bones:
        d = ref[:, b, :] - ref[:, a, :]
        ref_len.append(np.sqrt(np.sum(d * d, axis=-1) + eps))
    ref_len = np.stack(ref_len, axis=1)

    s = float(strength)
    s = max(0.0, min(1.0, s))

    for _ in range(int(iters)):
        for bi, (a, b) in enumerate(bones):
            a_pos = out[:, a, :]
            b_pos = out[:, b, :]
            d = b_pos - a_pos
            cur = np.sqrt(np.sum(d * d, axis=-1) + eps)  # (T,)
            target = ref_len[:, bi]                       # (T,)
            # 방향 유지하며 길이만 조정
            scale = (target / cur).astype(np.float32)
            new_b = a_pos + d * scale[:, None]
            out[:, b, :] = (1 - s) * b_pos + s * new_b

    return out


def constrain_joint_angles(seq, reference_seq=None, strength=0.8):
    """
    관절 각도를 제약하여 비정상적인 자세를 방지합니다.
    
    Args:
        seq: (T,33,3) - 수정 대상
        reference_seq: (T,33,3) - 기준 시퀀스 (None이면 seq 자체를 기준으로)
        strength: 제약 강도 (0.0~1.0, 높을수록 강하게 제약)
    
    제약 사항:
    - 무릎이 뒤로 꺾이지 않도록 (다리 앞쪽으로만 구부러지도록)
    - 팔꿈치가 뒤로 꺾이지 않도록
    - 발목이 비정상적으로 꺾이지 않도록
    """
    if seq.shape[0] == 0:
        return seq
    out = seq.copy().astype(np.float32)
    ref = reference_seq.astype(np.float32) if reference_seq is not None else out.copy()
    
    s = float(strength)
    s = max(0.0, min(1.0, s))
    
    # 무릎 각도 제약 (무릎이 뒤로 꺾이지 않도록)
    for side in ['left', 'right']:
        hip_idx = LEFT_HIP if side == 'left' else RIGHT_HIP
        knee_idx = LEFT_KNEE if side == 'left' else RIGHT_KNEE
        ankle_idx = LEFT_ANKLE if side == 'left' else RIGHT_ANKLE
        
        # 다리 방향 벡터
        thigh = out[:, knee_idx, :] - out[:, hip_idx, :]  # (T,3)
        shin = out[:, ankle_idx, :] - out[:, knee_idx, :]  # (T,3)
        
        # 정규화
        thigh_norm = np.linalg.norm(thigh, axis=-1, keepdims=True) + 1e-8
        shin_norm = np.linalg.norm(shin, axis=-1, keepdims=True) + 1e-8
        thigh_unit = thigh / thigh_norm
        shin_unit = shin / shin_norm
        
        # 내적 (각도 계산)
        dot = np.sum(thigh_unit * shin_unit, axis=-1)  # (T,)
        
        # 무릎이 뒤로 꺾이면 (각도가 180도에 가까우면) 제약
        # 정상적인 무릎 각도는 보통 90~170도 정도
        # dot < 0.5면 뒤로 꺾인 것으로 간주
        mask = dot < 0.5  # 뒤로 꺾인 프레임
        
        if mask.any():
            # 기준 시퀀스의 각도로 보정
            ref_thigh = ref[:, knee_idx, :] - ref[:, hip_idx, :]
            ref_shin = ref[:, ankle_idx, :] - ref[:, knee_idx, :]
            ref_thigh_norm = np.linalg.norm(ref_thigh, axis=-1, keepdims=True) + 1e-8
            ref_shin_norm = np.linalg.norm(ref_shin, axis=-1, keepdims=True) + 1e-8
            ref_thigh_unit = ref_thigh / ref_thigh_norm
            ref_shin_unit = ref_shin / ref_shin_norm
            
            # 기준 시퀀스의 각도로 보정 (더 간단한 방법)
            ref_dot = np.sum(ref_thigh_unit * ref_shin_unit, axis=-1, keepdims=True)
            target_dot = np.maximum(ref_dot, 0.5)  # 최소 0.5 (약 60도)
            
            # 현재 각도와 목표 각도의 차이
            dot_diff = target_dot - dot[:, None]
            
            # 발목 위치를 기준 시퀀스에 가깝게 조정
            ref_shin_dir = ref_shin_unit
            current_shin_dir = shin_unit
            
            # 각도 차이만큼 보정
            corrected_shin_dir = current_shin_dir + (ref_shin_dir - current_shin_dir) * np.clip(dot_diff, 0, 1)
            corrected_shin_dir = corrected_shin_dir / (np.linalg.norm(corrected_shin_dir, axis=-1, keepdims=True) + 1e-8)
            
            # 발목 위치 재조정
            shin_len = shin_norm.squeeze()
            new_ankle = out[:, knee_idx, :] + corrected_shin_dir * shin_len[:, None]
            
            # 제약 강도만큼만 적용
            out[mask, ankle_idx, :] = (1 - s) * out[mask, ankle_idx, :] + s * new_ankle[mask, :]
    
    # 팔꿈치 각도 제약
    for side in ['left', 'right']:
        shoulder_idx = LEFT_SHOULDER if side == 'left' else RIGHT_SHOULDER
        elbow_idx = LEFT_ELBOW if side == 'left' else RIGHT_ELBOW
        wrist_idx = LEFT_WRIST if side == 'left' else RIGHT_WRIST
        
        upper_arm = out[:, elbow_idx, :] - out[:, shoulder_idx, :]
        forearm = out[:, wrist_idx, :] - out[:, elbow_idx, :]
        
        upper_arm_norm = np.linalg.norm(upper_arm, axis=-1, keepdims=True) + 1e-8
        forearm_norm = np.linalg.norm(forearm, axis=-1, keepdims=True) + 1e-8
        upper_arm_unit = upper_arm / upper_arm_norm
        forearm_unit = forearm / forearm_norm
        
        dot = np.sum(upper_arm_unit * forearm_unit, axis=-1)
        mask = dot < 0.0  # 뒤로 꺾인 경우
        
        if mask.any():
            ref_upper_arm = ref[:, elbow_idx, :] - ref[:, shoulder_idx, :]
            ref_forearm = ref[:, wrist_idx, :] - ref[:, elbow_idx, :]
            ref_upper_arm_norm = np.linalg.norm(ref_upper_arm, axis=-1, keepdims=True) + 1e-8
            ref_forearm_norm = np.linalg.norm(ref_forearm, axis=-1, keepdims=True) + 1e-8
            ref_upper_arm_unit = ref_upper_arm / ref_upper_arm_norm
            ref_forearm_unit = ref_forearm / ref_forearm_norm
            
            ref_dot = np.sum(ref_upper_arm_unit * ref_forearm_unit, axis=-1, keepdims=True)
            target_dot = np.maximum(ref_dot, 0.0)  # 최소 0 (직선)
            
            # 현재 각도와 목표 각도의 차이
            dot_diff = target_dot - dot[:, None]
            
            # 손목 위치를 기준 시퀀스에 가깝게 조정
            ref_forearm_dir = ref_forearm_unit
            current_forearm_dir = forearm_unit
            
            # 각도 차이만큼 보정
            corrected_forearm_dir = current_forearm_dir + (ref_forearm_dir - current_forearm_dir) * np.clip(dot_diff, 0, 1)
            corrected_forearm_dir = corrected_forearm_dir / (np.linalg.norm(corrected_forearm_dir, axis=-1, keepdims=True) + 1e-8)
            
            forearm_len = forearm_norm.squeeze()
            new_wrist = out[:, elbow_idx, :] + corrected_forearm_dir * forearm_len[:, None]
            
            out[mask, wrist_idx, :] = (1 - s) * out[mask, wrist_idx, :] + s * new_wrist[mask, :]
    
    return out


def constrain_ankle_rotation(seq, reference_seq=None, strength=0.9):
    """
    발목 회전을 제약하여 발이 비정상적으로 꺾이지 않도록 합니다.
    
    Args:
        seq: (T,33,3) - 수정 대상
        reference_seq: (T,33,3) - 기준 시퀀스
        strength: 제약 강도 (0.0~1.0)
    """
    if seq.shape[0] == 0:
        return seq
    out = seq.copy().astype(np.float32)
    ref = reference_seq.astype(np.float32) if reference_seq is not None else out.copy()
    
    s = float(strength)
    s = max(0.0, min(1.0, s))
    
    for side in ['left', 'right']:
        ankle_idx = LEFT_ANKLE if side == 'left' else RIGHT_ANKLE
        heel_idx = LEFT_HEEL if side == 'left' else RIGHT_HEEL
        foot_idx = LEFT_FOOT if side == 'left' else RIGHT_FOOT
        
        # 발목-발꿈치-발끝 벡터
        ankle_heel = out[:, heel_idx, :] - out[:, ankle_idx, :]  # (T,3)
        heel_foot = out[:, foot_idx, :] - out[:, heel_idx, :]  # (T,3)
        
        # 기준 각도
        ref_ankle_heel = ref[:, heel_idx, :] - ref[:, ankle_idx, :]
        ref_heel_foot = ref[:, foot_idx, :] - ref[:, heel_idx, :]
        
        # 각도 차이 계산
        ankle_heel_norm = np.linalg.norm(ankle_heel, axis=-1, keepdims=True) + 1e-8
        heel_foot_norm = np.linalg.norm(heel_foot, axis=-1, keepdims=True) + 1e-8
        ref_ankle_heel_norm = np.linalg.norm(ref_ankle_heel, axis=-1, keepdims=True) + 1e-8
        ref_heel_foot_norm = np.linalg.norm(ref_heel_foot, axis=-1, keepdims=True) + 1e-8
        
        ankle_heel_unit = ankle_heel / ankle_heel_norm
        heel_foot_unit = heel_foot / heel_foot_norm
        ref_ankle_heel_unit = ref_ankle_heel / ref_ankle_heel_norm
        ref_heel_foot_unit = ref_heel_foot / ref_heel_foot_norm
        
        # 각도 차이
        dot_diff = np.sum(ankle_heel_unit * heel_foot_unit, axis=-1) - np.sum(ref_ankle_heel_unit * ref_heel_foot_unit, axis=-1)
        
        # 각도 차이가 크면 보정
        mask = np.abs(dot_diff) > 0.3  # 각도 차이가 30도 이상
        
        if mask.any():
            # 기준 각도로 보정 (더 간단한 방법)
            ref_angle = np.sum(ref_ankle_heel_unit * ref_heel_foot_unit, axis=-1, keepdims=True)
            target_angle = np.clip(ref_angle, -0.5, 0.9)  # 정상 범위로 제한
            
            # 현재 각도와 목표 각도의 차이
            current_angle = np.sum(ankle_heel_unit * heel_foot_unit, axis=-1, keepdims=True)
            angle_diff = target_angle - current_angle
            
            # 발끝 위치를 기준 시퀀스에 가깝게 조정
            ref_heel_foot_dir = ref_heel_foot_unit
            current_heel_foot_dir = heel_foot_unit
            
            # 각도 차이만큼 보정
            corrected_foot_dir = current_heel_foot_dir + (ref_heel_foot_dir - current_heel_foot_dir) * np.clip(np.abs(angle_diff), 0, 1)
            corrected_foot_dir = corrected_foot_dir / (np.linalg.norm(corrected_foot_dir, axis=-1, keepdims=True) + 1e-8)
            
            heel_foot_len = heel_foot_norm.squeeze()
            new_foot = out[:, heel_idx, :] + corrected_foot_dir * heel_foot_len[:, None]
            
            out[mask, foot_idx, :] = (1 - s) * out[mask, foot_idx, :] + s * new_foot[mask, :]
    
    return out


def interpolate_outliers(seq, velocity_threshold=3.0, window_size=5, method="linear", strength=1.0):
    """
    튀는 모션(비정상적으로 급격한 변화)을 감지하고 보간합니다.
    
    Args:
        seq: (T,33,3) - 수정 대상 시퀀스
        velocity_threshold: 튀는 것으로 간주할 속도 임계값 (평균 속도의 배수)
        window_size: 보간 시 사용할 주변 프레임 수 (홀수 권장)
        method: 보간 방법 ("linear" 또는 "smooth")
        strength: 보간 강도 (0.0~1.0, 1.0이면 완전 보간)
    
    Returns:
        보간된 시퀀스 (T,33,3)
    """
    if seq.shape[0] < 3:
        return seq
    
    out = seq.copy().astype(np.float32)
    T = seq.shape[0]
    
    # 각 관절의 프레임 간 속도(변화량) 계산
    # velocity[t, j] = ||seq[t+1, j] - seq[t, j]||
    velocities = np.zeros((T-1, 33), dtype=np.float32)
    for t in range(T-1):
        for j in range(33):
            diff = seq[t+1, j, :] - seq[t, j, :]
            velocities[t, j] = np.linalg.norm(diff)
    
    # 전체 평균 속도 계산
    mean_velocity = np.mean(velocities)
    std_velocity = np.std(velocities)
    
    # 튀는 프레임 감지: 평균 + threshold * std 이상의 속도를 가진 프레임
    threshold = mean_velocity + velocity_threshold * std_velocity
    
    # 각 프레임에서 튀는 관절 개수 계산
    outlier_frames = np.zeros(T, dtype=bool)
    outlier_joints = np.zeros((T, 33), dtype=bool)
    
    for t in range(T-1):
        for j in range(33):
            if velocities[t, j] > threshold:
                outlier_joints[t+1, j] = True
                outlier_frames[t+1] = True
    
    # 전체 포즈의 변화량도 체크 (모든 관절의 평균 이동 거리)
    frame_velocities = np.mean(velocities, axis=1)  # (T-1,)
    frame_threshold = np.mean(frame_velocities) + velocity_threshold * np.std(frame_velocities)
    
    for t in range(T-1):
        if frame_velocities[t] > frame_threshold:
            outlier_frames[t+1] = True
            outlier_joints[t+1, :] = True
    
    # 튀는 프레임/관절 보간
    s = float(strength)
    s = max(0.0, min(1.0, s))
    
    w = int(window_size)
    if w % 2 == 0:
        w += 1
    w = min(w, T)
    half_w = w // 2
    
    for t in range(T):
        if not outlier_frames[t]:
            continue
        
        # 튀는 관절만 보간
        for j in range(33):
            if not outlier_joints[t, j]:
                continue
            
            # 주변 프레임 범위 계산
            start_t = max(0, t - half_w)
            end_t = min(T, t + half_w + 1)
            
            # 주변 프레임에서 튀지 않는 관절 위치 수집
            valid_positions = []
            valid_indices = []
            
            for neighbor_t in range(start_t, end_t):
                if neighbor_t == t:
                    continue
                if not outlier_joints[neighbor_t, j]:
                    valid_positions.append(seq[neighbor_t, j, :])
                    valid_indices.append(neighbor_t)
            
            if len(valid_positions) == 0:
                # 주변에 유효한 프레임이 없으면 이전/다음 프레임 사용
                if t > 0:
                    valid_positions.append(seq[t-1, j, :])
                if t < T-1:
                    valid_positions.append(seq[t+1, j, :])
            
            if len(valid_positions) == 0:
                continue
            
            valid_positions = np.array(valid_positions)
            
            if method == "linear" and len(valid_indices) >= 2:
                # 선형 보간: 가장 가까운 이전/다음 프레임 사용
                prev_idx = None
                next_idx = None
                
                for idx in valid_indices:
                    if idx < t and (prev_idx is None or idx > prev_idx):
                        prev_idx = idx
                    if idx > t and (next_idx is None or idx < next_idx):
                        next_idx = idx
                
                if prev_idx is not None and next_idx is not None:
                    # 선형 보간
                    alpha = (t - prev_idx) / (next_idx - prev_idx)
                    interpolated = (1 - alpha) * seq[prev_idx, j, :] + alpha * seq[next_idx, j, :]
                elif prev_idx is not None:
                    interpolated = seq[prev_idx, j, :]
                elif next_idx is not None:
                    interpolated = seq[next_idx, j, :]
                else:
                    interpolated = np.mean(valid_positions, axis=0)
            else:
                # 가중 평균: 가까운 프레임에 더 높은 가중치
                if len(valid_indices) > 0:
                    weights = []
                    for idx in valid_indices:
                        dist = abs(idx - t)
                        weight = 1.0 / (1.0 + dist)
                        weights.append(weight)
                    weights = np.array(weights)
                    weights = weights / (np.sum(weights) + 1e-8)
                    interpolated = np.sum(valid_positions * weights[:, None], axis=0)
                else:
                    interpolated = np.mean(valid_positions, axis=0)
            
            # 보간 강도만큼만 적용
            out[t, j, :] = (1 - s) * out[t, j, :] + s * interpolated
    
    return out


