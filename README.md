# AI-ML
인공지능과 머신러닝 기말 프로젝트

## Assignment 02
BlazePose + Unity (~11/14)

### 0. 시연

사용한 영상: [이달의 신곡] Hearts2Hearts - Style🎀 포인트 안무 거울모드
https://www.youtube.com/shorts/t2_yCI_ftpI   
시연 영상: [링크 눌러주세요](https://file.notion.so/f/f/b1477d4e-7ebb-4170-aff4-98d41410766e/b6cba4a6-6025-4df7-9bd6-ca2c6bfa7f60/%ED%99%94%EB%A9%B4_%EB%85%B9%ED%99%94_%EC%A4%91_2025-11-12_010140.mp4?table=block&id=2a8ad6eb-216e-8097-ab78-d4bc14d6d1ca&spaceId=b1477d4e-7ebb-4170-aff4-98d41410766e&expirationTimestamp=1764100800000&signature=oUPZyZWznTb8r4D90xn1a3a5ZMWVO1YbtiLdZtuNPok&downloadName=%ED%99%94%EB%A9%B4+%EB%85%B9%ED%99%94+%EC%A4%91+2025-11-12+010140.mp4)**

### 1. 프로젝트 개요

본 프로젝트는 MediaPipe BlazePose를 이용해 인체 3D 관절 데이터를 추출하고, Unity 환경에서 Mixamo 캐릭터에 해당 동작을 재현하는 시스템을 구축하는 것을 목표로 한다.

이를 통해 2D 영상 기반으로 포즈 인식 및 3D 애니메이션 리타게팅이 가능한 기초 프레임워크를 구현하였다.

### 2. 개발 환경

| 구분 | 사용 기술 |
| --- | --- |
| OS / 언어 | Windows 10 / Python 3.9, C# |
| AI 프레임워크 | MediaPipe BlazePose (model_complexity=2) |
| 시각화 | OpenCV, Matplotlib (Python), Unity (C#) |
| 3D 모델링 | Mixamo Humanoid Rig |
| 엔진 | Unity 2022.3 LTS |
| 기타 | Anaconda / VS Code / CSV 기반 데이터 인터페이스 |

### 3. 구현 단계

### 3.1 BlazePose 데이터 추출 (Python)

- **입력:** YouTube에서 직접 선택한 전신 동작 영상
- **프로세스:**
    1. BlazePose로 3D 관절(33개) 추출
    2. 각 landmark의 `(x, y, z, visibility)` 저장
    3. `pose_world_landmarks` 좌표계를 사용하여 실제 3D 비율 유지
    4. CSV 파일(`pose3d_data_*.csv`)로 내보내기
    5. ground 기준 정규화(`y` 반전, 발 높이 0 기준화)
- **BlazePose 설정(기존과 동일):**
    
    ```python
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    ```
    

### 3.2 Unity 리타게팅 파이프라인 (C#)

**기존 제공된 코드 핵심 파일**

| 파일명 | 역할 | 핵심 기능 |
| --- | --- | --- |
| **BlazePoseDataFile.cs** | CSV 로더 | `ImportCSVData()`에서 33개 관절 × N프레임의 좌표를 `frameDict`에 저장 |
| **BlazePoseSkeletonBuilder.cs** | 스켈레톤 빌더 | 33개의 joint를 Unity Actor 본 구조로 생성하고 `boneMap` 생성 |
| **BlazeDataExtraction.cs** | 메인 실행 컴포넌트 | `Feed()`에서 CSV 프레임 데이터를 불러와 Actor 본의 위치 업데이트 / 캡슐 시각화 / 스켈레톤 렌더링 |

**수정 및 디벨롭**

- **BlazePoseDataFile.cs**
    - CSV를 불러와 `frameDict[frame][jointIndex] = Vector3` 형태로 저장
    - 포즈 데이터를 Unity 좌표계로 변환
- **BlazePoseSkeletonBuilder.cs**
    - BlazePose의 33개 관절 인덱스에 대응하는 Humanoid 본 자동 매핑 구현
    - `BuildHumanoidMapping()`을 통해 Mixamo Rig의 본과 자동 연결
    - `directionChildMap`을 이용해 각 본의 “회전 방향 관계(부모→자식)” 정의
- **BlazeDataExtraction.cs**
    - CSV 데이터를 프레임 단위로 불러와 Mixamo 캐릭터의 본에 반영
    - `UpdateBlazePose()` 내에서 회전 기반 리타게팅 수행:
        - `Quaternion.FromToRotation(restDir, targetDir)` 계산
        - `Transform.rotation = delta * restRot` 형태로 적용
    - 상체/머리/골반 회전 보정 코드 추가:
        - Chest 회전 → 어깨/엉덩이 중심 벡터 기반
        - Head / Neck 회전 → 목 = 어깨 중심, 머리 = 코(Nose) 방향 기반
        - Hips 회전 → 상체 회전에 동기화
    - ~~시각화를 위한 캡슐 본 생성(`UpdateCapsules`)~~

### **4. 리깅 및 리타게팅 과정 정리**

1. 데이터 불러오기: CSV를 Unity 내에서 BlazePoseDataFile.cs로 로드
2. 스켈레톤 매핑: BlazePose 33개 관절 → Unity Humanoid 본 연결
3. 회전 계산: 부모-자식 방향 벡터로 회전 (Quaternion) 계산
4. 회전 리타게팅 적용: Mixamo 캐릭터의 Transform.rotation 갱신
5. 보정: 골반(Hips), 상체(Chest), 머리(Head) 방향 보정 적용

### 5. 확장 기능 정리

| 항목 | 설명 |
| --- | --- |
| **1. Humanoid 자동 매핑** | MediaPipe 33개 관절과 Unity Humanoid 본 자동 연결 기능 구현 |
| **2. 회전 기반 리타게팅** | 위치 기반 대신 방향 벡터를 이용한 `Quaternion` 회전 매핑 |
| **3. 상체/머리/골반 보정** | Chest–Head–Hips 회전 보정으로 상체 꼬임 최소화 |
| **~~4. 캡슐 시각화~~** | ~~본 간 연결 구조를 Unity 내에서 실시간 확인 가능~~ |

---

### 6. 추가 응용 가능성

BlazePose로 추출한 CSV 기반 3D 모션 데이터는 Unity 외에도 Unreal Engine 등의 다른 3D 엔진에서도 재활용 가능하다.

ControlRig 또는 Python Script를 통해 동일한 CSV 데이터를 불러와 Skeletal Mesh에 애니메이션을 적용할 수 있으며, 이를 통해 고품질 렌더링 기반의 시네마틱 영상 제작으로 확장할 수 있다.

### 7. 결론

본 프로젝트를 통해 2D 영상 기반 인체 동작을 3D 캐릭터 애니메이션으로 자동 변환하는 파이프라인을 완성하였다.

특히, BlazePose의 관절 데이터를 Unity Mixamo 리그에 자동으로 매핑하고 회전 기반 리타게팅을 구현함으로써, 단순 위치 전송이 아닌 실제 애니메이션 수준의 재현이 가능함을 확인했다.
