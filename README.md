# AI-ML-Assignment 02
인공지능과 머신러닝 과제 2 - BlazePose + Unity (~11/14)    

사용한 영상: [[이달의 신곡] Hearts2Hearts - Style🎀 포인트 안무 거울모드](https://www.youtube.com/shorts/t2_yCI_ftpI)   
**시연 영상**:[노션으로 연결되는 링크](https://file.notion.so/f/f/b1477d4e-7ebb-4170-aff4-98d41410766e/b6cba4a6-6025-4df7-9bd6-ca2c6bfa7f60/%ED%99%94%EB%A9%B4_%EB%85%B9%ED%99%94_%EC%A4%91_2025-11-12_010140.mp4?table=block&id=2a8ad6eb-216e-8097-ab78-d4bc14d6d1ca&spaceId=b1477d4e-7ebb-4170-aff4-98d41410766e&expirationTimestamp=1764115200000&signature=r3bSlFyz2j4WjHDlcaeBYqRbB0tvojw55eWQXrbizJE&downloadName=%ED%99%94%EB%A9%B4+%EB%85%B9%ED%99%94+%EC%A4%91+2025-11-12+010140.mp4)

## 1. 프로젝트 개요
BlazePose를 이용해 인체 3D 관절 데이터를 csv로 추출하고, Unity 환경에서 3D Humanoid 모델에 해당 동작을 적용하는 시스템을 구축하였다.    
### 기본 요구사항
- [x] 입력 영상: 예제 영상이 아닌, 학생이 직접 선택하여 다운로드한 YouTube 영상을 사용한다.
- [x] 시연 영상: Unity 게임 엔진 내에서 BlazePose로 추출한 3차원 동작이 캐릭터로 재현되는 장면을 동영상으로 캡처하여 제출한다.    
### 가산점 항목
- [x] 위의 기본 내용을 바탕으로, 기말 프로젝트를 위한 확장 또는 응용 기능을 추가한 경우 자유 형식의 리포트(설명서)를 함께 제출한다.

## 2. 리깅 및 리타게팅 과정 정리
1. 데이터 불러오기: CSV를 Unity 내에서 `BlazePoseDataFile.cs`로 로드
2. 스켈레톤 매핑: BlazePose 33개 관절 → Unity Humanoid 본 연결
3. 회전 계산: 부모-자식 방향 벡터로 회전 (Quaternion) 계산
4. 회전 리타게팅 적용: Mixamo 캐릭터의 `Transform.rotation` 갱신
5. 보정: 골반(Hips), 상체(Chest), 머리(Head) 방향 보정 적용

## 3. 구현 단계

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

**수정 및 개선 사항**

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

## 4. 변경된 Component 사용 방법
1. mixamo와 같은 인간형 캐릭터에 Humanoid 설정
2. 에셋 캐릭터 루트에 Actor 컴포넌트 추가
3. BlazeDataExtraction 컴포넌트 Reset: Capsule List 값 0으로 설정
4. BlazeDataExtraction 컴포넌트 Source Actor에 캐릭터 에셋 설정
5. 이후 csv 불러와서 애니메이션 재생

## 5. 확장 기능 정리

| 항목 | 설명 |
| --- | --- |
| **1. Humanoid 자동 매핑** | MediaPipe 33개 관절과 Unity Humanoid 본 자동 연결 기능 구현 |
| **2. 회전 기반 리타게팅** | 위치 기반 대신 방향 벡터를 이용한 `Quaternion` 회전 매핑 |
| **3. 상체/머리/골반 보정** | Chest–Head–Hips 회전 보정으로 상체 꼬임 최소화 |


## 6. 트러블 슈팅
### 모델이 비틀려서 출력됨
- BlazePose에서 받아온 관절의 절대 위치를 Mixamo 본의 월드 위치에 직접 매핑시킴 → 모델이 BlazePose에 맞춰서 늘어나고 뒤틀림    
### 해결 방법
- 부모 → 자식 관절 방향을 이용해서 각 본의 회전(quaternion)만 조정하는 방식으로 리타게팅 (`BlazeDataExtraction.CaptureRestPose`)
 루트(Spine) 위치를 BlazePose 기준 양쪽 엉덩이의 중간 지점으로 설정 후 상체와 머리 회전 보정 (`BlazeDataExtraction.UpdateBlazePose`)
- 왼팔/오른팔, 왼다리/오른다리가 서로 뒤집힌 문제 → BlazePose 좌표를 Unity로 바꿀 때 X축 반전시켜 최종 해결 (`BlazeDataExtraction.ConvertBlazeToUnity`)

## 7. 결론 및 추가 응용 가능성

본 프로젝트를 통해 2D 영상 기반 인체 동작을 3D 캐릭터 애니메이션으로 자동 변환하는 파이프라인을 완성하였다.

특히, BlazePose의 관절 데이터를 Unity Mixamo 리그에 자동으로 매핑하고 회전 기반 리타게팅을 구현함으로써, 단순 위치 전송이 아닌 실제 애니메이션 수준의 재현이 가능함을 확인했다.

BlazePose로 추출한 CSV 기반 3D 모션 데이터는 Unity 외에도 Unreal Engine 등의 다른 3D 엔진에서도 재활용 가능하다.

ControlRig 또는 Python Script를 통해 동일한 CSV 데이터를 불러와 Skeletal Mesh에 애니메이션을 적용할 수 있으며, 이를 통해 고품질 렌더링 기반의 시네마틱 영상 제작으로 확장할 수 있다.

