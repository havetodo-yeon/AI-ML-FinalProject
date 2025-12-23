# AI-ML-FinalProject
### AI-ML Unity Final Project – BlazePose 기반 댄스 애니메이션 생성

본 프로젝트는 **MediaPipe BlazePose로부터 추출한 3D 인체 키포인트 시퀀스**를 기반으로,
머신러닝 모델을 통해 포즈를 정제(refinement)하고 이를 **Unity 애니메이션으로 재구성**하는 파이프라인을 구현한 프로젝트입니다.

특히 단일 프레임 정확도뿐 아니라 **시간적 연속성(temporal smoothness)** 을 고려하여
모델 구조 개선과 후처리(Post-processing)를 함께 설계하였으며,
최종적으로는 **아이돌 댄스 동작을 자연스럽게 재현하는 시네마틱 영상**을 Unity Timeline과 Cinemachine을 활용해 제작하였습니다.

---

## 1. 프로젝트 개요

* **입력**

  * YouTube 영상에서 추출한 BlazePose 3D keypoints (raw keypoints)
* **처리**

  * 키포인트 전처리 및 신체 비율 정규화
  * 머신러닝 기반 포즈 보정(Student 모델)
  * 후처리를 통한 안정화
* **출력**

  * Unity에서 재생 가능한 애니메이션 CSV
  * Timeline 기반 시네마틱 영상

---

## 2. 전체 파이프라인

### 머신러닝 파이프라인 실행 순서

```text
01_create_raw_keypoints.py
→ 02_process_height_dataset.py
→ 03_create_test_keypoints.py
→ train.py
→ test.py
```

### 데이터 흐름 요약

| 단계             | 설명                             |
| -------------- | ------------------------------ |
| Raw Keypoints  | BlazePose로부터 프레임 단위 3D 키포인트 추출 |
| Processed Data | 신체 높이 기반 정규화 및 노이즈 완화          |
| Train          | Student 모델 학습                  |
| Test           | 학습에 사용되지 않은 test 영상으로 추론       |
| Unity          | CSV를 로드하여 애니메이션 재생             |

---

## 3. Baseline 모델과 한계

### Baseline 모델 구조

* 단순 MLP 기반 네트워크
* 프레임 단위 입력 → 프레임 단위 보정값 출력
* 활성화 함수: `tanh_relu_tanh`

### Baseline 모델의 한계

* 프레임 단위 정확도는 확보되지만,
* **연속 프레임 간 불연속성(튐 현상)** 이 빈번
* 발 미끄러짐, 관절 과도 회전, yaw 편향 등 발생
* Unity 애니메이션으로 사용하기에는 안정성이 부족

---

## 4. Student 모델 설계

### 모델 구조

* **MLP 기반 Student 모델**
* 입력: 현재 프레임의 keypoints
* 출력: baseline 대비 보정 delta
* 최종 포즈 = `raw + delta`

### 활성화 함수

* `tanh`
* `ReLU`
* `Mish`

#### Mish 활성화 함수 채택 이유

* 음수 영역에서도 부드러운 기울기 유지
* gradient 흐름이 안정적
* 프레임 간 출력 변화가 상대적으로 완만함

---

## 5. 학습 데이터 설계

### Train / Test 데이터 분리 원칙

* **영상 단위 분리**
* 동일 영상에서 추출된 프레임이 train과 test에 동시에 포함되지 않도록 구성
* `test.py`는 `data/test_keypoints/`만 읽도록 설계

### 학습 영상 선정 기준

* 테스트 데이터(아이돌 댄스 영상)와 **유사한 조건의 영상 위주**
* 전신이 안정적으로 보이는 댄스 영상
* 발이 프레임 밖으로 벗어나지 않는 영상
* 턴, 회전, 빠른 팔 동작 등 다양한 동작 포함
* 조명·의상·카메라 거리 등 일부 조건 변형 포함 → 일반화 목적

---

## 6. 후처리(Post-processing)

모델 출력만으로는 Unity 애니메이션에 바로 사용하기 어렵기 때문에,
아래와 같은 **후처리 단계를 적용하여 안정성을 확보**하였습니다.

### 적용된 후처리 기법

* 프레임 간 모션 보간 (2회)
* 발 기준 재지면화
* 뼈 길이 제약
* 관절 각도 제한
* 발목 회전 제한
* y ≥ 0 안전 클램프

### 후처리 효과

* 단일 프레임 정확도 향상
* 프레임 간 튐 현상 제거
* 발 미끄러짐 및 관절 비정상 회전 감소
* Unity 애니메이션으로 사용 가능한 안정성 확보

---

## 7. Unity 연출 구성

### 카메라 연출

* **2개의 Cinemachine 카메라 사용**

  * 초반: 캐릭터 전신 소개
  * 중반: 캐릭터를 중심으로 회전하는 카메라 무빙
  * 후반: 캐릭터에 가까이 접근하는 엔딩 연출
* 카메라 전환은 Timeline 기반으로 제어

### Cinemachine Impulse

* 회전 카메라 구간에서 **미세한 충격 효과 추가**
* 과도하게 매끄러운 인공적 움직임을 방지
* 사람이 촬영한 듯한 자연스러운 카메라 감각 구현

### 동작 구성

* 단일 춤 동작 사용
* 여러 레퍼런스 중 **마지막 포즈(얼굴에 손을 가져가는 동작)**가 가장 잘 표현되는 입력을 선택
* 곡의 감정과 메시지를 강조하는 엔딩 포즈로 활용

---

## 8. 코드 구조

```text
AI_ML_Unity_Final/
├── AI_ML_Python_Final/
│   ├── 01_create_raw_keypoints.py
│   ├── 02_process_height_dataset.py
│   ├── 03_create_test_keypoints.py
│   ├── train.py
│   ├── test.py
│   └── postprocess.py
│
├── AI_ML_blazepose/
│   ├── BlazeDataExtraction.cs
│   ├── BlazeDataExtraction_v1.cs
│   ├── BlazeTimelineController.cs
│   ├── BlazePoseDataFile.cs
│   └── Unity Scene / Timeline
```

---

## 9. 프로젝트 특징 요약

* BlazePose 기반 3D 포즈 추출 → ML 기반 보정 → Unity 애니메이션
* 단일 프레임 정확도 + 시간적 연속성 동시 개선
* 모델 개선과 후처리를 **분리된 모듈**로 설계
* Timeline + Cinemachine을 활용한 시네마틱 연출
* 머신러닝 결과를 실제 콘텐츠 제작까지 연결한 End-to-End 파이프라인
