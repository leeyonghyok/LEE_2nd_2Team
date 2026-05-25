# 🎵 Spotify Churn Insight AI (SKN22-2nd-3Team)

> **스포티파이 사용자 이탈 예측 및 비즈니스 인사이트 제공 대시보드** > 사용자의 행동 패턴을 분석하여 이탈 가능성을 사전에 예측하고, 맞춤형 방어 전략을 제시합니다.

## 📌 프로젝트 개요 (Overview)
이 프로젝트는 음악 스트리밍 서비스 Spotify의 사용자 데이터를 머신러닝(ML)과 딥러닝(DL) 모델로 분석합니다.  
단순한 예측을 넘어, **'누가', '왜' 이탈하는지**를 설명하고 이를 막기 위한 **비즈니스 액션 플랜**을 제안하는 AI 웹 애플리케이션입니다.

## 👥 팀원 및 역할 (Members)
| 이름 | 역할 | 담당 업무 |
| :--- | :--- | :--- |
| **이신재** | 🎨 UI 구현 | Streamlit 웹 대시보드 기획 및 프론트엔드 구현, 시각화 |
| **장완식** | 🧠 모델 학습 | 머신러닝(RandomForest, XGBoost) 및 딥러닝 모델 설계 및 튜닝 |
| **구연미** | 🛠 데이터 전처리 | 결측치/이상치 처리, 파생 변수 생성, 데이터 스케일링 |
| **정세환** | 📊 EDA | 탐색적 데이터 분석, 상관관계 분석 및 주요 특징 도출 |

## 🚀 주요 기능 (Key Features)
1.  **모델 성능 비교 대시보드**: 다양한 AI 모델의 정확도와 F1-Score 비교 분석
2.  **실전 이탈 예측**: 사용자 정보를 입력하면 실시간으로 이탈 확률(%) 예측
3.  **심층 원인 분석**: 레이더 차트와 상세 리포트를 통해 이탈 위험 요인(Risk Factor) 시각화
4.  **비즈니스 전략 가이드**: 예측된 위험도(안전/주의/위험)에 따른 구체적인 마케팅/서비스 개선 전략 제안

## 📂 데이터셋 설명 (Data Dictionary)
이 데이터셋은 Spotify 사용자의 인구통계 정보, 이용 행태, 구독 정보 등을 포함하며, **이탈 여부(`is_churned`)**를 예측하는 것이 핵심 목표입니다.

| 구분 | 컬럼명 (Feature) | 설명 | 데이터 타입 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **식별자** | `user_id` | 사용자 고유 ID | String | 모델 학습 시 제외 |
| **유저 정보** | `gender` | 성별 (Male, Female, Other) | Categorical | |
| | `age` | 나이 | Numeric | |
| | `country` | 국가 / 지역 | Categorical | |
| **구독 정보** | `subscription_type` | 구독 요금제 유형 | Categorical | Free, Premium, Family, Student |
| **활동성** | `listening_time` | 하루 평균 청취 시간 (분) | Numeric | 서비스 몰입도 지표 |
| | `songs_played_per_day`| 하루 재생 곡 수 | Numeric | 활동량 지표 |
| | `device_type` | 주 사용 기기 | Categorical | Mobile, Desktop, Web |
| | `offline_listening` | 오프라인 모드 사용 여부 | Binary | 1: 사용, 0: 미사용 (Premium 기능) |
| **만족도/부정** | `skip_rate` | 노래 스킵 비율 (0.0 ~ 1.0) | Numeric | 높을수록 불만족 가능성 높음 |
| | `ads_listened_per_week`| 주간 광고 청취 수 | Numeric | Free 유저의 피로도 측정 지표 |
| **타겟(Target)**| `is_churned` | **이탈 여부** | Binary | **0: 유지 (Active)**<br>**1: 이탈 (Churned)** |

## 🛠 기술 스택 (Tech Stack)
* **Language**: Python 3.9+
* **Web Framework**: Streamlit
* **ML/DL**: Scikit-learn, XGBoost, TensorFlow (Keras)
* **Sampling**: Imbalanced-learn (BorderlineSMOTE)
* **Data Analysis**: Pandas, NumPy
* **Visualization**: Plotly, Matplotlib, Seaborn

## 📁 프로젝트 구조 (Directory Structure)

```
SKN22-2nd-3Team/
├── data/                  # 원본 및 전처리된 데이터
├── models/                # 학습된 모델 (.pkl, .h5) 및 전처리기
├── notebooks/             # 데이터 분석 및 모델 학습 노트북 (ipynb)
├── app.py                 # Streamlit 메인 애플리케이션
└── README.md              # 프로젝트 설명 파일
```
