# Preprocessing Report: Telco Customer Churn Data 


**프로젝트**: 통신사 고객 이탈(Churn) 예측 모델링 구축

**데이터셋**: synthetic_customer_churn_100k.csv<br>
(출처 : https://www.kaggle.com/datasets/dhrubangtalukdar/telco-customer-churn-data)

## 1. 데이터셋 개요

### 1.1 데이터셋 설명
데이터셋은 **가상의 통신사** Telco의 고객 이탈(churn) 자료로 모든 데이터는 Python(pandas + numpy)으로 시드 고정 후 생성한 **가공 데이터**이다

※ 통신사 고객 이탈 가공 데이터의 시초는 IBM Cognos Analytics용 샘플 데이터로 제공되는 Telco Customer Churn이다. 이후 Kaggle에서 비슷한 유형의 데이터셋을 찾아볼 수 있으며 synthetic_customer_churn_100k도 그중 하나다.

### 1.2 변수에 대한 설명

| 컬럼명 | 설명 | 데이터 타입 | 예시 |
|---|---|---|---|
| CustomerID | 고객 고유 식별자 | int | 1, 2, …, 100000 |
| Age | 고객 나이 (18–80세) | int | 51 |
| Gender | 고객 성별 | string | Male / Female / Other |
| Tenure | 서비스 이용 기간 (월, 1–72) | int | 58 |
| MonthlyCharges | 월 청구 금액 (USD, 약 10–150) | float | 95.92 |
| TotalCharges | 누적 청구 금액 (Tenure × MonthlyCharges + 노이즈) | float | 5530.46 |
| Contract | 계약 유형 | string | 월별 / 1년 / 2년 |
| PaymentMethod | 결제 수단 | string | 전자수표 / 우편수표 / 계좌이체 / 신용카드 |
| Churn | 이탈 여부 (타깃 변수) | string | Yes / No |


# 2. EDA 보고서 검토

## 2.1 EDA 개요
- **데이터 규모:** 총 100,000행(Rows), 9열(Columns)
- **변수 타입 구성:**
  - **수치형(Numerical) 변수:** `CustomerID`, `Age`, `Tenure`, `MonthlyCharges`, `TotalCharges` (5개)
  - **범주형(Categorical) 변수:** `Gender`, `Contract`, `PaymentMethod`, `Churn` (4개)
- **결측치 현황:** 모든 변수의 Non-Null Count가 100,000개로 일치하여 결측치 미존재
- **이상치 현황:** TotalCharges의 음수(-) 값:** 총 청구 금액에 마이너스 값이 잡혀 있습니다. 이는 환불, 프로모션 크레딧 또는 시스템 기록 오류일 수 있으므로 **0 이하의 값은 0으로 대체하는 데이터 전처리**가 필요

## 2.2. 타겟 변수(Churn) 분포 분석

모델이 예측해야 하는 핵심 타겟 변수인 **고객 이탈 여부 (`Churn`)** 의 비율을 파악

![이탈 분포](01_churn_distribution.png)

- **이탈하지 않은 고객 (No):** 66,856명 (**66.86%**)
- **이탈한 고객 (Yes):** 33,144명 (**33.14%**)
- **클래스 불균형 분석**<br>
본 데이터의 이탈률이 **약 33%** 로 꽤 높은 편이나 별도의. 불균형 전처리를 적용하지 않음

# 3. 데이터 전처리

### 3.1 이상치 처리

- TotalCharges 음수(-) 값을 0으로 대체
- 총 100,000개 중 265개의 음수(-)을 0으로 대체

### 3.2 feature와 target 분리(X, y 분리)

- 총 컬럼(9개): CustomerID, Age, Gender, Tenure, MonthlyCharges, Contract, PaymentMethod, TotalCharges, Churn
- feature(7개): Age, Gender, Tenure, MonthlyCharges, Contract, PaymentMethod, TotalCharges<br>
  - categorical_columns = ['Gender', 'Contract', 'PaymentMethod']<br>
  - numeric_columns = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'Age']

- target(1개): Churn
- target은 LabelEncoding을 통해 Yes를 0으로 , No를 1로 변환

| 구분 | 레이블 | 의미 | 건수 |
|------|--------|------|-----:|
| 변환 전 | No | 유지 | 66,856 |
| 변환 전 | Yes | 이탈 | 33,144 |
| 변환 후 | 0 | 유지 | 66,856 |
| 변환 후 | 1 | 이탈 | 33,144 |

### 3.3 train/validation/test set 분리

- train set(60%), validation set(20%), test set(20%)로 분리하여 데이터 준비

| 변수 | 크기 |
|------|-----:|
| X_train | 60,000 |
| y_train | 60,000 |
| X_val | 20,000 |
| y_val | 20,000 |
| X_test | 20,000 |
| y_test | 20,000 |


### 3.4 OneHotEncoding과 labelEcoding
- 범주형(categorical) 변수에 대해서는 LabelEncoding을 함



### 3.5 Scaling
- 수치형(numeric) 변수에 대해서는 별도의 Scaling을 하지 않음

### 3.6 요약

- 머신러닝 Encoding과 ㄴScaling 정리


| 척도 | | 선형 알고리즘 | 트리 알고리즘 |
|---|---|---|---|
| 범주형 | 명목척도 | `OneHotEncoder(drop='first')` | `OneHotEncoder -> LabelEncoder() 가능` |
| 범주형 | 순서척도 | `LabelEncoder()` | `LabelEncoder())` |
| 수치형 | 간격척도 | Scaling 필요 | Scaling 불필요 |
| 수치형 | 비율척도 | Scaling 필요 | Scaling 불필요 |


\* OneHotEncoder(drop='first'): dummy 변수화<br>
\* 선형 알고리즘: Logistic Regression(Linear Regression), SVM, KNN<br>
\* 트리형 알고리즘: Decision Tree, Random Forest, Gradient Boosing(XGBoost, LightGBM, CatBoost)
