import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_pipeline(file_path):
    # ==========================================
    # 1. 데이터 불러오기
    # ==========================================
    print("1. 데이터셋 로드 중...")
    df = pd.read_csv(file_path)
    print(f"데이터 크기: {df.shape}")

    # ==========================================
    # 2. 이상치 처리 : 음수값을 0으로 처리
    # ==========================================
    print("\n2. 이상치 처리 중...")
    df.loc[df['TotalCharges'] < 0, 'TotalCharges'] = 0.0

    # ==========================================
    # 3. Feature와 Target 분리(CustomID 이용X)
    # ==========================================
    print("\n3. Feature와 Target 분리 중...")
    feature_columns = ['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'Contract', 'PaymentMethod', 'TotalCharges']
    X = df[feature_columns].copy()
    y = df['Churn'].copy()

    # Target 변수 LabelEncoding (No -> 0, Yes -> 1)
    label_encoder_y = LabelEncoder()
    y_encoded = label_encoder_y.fit_transform(y)

    # ==========================================
    # 4. Train / Validation / Test 데이터셋 분리
    # ==========================================
    print("\n4. 데이터셋 분리 중 (Train: 60%, Val: 20%, Test: 20%)...")
    # 1단계: Train(60%)과 임시(40%) 분리
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.40, random_state=42, stratify=y_encoded
    )
    # 2단계: 임시(40%) 데이터를 Validation(20%)과 Test(20%)로 반반 분리
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # ==========================================
    # 5. 범주형 변수 인코딩
    # ==========================================
    print("\n5. 범주형 변수 LabelEncoding 적용 중...")
    categorical_columns = ['Gender', 'Contract', 'PaymentMethod']
    
    for col in categorical_columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_val[col] = le.transform(X_val[col])
        X_test[col] = le.transform(X_test[col])

    print("\n6. 수치형 변수 스케일링: 트리형 알고리즘 기준이므로 Scaling을 건너뜁니다.")

    # ==========================================
    # 7. 전처리 완료된 데이터 CSV 파일로 저장
    # ==========================================
    print("\n7. 전처리된 3분할 데이터셋을 CSV 파일로 저장 중...")
    
    # Feature 데이터 저장
    X_train.to_csv('X_train.csv', index=False)
    X_val.to_csv('X_val.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    
    # Target 데이터 저장 (훈련 파일 로드 시 .values.ravel() 연산과 호환되도록 단일 컬럼 형태로 저장)
    pd.Series(y_train, name='Churn').to_csv('y_train.csv', index=False)
    pd.Series(y_val, name='Churn').to_csv('y_val.csv', index=False)
    pd.Series(y_test, name='Churn').to_csv('y_test.csv', index=False)
    
    print(" - 파일 저장 완료: X_train.csv, y_train.csv")
    print(" - 파일 저장 완료: X_val.csv, y_val.csv")
    print(" - 파일 저장 완료: X_test.csv, y_test.csv")

    print("\n[완료] 모든 데이터 전처리 프로세스 및 파일 저장이 정상적으로 종료되었습니다.")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    data_path = "synthetic_customer_churn_100k.csv"
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_pipeline(data_path)
    except FileNotFoundError:
        print(f"\n[오류] '{data_path}' 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")