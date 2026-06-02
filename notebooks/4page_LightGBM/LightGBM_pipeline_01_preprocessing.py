# LightGBM_pipeline_01_preprocessing.py
# # -*- coding: utf-8 -*-
"""
[Pipeline 1] 고객 이탈 예측을 위한 데이터 전처리 스크립트
-------------------------------------------------------------------------
1. 데이터 로드 및 결측치 확인
2. 타겟 변수 및 범주형 변수 인코딩 (문자열 변환 오류 방지)
3. 학습 및 검증 데이터셋 분할 (Train/Test set Split)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def run_preprocessing():
    # 1. 데이터 로드
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'synthetic_customer_churn_100k.csv')

    # 파일이 없을 때 오류 메세지 노출
    if not os.path.exists(DATA_PATH):
        print(f"[오류] '{DATA_PATH}' 파일이 현재 디렉토리에 없음.")
        return None, None, None, None

    # 데이터 불러오기
    df = pd.read_csv(DATA_PATH)
    print(f"데이터 크기: {df.shape})")

    # CustormerID 컬럼 제외하기
    df = df.drop('CustomerID', axis=1)

    # 2. 데이터 전처리 및 인코딩
    # 타겟 변수가 문자열일 때 숫자로 인코딩 ('Yes'->1, 'No'->0)
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        print("타겟 변수(Churn) 변환: 'Yes' -> 1, 'No' -> 0")

    # 범주형 변수 처리 (LightGBM 최적화를 위해 category 타입으로 변경)
    # 변수명이 바뀔 경우를 대비하여 categorical type의 변수들을 category로 묶어서 LightGBM이 스스로 탐색할 수 있도록 하기 위함.
    cat_features = ['Gender', 'Contract', 'PaymentMethod']
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"범주형 변수 타입 변환: [{col}] -> category")

    # 3. 학습 및 검증 데이터셋 분할 (Train / Test set Split)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Train 셋 크기: {X_train.shape}, Test 셋 크기: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = run_preprocessing()