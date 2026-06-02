# LightGBM_pipeline_02_training.py
# -*- coding: utf-8 -*-
"""
[Pipeline 2] LightGBM 모델 학습, 평가
-------------------------------------------------------------------------
4. LightGBM 모델 생성 및 학습 (early_stopping 적용 포함)
5. 테스트 데이터를 활용한 모델 예측 및 평가 (정확도, ROC-AUC, 분류 리포트)
6. 피처 중요도(Feature Importance) 그래프 저장
"""

import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def run_model_training_and_evaluation(X_train, X_test, y_train, y_test, best_params=None):
    # 전처리가 실행되지 않았을 경우 오류 메세지 노출
    if X_train is None:
        print("[오류] 입력 데이터가 올바르지 않습니다.")
        return

    # 4. LightGBM 모델 생성 및 학습
    # 기본 설정 지정 > 튜닝 후 최적 파라미터(best_params)가 들어오면 해당 값으로 변경 실행 
    lr = best_params['learning_rate'] if best_params else 0.05
    num_leaves = best_params['num_leaves'] if best_params else 31
    max_depth = best_params['max_depth'] if best_params else -1

    print(f"최종 모델 파라미터 -> lr: {lr}, num_leaves: {num_leaves}, max_depth: {max_depth}")

    model = lgb.LGBMClassifier(
        n_estimators=500,       # 반복횟수를 크게 지정 > 밑에서 early_stopping 설정.
        learning_rate=lr,     
        num_leaves=num_leaves,          
        max_depth=max_depth,           
        random_state=42,
        class_weight='balanced' 
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            # n_estimators=500으로 크게 지정해두었기때문에, 최고 성능이 도달했을때 stopping_rounds만큼만 더 반복하고 
            # 더이상 성능에 변화가 없으면 반복을 멈추도록 early_stopping 설정.
            # 과적합을 방지하고 시간과 메모리 낭비를 줄일 수 있음.
            lgb.early_stopping(stopping_rounds=10, verbose=False), 
            lgb.log_evaluation(period=50)
        ]
    )
    print("모델 생성 및 학습 완료")

    # 5. 모델 평가
    # 훈련 데이터 지표도 대시보드 표에 함께 뿌려주기 위해 예측값 추가 산출
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 데이터 타입(object, <U3 등)에 상관없이 예측값과 정답지를 무조건 숫자 0과 1로 강제 통일.
    y_train_pred = np.array([1 if str(x).strip().lower() in ['yes', '1', '1.0'] else 0 for x in y_train_pred])
    y_pred = np.array([1 if str(x).strip().lower() in ['yes', '1', '1.0'] else 0 for x in y_pred])
    
    y_train_numeric = np.array([1 if str(x).strip().lower() in ['yes', '1', '1.0'] else 0 for x in y_train])
    y_test_numeric = np.array([1 if str(x).strip().lower() in ['yes', '1', '1.0'] else 0 for x in y_test])

    print("\n------------------ [ LightGBM 모델 평가 결과 ] ------------------")
    # 원본 리포트와 완벽히 일치하도록, 규격화된 numeric 변수들을 매칭.
    print(f"1. 정확도 (Accuracy): {accuracy_score(y_test_numeric, y_pred):.4f}")
    print(f"2. ROC-AUC 점수     : {roc_auc_score(y_test_numeric, y_pred_proba):.4f}")
    print("\n3. 분류 리포트 (Classification Report):")
    print(classification_report(y_test_numeric, y_pred))
    print("-----------------------------------------------------------------")

    # 6. 피쳐 중요도 (Feature Importance) 그래프 생성한 후 그림파일로 저장하기
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    sns.set_theme(style="whitegrid")
    # OS별 폰트 확인하기
    for font in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']:
        try:
            plt.rcParams['font.family'] = font
            break
        except:
            pass
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('LightGBM Feature Importance for Churn Prediction', fontsize=14)
    plt.xlabel('Importance')
    plt.tight_layout()
    
    output_image = '4page_LightGBM/lgb_feature_importance.png'
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"피처 중요도 그래프 저장: '{output_image}'")
    print("-----------------------------------------------------------------")

    # app.py 대시보드가 읽을 수 있도록 평가지표 데이터프레임 구조를 딕셔너리로 반환
    # 양쪽 모두 완전한 숫자 [0, 1] 상태이므로 에러 없이 정상 연산됩니다.
    metrics = {
        '평가 지표 (Metric)': ['정확도 (Accuracy)', '정밀도 (Precision)', '재현율 (Recall)', 'F1-Score', 'ROC-AUC'],
        '훈련 데이터 성능 (Train)': [
            f"{accuracy_score(y_train_numeric, y_train_pred):.4f}",
            f"{precision_score(y_train_numeric, y_train_pred):.4f}",
            f"{recall_score(y_train_numeric, y_train_pred):.4f}",
            f"{f1_score(y_train_numeric, y_train_pred):.4f}",
            f"{roc_auc_score(y_train_numeric, y_train_proba):.4f}"
        ],
        '검증 데이터 성능 (Test)': [
            f"{accuracy_score(y_test_numeric, y_pred):.4f}",
            f"{precision_score(y_test_numeric, y_pred):.4f}",
            f"{recall_score(y_test_numeric, y_pred):.4f}",
            f"{f1_score(y_test_numeric, y_pred):.4f}",
            f"{roc_auc_score(y_test_numeric, y_pred_proba):.4f}"
        ]
    }
    return metrics

if __name__ == '__main__':
    # pipeline_02_training.py 파일을 독립 실행 시 자동으로 pipeline_01_preprocessing.py 파일을 실행
    try:
        import LightGBM_pipeline_01_preprocessing as p1
        X_train, X_test, y_train, y_test = p1.run_preprocessing()
        if X_train is not None:
            run_model_training_and_evaluation(X_train, X_test, y_train, y_test)
    # pipeline_01_preprocessing.py 파일 자동실행이 실패할 때 오류 메세지 노출
    except ImportError:
        print("[오류] 'LightGBM_pipeline_01_preprocessing.py' 파일이 동일 폴더에 있어야 자동으로 실행합니다.")