# LightGBM_pipeline_03_tuning.py
# -*- coding: utf-8 -*-
"""
[Pipeline 03] LightGBM 하이퍼파라미터 튜닝 및 상위 10개 결과 테이블 출력
-------------------------------------------------------------------------
- 다양한 파라미터 조합을 순회하며 학습 및 검증을 진행합니다.
- 각 조합별 파라미터 값과 성능 지표(Accuracy, ROC-AUC)를 수집합니다.
- 검증 성능(ROC-AUC) 기준 상위 10개의 조합을 정렬된 테이블로 출력하고 CSV로 저장합니다.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 전처리 파이프라인(pipeline_01_preprocessing.py) 가져오기
try:
    import LightGBM_pipeline_01_preprocessing as p1
except ImportError:
    # 전처리 파이프라인 파일이 없을 때 오류 메세지 노출
    print("[오류] 'pipeline_01_preprocessing.py' 파일이 동일한 폴더에 있어야 합니다.")
    exit()

def run_parameter_tuning():
    # 1. 전처리된 데이터 불러오기 (pipeline_01_preprocessing.py 모듈 호출)
    X_train, X_test, y_train, y_test = p1.run_preprocessing()
    if X_train is None:
        return
    print("-------------------------------------------------------------------------")
    # 검증 하이퍼파라미터 그리드서치 후보군 정의
    learning_rates = [0.01, 0.05, 0.1]
    num_leaves_list = [15, 31, 63]
    max_depths = [4, 6, -1]
    
    results_list = []
    trial_count = 1
    
    # 10만개 데이터셋의 모든 조합을 빠르게 연산하기 위해 튜닝 단계에서 훈련 데이터 중 최소 20,000개만을 사용
    X_train_sample = X_train.sample(n=min(20000, len(X_train)), random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]

    # 커스텀 그리드 서치 파이프라인 구축
    for lr in learning_rates:
        for num_leaves in num_leaves_list:
            for depth in max_depths:
                # 불필요하거나 모순되는 조합 제어 (트리 깊이 대비 리프 수가 과도하게 많으면 제외)
                if depth != -1 and num_leaves > (2 ** depth):
                    continue
                    
                print(f"[Trial {trial_count:02d}] 검증 진행 중 -> learning_rate: {lr}, num_leaves: {num_leaves}, max_depth: {depth}")
                
                # 모델 정의
                model = lgb.LGBMClassifier(
                    n_estimators=100,   # 트리 개수를 100개 정도로 가볍게 제한하여 "어떤 파라미터 조합이 상대적으로 우수한가?"의 경향성만 빠르게 파악
                    learning_rate=lr,
                    num_leaves=num_leaves,
                    max_depth=depth,
                    random_state=42,
                    class_weight='balanced',
                    verbose=-1 # 불필요한 트리에 대한 경고 숨겨주기
                )
                
                # 학습 및 예측 평가
                model.fit(X_train_sample, y_train_sample)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                # 테이블 구성을 위해 딕셔너리 형태로 데이터 적재
                results_list.append({
                    'Trial': trial_count,
                    'learning_rate': lr,
                    'num_leaves': num_leaves,
                    'max_depth': depth,
                    'Accuracy': round(acc, 4),
                    'ROC-AUC': round(auc, 4)
                })
                trial_count += 1

    # 2. 전체 결과를 판다스 데이터프레임 테이블로 변환
    df_results = pd.DataFrame(results_list)
    
    # 3. 중요 성능 지표인 ROC-AUC 기준으로 내림차순 정렬 후 동일값이 나올경우 Accuracy기준으로 상위 10개 행만 노출
    top_10_results = df_results.sort_values(by=['ROC-AUC', 'Accuracy'], ascending=False).head(10)
    
    # 4. 출력용 인덱스를 순위(Rank 1~10)로 재조정
    top_10_results = top_10_results.reset_index(drop=True)
    top_10_results.index = top_10_results.index + 1
    top_10_results.index.name = 'Rank'
    
    # 5. 화면에 테이블 결과 출력
    print("-------------------------------------------------------------------------")
    print("하이퍼파라미터 튜닝 성능 기준 상위 10개 결과")
    print("-------------------------------------------------------------------------")
    print(top_10_results.to_string())
    print("-------------------------------------------------------------------------")
    
    # 6. CSV 파일로 저장
    output_csv = '4page_LightGBM/lgb_tuning_top10_results.csv'
    top_10_results.to_csv(output_csv, encoding='utf-8-sig')
    print(f"튜닝 성능 상위 10개 결과 테이블이 '{output_csv}' 파일로 저장되었습니다.\n")
    
    return top_10_results

if __name__ == '__main__':
    run_parameter_tuning()