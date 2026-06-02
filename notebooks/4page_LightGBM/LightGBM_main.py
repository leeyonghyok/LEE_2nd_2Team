# LightGBM_main.py
# -*- coding: utf-8 -*-
"""
[필수 라이브러리 설치 안내]
이 프로젝트를 실행하기 위해 아래 라이브러리들이 필요합니다.
터미널(Terminal)에서 다음 명령어를 실행하여 한 번에 설치할 수 있습니다:

pip install streamlit pandas numpy scikit-learn lightgbm matplotlib seaborn
-------------------------------------------------------------------------
-------------------------------------------------------------------------
고객 이탈 예측 메인(main.py)
-------------------------------------------------------------------------
1. Pipeline 01: 데이터 로드 및 전처리/분할
2. Pipeline 03: 하이퍼파라미터 튜닝 수행 ➔ 상위 10개 출력 및 최적 조합 도출
3. Pipeline 02: 도출된 최적 파라미터 주입 ➔ 최종 대규모 학습 및 평가
"""

import LightGBM_pipeline_01_preprocessing as pipe1
import LightGBM_pipeline_02_training as pipe2
import LightGBM_pipeline_03_tuning as pipe3

def main():
    print("<데이터 전처리>")

    # 단계 1: 데이터 전처리 및 train / test set 분할
    X_train, X_test, y_train, y_test = pipe1.run_preprocessing()
    
    if X_train is None:
        print("[오류] 데이터 전처리 단계를 실패하여 파이프라인을 중단합니다.")
        return

    # 단계 2: 하이퍼파라미터 튜닝 및 상위 10개 테이블 출력
    print("\n" + "-------------------------------------------------------------------------")
    print("<하이퍼파라미터 튜닝 진행>")
    print("-------------------------------------------------------------------------")
    
    # 튜닝을 실행하여 상위 10개 결과를 뽑고, 최적의 파라미터를 딕셔너리로 받아옵니다.
    top_10_df = pipe3.run_parameter_tuning()
    
    # 테이블의 가장 첫 번째 행(Rank 1)에서 최적의 파라미터를 추출합니다.
    best_params = {
        'learning_rate': float(top_10_df.iloc[0]['learning_rate']),
        'num_leaves': int(top_10_df.iloc[0]['num_leaves']),
        'max_depth': int(top_10_df.iloc[0]['max_depth'])
    }
    
    print(f"[튜닝 완료] 최적의 파라미터 조합: {best_params}")

    # 단계 3: 최적 파라미터를 적용한 최종 모델 학습 및 평가
    print("\n" + "-------------------------------------------------------------------------")
    print("<최적 파라미터 기반 최종 모델 학습 진행>")
    print("-------------------------------------------------------------------------")
    
    # pipeline_02_training.py에 파라미터를 자동으로 반환.
    pipe2.run_model_training_and_evaluation(
        X_train, X_test, y_train, y_test, 
        best_params=best_params
    )
    
    print("\n[!전체 예측 시스템 파이프라인 완료.]")

if __name__ == '__main__':
    main()