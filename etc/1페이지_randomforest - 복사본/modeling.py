import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from preprocessing_all import *

# 데이터 전처리
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_pipeline()

# 모델 정의
model = RandomForestClassifier(random_state=0)

# 파라미터 그리드
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# RandomizedSearchCV
grid_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=0,
    n_iter=20
)

print("학습시작")
grid_search.fit(X_train, y_train)

# 최적 파라미터 확인
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 점수: {grid_search.best_score_:.4f}")

# 저장
base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, "saved_models")
os.makedirs(save_dir, exist_ok=True)

joblib.dump(grid_search.best_estimator_, os.path.join(save_dir, "randomforest_model.pkl"))
joblib.dump(X_train, os.path.join(save_dir, "X_train.pkl"))
joblib.dump(X_val,   os.path.join(save_dir, "X_val.pkl"))
joblib.dump(X_test,  os.path.join(save_dir, "X_test.pkl"))
joblib.dump(y_train, os.path.join(save_dir, "y_train.pkl"))
joblib.dump(y_val,   os.path.join(save_dir, "y_val.pkl"))
joblib.dump(y_test,  os.path.join(save_dir, "y_test.pkl"))

print(f"저장 완료: {save_dir}")