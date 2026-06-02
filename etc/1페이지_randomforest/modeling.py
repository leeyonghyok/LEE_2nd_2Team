import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from data_scaling import *
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import joblib

# 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, "data", "synthetic_customer_churn_100k.csv")

data = load_data(path)
x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data(data)

model = RandomForestClassifier(random_state=0)
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

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
grid_search.fit(x_train, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 점수: {grid_search.best_score_:.4f}")

# 저장
save_dir = os.path.join(base_dir, "saved_models")
os.makedirs(save_dir, exist_ok=True)

joblib.dump(grid_search.best_estimator_, os.path.join(save_dir, "randomforest_model.pkl"))
joblib.dump(x_train, os.path.join(save_dir, "x_train.pkl"))
joblib.dump(x_val,   os.path.join(save_dir, "x_val.pkl"))
joblib.dump(x_test,  os.path.join(save_dir, "x_test.pkl"))
joblib.dump(y_train, os.path.join(save_dir, "y_train.pkl"))
joblib.dump(y_val,   os.path.join(save_dir, "y_val.pkl"))
joblib.dump(y_test,  os.path.join(save_dir, "y_test.pkl"))

print(f"저장 완료: {save_dir}")
