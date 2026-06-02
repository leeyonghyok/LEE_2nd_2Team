# %% [markdown]
# # RandomForest
# ### 수업자료 06_machin_learning\09_결정트리와 랜덤포레스트.ipynb에 근거한 Decision Tree와 Random Forest 모델링 

# %%
# 수업자료 06_machin_learning\09_결정트리와 랜덤포레스트.ipynb에 근거한 Decision Tree와 Random Forest 모델링 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import os


# %%
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "data/synthetic_customer_churn_100k.csv")

df = pd.read_csv(file_path, skipinitialspace=True)
df.shape

# %%
df.head()

# %%
df.info()

# %%
df.isnull().sum()

# %% [markdown]
# ### EDA 분석 권고에 따라 TotalCharges 음수값을 0으로 대체
# ### categorical_columns은 label encoding을 하고
# ### numeric_columns은 scalinf 하지 않음

# %%
# 음수 개수 확인
neg_count = (df['TotalCharges'] < 0).sum()
print(f"음수 개수: {neg_count}개")

# 0으로 교체
df['TotalCharges'] = df['TotalCharges'].apply(lambda x: 0 if x < 0 else x)
neg_count = (df['TotalCharges'] < 0).sum()
print(f"음수 개수: {neg_count}개")

# %%
# X, y 분리

from sklearn.preprocessing import LabelEncoder
print("분리전:", df.shape)
print(df["Churn"].value_counts())

le = LabelEncoder()
y = le.fit_transform(df["Churn"])
X = df.drop(columns=["CustomerID", "Churn"]) # CustomerID는 의미 없어서 지움
print("분리후:", X.shape, y.shape)
print(pd.Series(y).value_counts())




# %%
# train/val/test set 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train) # train 60%, val 20%, test 20% 비율로 생성
    

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

categorical_columns = ['Gender', 'Contract', 'PaymentMethod']
numeric_columns = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'Age']


# ── 1. 범주형 전처리 ──────────────────────────────
cat_imputer = SimpleImputer(strategy="most_frequent")
X_train[categorical_columns] = cat_imputer.fit_transform(X_train[categorical_columns])
X_val[categorical_columns]   = cat_imputer.transform(X_val[categorical_columns])   # ← 추가
X_test[categorical_columns]  = cat_imputer.transform(X_test[categorical_columns])

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[categorical_columns] = encoder.fit_transform(X_train[categorical_columns])
X_val[categorical_columns]   = encoder.transform(X_val[categorical_columns])       # ← 추가
X_test[categorical_columns]  = encoder.transform(X_test[categorical_columns])

# ── 2. 수치형 전처리 ──────────────────────────────
num_imputer = SimpleImputer(strategy="median")
X_train[numeric_columns] = num_imputer.fit_transform(X_train[numeric_columns])
X_val[numeric_columns]   = num_imputer.transform(X_val[numeric_columns])           # ← 추가
X_test[numeric_columns]  = num_imputer.transform(X_test[numeric_columns])


# %%
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(
    n_estimators=200, # DecisionTree 개수. (최소 200개)
    max_features=10,  # 지정한 feature수 내에서 random하게 feature들을 선택.
    max_depth=6,      # DecisionTree hyper parameter (모든 Decision Tree 모델들은 동일한 하이퍼파라미터를 가진다..)
    random_state=0,
    n_jobs=-1,        # 개별 DecisionTree 학습, 추론시 병렬 처리 할 때 사용할 프로세서 개수.(각 모델은 독립적으로 학습/추정한다. -1 : 모든 프로세서 다 사용)
)

# %%
# 학습(Train)
rfc.fit(X_train, y_train)

# 검증
## 추론: 클래스 결과과
pred_train = rfc.predict(X_train)
pred_val = rfc.predict(X_val)


## 추론: 클래스별 확률 결과
pred_train_proba = rfc.predict_proba(X_train)
pred_val_proba = rfc.predict_proba(X_val)

# %%
## 평가
from metrics import print_binary_classification_metrics
print_binary_classification_metrics(
    y_train, pred_train, pred_train_proba[:, 1], "Train set 검증결과"
)

# %%

print_binary_classification_metrics(
    y_val, pred_val, pred_val_proba[:, 1], "Validation set"
)

# %%
### fit() 뒤에 feature 중요도 조회
fi = rfc.feature_importances_
fi

# %%
fi.sum()

# %%
feature_names = categorical_columns + numeric_columns
fi_s = pd.Series(rfc.feature_importances_, index=feature_names).sort_values(ascending=False)


fi_s

# %%
fi_s.plot(kind='barh');

# %%
# 10여분 소요
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier(random_state=0)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 3, 6, 10],
    "min_samples_leaf": [1, 2, 4, 6],
}

total = (
    len(param_grid["n_estimators"]) *
    len(param_grid["max_depth"]) *
    len(param_grid["min_samples_leaf"])
)
print(f"총 조합 수: {total}개 × cv=5 = {total * 5}회 학습 예정")

gs = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=['r2', 'neg_mean_squared_error'],
    refit="r2",
    n_jobs=-1,
    verbose=2,      # ← 추가
)
gs.fit(X_train, y_train)



# %%
pd.DataFrame(gs.cv_results_).sort_values("rank_test_neg_mean_squared_error")

# %%
gs.best_params_

# %%
# "model__" 접두사 제거
best_params = {k.replace("model__", ""): v for k, v in gs.best_params_.items() if k.startswith("model__")}
print("정제된 파라미터:", best_params)

# 모델 생성 및 학습
best_model = RandomForestClassifier(
    **best_params,
    random_state=0,
    n_jobs=-1
)
best_model.fit(X_train, y_train)

# 추론
pred_test       = best_model.predict(X_test)
pred_test_proba = best_model.predict_proba(X_test)

# %%
# 수업중 사용한 metrics.py 파일
from metrics import print_binary_classification_metrics
print_binary_classification_metrics(
    y_test, pred_test, pred_test_proba[:, 1], "Test set"
)

# %%

import os
import joblib

# 현재 파일 기준 saved_models 폴더 (같은 레벨)
base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, "saved_models")
os.makedirs(save_dir, exist_ok=True)

# 모델 저장
joblib.dump(best_model, os.path.join(save_dir, "randomforest_model.pkl"))

# 전처리 변수 저장
joblib.dump(X_train, os.path.join(save_dir, "X_train.pkl"))
joblib.dump(X_val,   os.path.join(save_dir, "X_val.pkl"))
joblib.dump(X_test,  os.path.join(save_dir, "X_test.pkl"))
joblib.dump(y_train, os.path.join(save_dir, "y_train.pkl"))
joblib.dump(y_val,   os.path.join(save_dir, "y_val.pkl"))
joblib.dump(y_test,  os.path.join(save_dir, "y_test.pkl"))

print(f"저장 완료: {save_dir}")



# %%
