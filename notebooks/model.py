from data_scaling_lyc import *
from sklearn.ensemble import RandomForestClassifier


a = load_data("synthetic_customer_churn_100k.csv")


x_train, x_val, x_test, y_train, y_val, y_test=preprocess_data(a)

model = RandomForestClassifier()


print(x_train.shape)