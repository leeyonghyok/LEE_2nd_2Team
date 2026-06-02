from data_scaling import *
import torch.nn as nn

x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data(load_data("data/synthetic_customer_churn_100k.csv"))
input_dim = x_train.shape[1]

# deeplearning 모델 정의
class DeepLearnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr1 = nn.Linear(input_dim,7)
        self.lr2 = nn.Linear(7,1)
        self.relu = nn.ReLU()

    def forward( self, x):
        # 입력데이터가 float64 일 경우 float32로 변환
        x = x.float() 
        out = self.lr1(x)
        out = self.relu(out)

        output = self.lr2(out)

        return output
    

   