from data_scaling import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from deep_learning_model import DeepLearnModel
import os
from train import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data(load_data("data/synthetic_customer_churn_100k.csv"))

#모델 불러오기
model = DeepLearnModel()

#학습및 검증 파라미터 조정
epochs = 500  # early stopping 적용할 거라서 크게 설정.
batch_size = 128
threshold = 0.57
learning_rate= 0.001
# pos_weight는 데이터불균형 상태에서 가중치를 주기위함.
loss_weight = torch.tensor([2.0]).to(device)


#loss함수 및 옵티마이저 지정
loss_fn = nn.BCEWithLogitsLoss(pos_weight=loss_weight)    
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#Dataset, DataLoader 생성
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size*4)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size*4)

##학습 시작 (early stopping적용)

train_loss_list = []
val_loss_list = []
val_acc_list = []

best_score = torch.inf # 점차 내려가야 해서 infinite 설정
save_model_path = "saved_model/deep_model.pt"  # early stopping 할 때 저장할 파일 경로

patience = 10   # 몇 epoch동안 성능이 변함 없는지
stop = 0   # 연속적인 개선이 안 될 경우 count

print("학습 시작 하겠습니다.")

for epoch in range(epochs):             
    avg_train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, threshold, device)
    avg_val_loss, val_acc = validate(model, val_loader, loss_fn, threshold, device)
    
    print(f"Epoch [{epoch+1:02d}/{epochs}] | "
          f"Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2f}%) | "
          f"Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2f}%)")

    # Early Stopping
    if avg_val_loss < best_score:
        best_score = avg_val_loss  
        stop = 0    # stop 초기화           
        # 모델 저장
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        print(f"({best_score:.4f}) -> 모델 저장: {save_model_path}")
        torch.save(model, save_model_path)

    else:
        stop += 1
        print(f"성능 개선 없음. (Early Stopping 카운트: {stop}/{patience})")
        if stop >= patience:
            print(f"{patience}횟수 동안 성능 개선이 없어 학습을 종료합니다.")
            break