from sklearn.metrics import classification_report, confusion_matrix,  roc_auc_score
from data_scaling import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data(load_data("05_deeplearning/synthetic_customer_churn_100k.csv"))

#nn.BCEWithLogitsLoss() 사용시 float type이어야 해서 변환.
x_train, y_train = torch.FloatTensor(x_train), torch.FloatTensor(y_train).unsqueeze(-1)
x_val, y_val = torch.FloatTensor(x_val), torch.FloatTensor(y_val).unsqueeze(-1)
x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test).unsqueeze(-1)

input_dim = x_train.shape[1]
model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),

    nn.Linear(64, 32),
    nn.ReLU(),

    nn.Linear(32, 1)
).to(device)

# 이진 분류에 적합 모델 마지막에 sigmoid() 사용 하면 안됌. pos_weight는 데이터불균형 상태에서 가중치를 주기위함.
loss_weight = torch.tensor([2.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight)    

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

epochs = 75
batch_size = 200
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=10000, shuffle=False)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)


print('학습 시작')

threshold = 0.55


for epoch in range(epochs):             
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # threshold 조정 구간
        predicted = (torch.sigmoid(outputs) >=threshold).float()
        
        train_loss += loss.item() # 평균값 측정에 사용할 변수

        # accuracy측정에 사용할 변수
        train_total += batch_y.size(0) 
        train_correct += (predicted == batch_y).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = (train_correct / train_total) * 100

    model.eval() 
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            val_outputs = model(batch_x)
            v_loss = criterion(val_outputs, batch_y)
            
            val_predicted = (torch.sigmoid(val_outputs) >= threshold).float()

            val_loss += v_loss.item()
            val_total += batch_y.size(0)
            val_correct += (val_predicted == batch_y).sum().item()
  

    avg_val_loss = val_loss / len(val_loader)
    val_acc = (val_correct / val_total) * 100 
    
    # 매 학습마다 Loss와 함께 Accuracy(정확도)도 출력

    print(f"Epoch [{epoch+1:02d}/{epochs}] | "
          f"Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2f}%) | "
          f"Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2f}%)")
     

model.eval()
final_preds = []
final_targets = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        
        predicted = (torch.sigmoid(outputs) >= threshold).float()

        # numpy배열은 cpu만 가능.
        final_preds.extend(predicted.cpu().numpy().flatten())
        final_targets.extend(batch_y.cpu().numpy().flatten())

print("Confusion Matrix")
print(confusion_matrix(final_targets, final_preds))

print("상세 분류 리포트")
print(classification_report(final_targets, final_preds, target_names=['Stay (0)', 'Churn (1)']))
auc = roc_auc_score(final_targets, final_preds)
print(auc)
