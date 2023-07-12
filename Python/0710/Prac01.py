import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # 이미지를 -1 ~ 1로 정규화
])

train_dataset = CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# Resnet-18 모델 정의
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10) # CIFAR-10의 클래스 개수 10

# 배깅 앙상블 모델 정의
bagging_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=7),
    n_estimators=5
)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

def train(model, device, train_loader, optimizer, criterion) :
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader) :
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, device, test_loader) :
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad() :
        for data, target in test_loader :
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(target.cpu().numpy())
    accuracy = accuracy_score(targets, predictions)
    return accuracy

def ensemble_predict(models, device, test_loader) :
    predictions = []
    with torch.no_grad() :
        for data, _ in test_loader :
            data = data.to(device)
            outputs = []
            for model in models :
                model = model.to(device)
                model.eval()
                output = model(data)
                outputs.append(output)
            ensemble_output = torch.stack(outputs).mean(dim=0)
            _, predicted = torch.max(ensemble_output, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

if __name__ == '__main__' :
    models = []
    for epoch in range(1, 6) :# 배깅 앙상블에는 5개의 모델 사용
        print(f"Training Model : {epoch}")
        model = model.to(device)
        train(model, device, train_loader, optimizer, criterion)
        accuracy = evaluate(model, device, test_loader)
        print(f"Model {epoch} Accuracy : {accuracy:.2f}")
        models.append(model)
    
    ensemble_predictions = ensemble_predict(models, device, test_loader)
    ensemble_accuracy = accuracy_score(test_dataset.targets, ensemble_predictions)
    print(f"\nEnsemble Accuracy : {ensemble_accuracy:.2f}")

"""
Training Model : 1
Model 1 Accuracy : 0.75
Training Model : 2
Model 2 Accuracy : 0.77
Training Model : 3
Model 3 Accuracy : 0.79
Training Model : 4
Model 4 Accuracy : 0.81
Training Model : 5
Model 5 Accuracy : 0.81

Ensemble Accuracy : 0.81
"""