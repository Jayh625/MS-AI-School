import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg11, resnet18, VGG11_Weights, ResNet18_Weights
from sklearn.metrics import accuracy_score

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 데이터셋을 불러와서 전처리
train_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
])

train_dataset = CIFAR10(root='./data', train=True, download=False, transform=train_transform)
test_dataset = CIFAR10(root='./data', train=False, download=False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

# VGG11 모델과 ResNet18 모델 정의
vgg_model = vgg11(weights=VGG11_Weights.DEFAULT)
resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features_vgg = vgg_model.classifier[6].in_features
num_features_resnet = resnet_model.fc.in_features
vgg_model.classifier[6] = nn.Linear(num_features_vgg, 10)
resnet_model.fc = nn.Linear(num_features_resnet, 10)

class EnsembleModel(nn.Module) :
    def __init__(self, models) :
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x) :
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=1) # 소프트 보팅 적용
        return outputs

ensemble_model = EnsembleModel([vgg_model, resnet_model])
criterion = nn.CrossEntropyLoss()
optimzer = optim.AdamW(ensemble_model.parameters(), lr=0.001)

def train(model, device, train_loader, optimizer, criterion) :
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader) :
        data, target = data.to(device), target.to(device)
        optimzer.zero_grad()
        output = model(data)
        loss = criterion(output.mean(dim=1), target) # 소프트 보팅 적용
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
            predictions.append(output)

    combined_predictions = torch.cat(predictions, dim=0).mean(dim=1) # 소프트 보팅 적용
    _, predicted_labels = torch.max(combined_predictions, dim=1)
    accuracy = accuracy_score(test_dataset.targets, predicted_labels.cpu().numpy())
    return accuracy

if __name__ == '__main__' :
    for epoch in range(1,6) :
        print(f"Training Model {epoch}")
        ensemble_model = ensemble_model.to(device)
        train(ensemble_model, device, train_loader, optimzer, criterion)
        
        accuracy = evaluate(ensemble_model, device, test_loader) 
        print(f"Epoch {epoch} Accuracy : {accuracy:.2f}")
    
"""
Training Model 1
Epoch 1 Accuracy : 0.76
Training Model 2
Epoch 2 Accuracy : 0.78
Training Model 3
Epoch 3 Accuracy : 0.79
Training Model 4
Epoch 4 Accuracy : 0.79
Training Model 5
Epoch 5 Accuracy : 0.80
"""