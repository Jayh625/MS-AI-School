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
        outputs = torch.stack(outputs, dim=0)
        avg_outputs = torch.mean(outputs, dim=0)
        return avg_outputs

ensemble_model = EnsembleModel([vgg_model, resnet_model])
criterion = nn.CrossEntropyLoss()
optimzer = optim.AdamW(ensemble_model.parameters(), lr=0.001)

def train(model, device, train_loader, optimizer, criterion) :
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader) :
        data, target = data.to(device), target.to(device)
        optimzer.zero_grad()
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

def combine_predictions(predictions) :
    combined = torch.cat(predictions, dim=0)
    _, predicted_labels = torch.max(combined, 1)
    return predicted_labels

if __name__ == '__main__' :
    for epoch in range(1,6) :
        print(f"Training Model {epoch}")
        ensemble_model = ensemble_model.to(device)
        train(ensemble_model, device, train_loader, optimzer, criterion)
        predictions = []
        with torch.no_grad() :
            for data, _ in test_loader :
                data = data.to(device)
                output = ensemble_model(data)
                predictions.append(output)

        conbined_predictions = combine_predictions(predictions)
        accuracy = accuracy_score(test_dataset.targets, conbined_predictions.cpu().numpy())
        print(f"Model {epoch} Accuracy : {accuracy:.2f}")
    
"""
Training Model 1
Model 1 Accuracy : 0.77
Training Model 2
Model 2 Accuracy : 0.80
Training Model 3
Model 3 Accuracy : 0.81
Training Model 4
Model 4 Accuracy : 0.82
Training Model 5
Model 5 Accuracy : 0.82
"""