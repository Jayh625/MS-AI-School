import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.functional as F 
import torch.optim as optim

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from Prac03_custom_dataset import CustomDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from lion_pytorch import Lion
import os 
from datetime import datetime
def make_dir_by_time () :
    now = datetime.now()
    now = str(now) 
    now = now.split(".")[0] 
    now = now.replace("-","").replace(" ","_").replace(":","")
    dir = os.path.join("./result3", now)
    os.makedirs(dir, exist_ok=True)
    return dir

def train(model, train_loader, val_loader, epochs, device, optimizer, criterion, result_dir) :
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs) :
        train_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        train_acc = 0.0
        
        # train
        model.train()
        train_loader_iter = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        for i, (data, target) in enumerate(train_loader_iter) :
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, pred = torch.max(output, 1)
            train_acc += (pred == target).sum().item()

            if i % 10 == 9 : 
                train_loader_iter.set_postfix({"Loss" : loss.item()})
                
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # eval
        model.eval()
        with torch.no_grad() :
            for data, target in val_loader :
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                pred = outputs.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(target.view_as(pred)).sum().item()
                val_loss += criterion(outputs, target).item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # save the model with the best val acc 
        if val_acc > best_val_acc :
            torch.save(model.state_dict(), os.path.join(result_dir,'best.pt'))
            best_val_acc = val_acc

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss : {train_loss:.4f}, "
              f"Train Acc : {train_acc:.4f}, "
              f"Val Loss : {val_loss:.4f}, "
              f"Val Acc : {val_acc:.4f}")

    return model, train_losses, val_losses, train_accs, val_accs

def main() :
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    result_dir = make_dir_by_time()
    # model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    # num_features = 1280
    # model.classifier[1] = nn.Linear(num_features, 6)
    # model.to(device)

    # model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # num_features = 1280
    # model.classifier[1] = nn.Linear(num_features, 6)
    # model.to(device)

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    num_features = 1280
    model.classifier[1] = nn.Linear(num_features, 6)
    model.to(device)

    # model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 6)
    # model.to(device)

    

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(degrees=15),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.2,0.2,0.2))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.2,0.2,0.2))
    ])

    train_dataset = CustomDataset("./data/fold/train", transform=train_transform)
    val_dataset = CustomDataset("./data/fold/val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=False)

    epochs = 20 
    criterion = CrossEntropyLoss().to(device)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    model, train_losses, val_losses, train_accs, val_accs = train(model, train_loader, val_loader, epochs, device, optimizer, criterion, result_dir)

    
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.savefig(os.path.join(result_dir,'Train_Val_loss.png'))
    plt.show()
    
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(result_dir,'Train_Val_Accuracy.png'))
    plt.show()

if __name__ == "__main__" :
    main()

"""
result3/20230714_170450
model = resnet18(weights=ResNet18_Weights.DEFAULT)
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
Epoch [1/20], Train Loss : 1.4961, Train Acc : 0.4358, Val Loss : 1.1908, Val Acc : 0.5899
Epoch [2/20], Train Loss : 0.9015, Train Acc : 0.7113, Val Loss : 0.7282, Val Acc : 0.7622
Epoch [3/20], Train Loss : 0.6151, Train Acc : 0.7894, Val Loss : 0.5485, Val Acc : 0.8308
Epoch [4/20], Train Loss : 0.5031, Train Acc : 0.8225, Val Loss : 0.3516, Val Acc : 0.8841
Epoch [5/20], Train Loss : 0.3627, Train Acc : 0.8781, Val Loss : 0.3002, Val Acc : 0.8979
Epoch [6/20], Train Loss : 0.2781, Train Acc : 0.8940, Val Loss : 0.1871, Val Acc : 0.9284
Epoch [7/20], Train Loss : 0.2634, Train Acc : 0.9086, Val Loss : 0.1308, Val Acc : 0.9527
Epoch [8/20], Train Loss : 0.2479, Train Acc : 0.9020, Val Loss : 0.1418, Val Acc : 0.9573
Epoch [9/20], Train Loss : 0.2083, Train Acc : 0.9311, Val Loss : 0.1581, Val Acc : 0.9604
Epoch [10/20], Train Loss : 0.1874, Train Acc : 0.9232, Val Loss : 0.1225, Val Acc : 0.9680
Epoch [11/20], Train Loss : 0.1499, Train Acc : 0.9536, Val Loss : 0.1018, Val Acc : 0.9619
Epoch [12/20], Train Loss : 0.1553, Train Acc : 0.9444, Val Loss : 0.2010, Val Acc : 0.9238
Epoch [13/20], Train Loss : 0.1540, Train Acc : 0.9364, Val Loss : 0.1268, Val Acc : 0.9604
Epoch [14/20], Train Loss : 0.1296, Train Acc : 0.9603, Val Loss : 0.3660, Val Acc : 0.9085
Epoch [15/20], Train Loss : 0.1236, Train Acc : 0.9576, Val Loss : 0.0719, Val Acc : 0.9756
Epoch [16/20], Train Loss : 0.1392, Train Acc : 0.9523, Val Loss : 0.0601, Val Acc : 0.9787
Epoch [17/20], Train Loss : 0.1041, Train Acc : 0.9616, Val Loss : 0.3014, Val Acc : 0.9131
Epoch [18/20], Train Loss : 0.1084, Train Acc : 0.9563, Val Loss : 0.4139, Val Acc : 0.9055
Epoch [19/20], Train Loss : 0.1309, Train Acc : 0.9483, Val Loss : 0.2951, Val Acc : 0.9207
Epoch [20/20], Train Loss : 0.0807, Train Acc : 0.9762, Val Loss : 0.2030, Val Acc : 0.9558
"""

"""
result3/20230714_171714
model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)

Epoch [1/20], Train Loss : 1.6619, Train Acc : 0.3589, Val Loss : 1.4324, Val Acc : 0.4787
Epoch [2/20], Train Loss : 1.1380, Train Acc : 0.6503, Val Loss : 0.9172, Val Acc : 0.7149
Epoch [3/20], Train Loss : 0.7088, Train Acc : 0.7536, Val Loss : 0.4275, Val Acc : 0.8933
Epoch [4/20], Train Loss : 0.4349, Train Acc : 0.8715, Val Loss : 0.1684, Val Acc : 0.9649
Epoch [5/20], Train Loss : 0.2776, Train Acc : 0.9179, Val Loss : 0.0937, Val Acc : 0.9771
Epoch [6/20], Train Loss : 0.1794, Train Acc : 0.9444, Val Loss : 0.0700, Val Acc : 0.9756
Epoch [7/20], Train Loss : 0.2087, Train Acc : 0.9351, Val Loss : 0.0621, Val Acc : 0.9893
Epoch [8/20], Train Loss : 0.1343, Train Acc : 0.9563, Val Loss : 0.0215, Val Acc : 0.9970
Epoch [9/20], Train Loss : 0.1338, Train Acc : 0.9523, Val Loss : 0.0139, Val Acc : 0.9954
Epoch [10/20], Train Loss : 0.1162, Train Acc : 0.9550, Val Loss : 0.0201, Val Acc : 0.9909
Epoch [11/20], Train Loss : 0.1069, Train Acc : 0.9629, Val Loss : 0.0437, Val Acc : 0.9924
Epoch [12/20], Train Loss : 0.1554, Train Acc : 0.9497, Val Loss : 0.0221, Val Acc : 0.9954
Epoch [13/20], Train Loss : 0.1214, Train Acc : 0.9642, Val Loss : 0.0257, Val Acc : 0.9893
Epoch [14/20], Train Loss : 0.1534, Train Acc : 0.9523, Val Loss : 0.0270, Val Acc : 0.9924
Epoch [15/20], Train Loss : 0.0947, Train Acc : 0.9669, Val Loss : 0.0449, Val Acc : 0.9863
Epoch [16/20], Train Loss : 0.1235, Train Acc : 0.9616, Val Loss : 0.0299, Val Acc : 0.9878
Epoch [17/20], Train Loss : 0.1462, Train Acc : 0.9497, Val Loss : 0.0185, Val Acc : 0.9924
Epoch [18/20], Train Loss : 0.1302, Train Acc : 0.9576, Val Loss : 0.0195, Val Acc : 0.9909
Epoch [19/20], Train Loss : 0.1041, Train Acc : 0.9695, Val Loss : 0.0254, Val Acc : 0.9939
Epoch [20/20], Train Loss : 0.0874, Train Acc : 0.9722, Val Loss : 0.0348, Val Acc : 0.9878
"""