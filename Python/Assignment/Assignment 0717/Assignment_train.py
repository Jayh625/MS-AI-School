import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.functional as F 
import torch.optim as optim

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from Assignment_custom_dataset import CustomDataset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from lion_pytorch import Lion
import os 
from datetime import datetime

def make_dir_by_time () :
    now = datetime.now()
    now = str(now) 
    now = now.split(".")[0] 
    now = now.replace("-","").replace(" ","_").replace(":","")
    dir = os.path.join("./assignment", now)
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
        for i, (data, target, _) in enumerate(train_loader_iter) :
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
            for data, target, _ in val_loader :
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
    
    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(result_dir,'loss.png'))
    
    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(result_dir,'accuracy.png'))

    df = pd.DataFrame({
        'Train Loss' : train_losses,
        'Train Accuracy' : train_accs,
        'Validation Loss' : val_losses,
        'Validation Accuracy' : val_accs
    })

    df.to_csv(os.path.join(result_dir, 'train_val_results.csv'), index=False)

    return model, train_losses, val_losses, train_accs, val_accs

def main() :
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    result_dir = make_dir_by_time()

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    num_features = 1280
    model.classifier[1] = nn.Linear(num_features, 10)
    model.to(device)

    # model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 10)
    # model.to(device)

    # model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    # num_features = 1280
    # model.classifier[1] = nn.Linear(num_features, 10)
    # model.to(device)

    train_transform = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(degrees=15),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.2,0.2,0.2))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.2,0.2,0.2))
    ])

    train_dataset = CustomDataset("./data/metal_damaged/train", transform=train_transform)
    val_dataset = CustomDataset("./data/metal_damaged/val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=False)

    epochs = 20 
    criterion = CrossEntropyLoss().to(device)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    model, train_losses, val_losses, train_accs, val_accs = train(model, train_loader, val_loader, epochs, device, optimizer, criterion, result_dir)

if __name__ == "__main__" :
    main()

"""
model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
batch_size=64
epochs = 20 
criterion = CrossEntropyLoss().to(device)
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)

Epoch [1/20], Train Loss : 1.4055, Train Acc : 0.5723, Val Loss : 0.4810, Val Acc : 0.8507
Epoch [2/20], Train Loss : 0.5005, Train Acc : 0.8431, Val Loss : 0.4443, Val Acc : 0.8676
Epoch [3/20], Train Loss : 0.3955, Train Acc : 0.8798, Val Loss : 0.3422, Val Acc : 0.8873
Epoch [4/20], Train Loss : 0.3266, Train Acc : 0.8987, Val Loss : 0.2809, Val Acc : 0.9070
Epoch [5/20], Train Loss : 0.3325, Train Acc : 0.8968, Val Loss : 0.3390, Val Acc : 0.8958
Epoch [6/20], Train Loss : 0.2797, Train Acc : 0.9112, Val Loss : 0.3260, Val Acc : 0.9155
Epoch [7/20], Train Loss : 0.2768, Train Acc : 0.9118, Val Loss : 0.2991, Val Acc : 0.9127
Epoch [8/20], Train Loss : 0.2678, Train Acc : 0.9156, Val Loss : 0.2359, Val Acc : 0.9211
Epoch [9/20], Train Loss : 0.2422, Train Acc : 0.9247, Val Loss : 0.2757, Val Acc : 0.9239
Epoch [10/20], Train Loss : 0.2405, Train Acc : 0.9200, Val Loss : 0.3190, Val Acc : 0.9183
Epoch [11/20], Train Loss : 0.2332, Train Acc : 0.9216, Val Loss : 0.2994, Val Acc : 0.9070
Epoch [12/20], Train Loss : 0.2131, Train Acc : 0.9278, Val Loss : 0.4208, Val Acc : 0.9099
Epoch [13/20], Train Loss : 0.2243, Train Acc : 0.9272, Val Loss : 0.3137, Val Acc : 0.9155
Epoch [14/20], Train Loss : 0.1893, Train Acc : 0.9426, Val Loss : 0.3080, Val Acc : 0.9155
Epoch [15/20], Train Loss : 0.1893, Train Acc : 0.9366, Val Loss : 0.2732, Val Acc : 0.9239
Epoch [16/20], Train Loss : 0.1808, Train Acc : 0.9432, Val Loss : 0.3007, Val Acc : 0.9042
Epoch [17/20], Train Loss : 0.1699, Train Acc : 0.9435, Val Loss : 0.3201, Val Acc : 0.8986
Epoch [18/20], Train Loss : 0.1726, Train Acc : 0.9426, Val Loss : 0.2417, Val Acc : 0.9268
Epoch [19/20], Train Loss : 0.1743, Train Acc : 0.9404, Val Loss : 0.2570, Val Acc : 0.9155
Epoch [20/20], Train Loss : 0.1649, Train Acc : 0.9463, Val Loss : 0.2431, Val Acc : 0.9380
"""

"""
model = resnet18(weights=ResNet18_Weights.DEFAULT)
batch_size=64
epochs = 20 
criterion = CrossEntropyLoss().to(device)
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)

Epoch [1/20], Train Loss : 1.1258, Train Acc : 0.6332, Val Loss : 0.7497, Val Acc : 0.8282
Epoch [2/20], Train Loss : 0.5551, Train Acc : 0.8240, Val Loss : 0.4579, Val Acc : 0.8507
Epoch [3/20], Train Loss : 0.4968, Train Acc : 0.8488, Val Loss : 0.4418, Val Acc : 0.8845
Epoch [4/20], Train Loss : 0.4629, Train Acc : 0.8463, Val Loss : 0.4549, Val Acc : 0.8845
Epoch [5/20], Train Loss : 0.4270, Train Acc : 0.8566, Val Loss : 0.4616, Val Acc : 0.8676
Epoch [6/20], Train Loss : 0.3965, Train Acc : 0.8742, Val Loss : 0.3559, Val Acc : 0.8930
Epoch [7/20], Train Loss : 0.3865, Train Acc : 0.8770, Val Loss : 0.3582, Val Acc : 0.8789
Epoch [8/20], Train Loss : 0.3908, Train Acc : 0.8789, Val Loss : 0.4114, Val Acc : 0.8648
Epoch [9/20], Train Loss : 0.3563, Train Acc : 0.8792, Val Loss : 0.5503, Val Acc : 0.8479
Epoch [10/20], Train Loss : 0.3317, Train Acc : 0.8914, Val Loss : 0.4157, Val Acc : 0.8873
Epoch [11/20], Train Loss : 0.3248, Train Acc : 0.9002, Val Loss : 0.4932, Val Acc : 0.8704
Epoch [12/20], Train Loss : 0.2948, Train Acc : 0.9059, Val Loss : 0.4520, Val Acc : 0.8648
Epoch [13/20], Train Loss : 0.3162, Train Acc : 0.8974, Val Loss : 0.3971, Val Acc : 0.8901
Epoch [14/20], Train Loss : 0.2786, Train Acc : 0.9109, Val Loss : 0.3519, Val Acc : 0.9099
Epoch [15/20], Train Loss : 0.3060, Train Acc : 0.9062, Val Loss : 0.3317, Val Acc : 0.9014
Epoch [16/20], Train Loss : 0.2791, Train Acc : 0.9052, Val Loss : 0.4013, Val Acc : 0.8845
Epoch [17/20], Train Loss : 0.2771, Train Acc : 0.9115, Val Loss : 0.4200, Val Acc : 0.8789
Epoch [18/20], Train Loss : 0.2718, Train Acc : 0.9090, Val Loss : 0.4013, Val Acc : 0.8873
Epoch [19/20], Train Loss : 0.2779, Train Acc : 0.9121, Val Loss : 0.3836, Val Acc : 0.9042
Epoch [20/20], Train Loss : 0.2506, Train Acc : 0.9165, Val Loss : 0.4414, Val Acc : 0.8873
"""

"""
model = resnet50(weights=ResNet50_Weights.DEFAULT)
batch_size=128
epochs = 20 
criterion = CrossEntropyLoss().to(device)
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)

Epoch [1/20], Train Loss : 1.7540, Train Acc : 0.4292, Val Loss : 1.4096, Val Acc : 0.5296
Epoch [2/20], Train Loss : 0.7450, Train Acc : 0.7813, Val Loss : 0.6474, Val Acc : 0.8254
Epoch [3/20], Train Loss : 0.4746, Train Acc : 0.8444, Val Loss : 0.4106, Val Acc : 0.8704
Epoch [4/20], Train Loss : 0.3736, Train Acc : 0.8795, Val Loss : 0.3783, Val Acc : 0.8789
Epoch [5/20], Train Loss : 0.3116, Train Acc : 0.8952, Val Loss : 0.3063, Val Acc : 0.9070
Epoch [6/20], Train Loss : 0.2763, Train Acc : 0.9156, Val Loss : 0.3169, Val Acc : 0.9042
Epoch [7/20], Train Loss : 0.2542, Train Acc : 0.9175, Val Loss : 0.4119, Val Acc : 0.8901
Epoch [8/20], Train Loss : 0.2261, Train Acc : 0.9209, Val Loss : 0.4323, Val Acc : 0.8873
Epoch [9/20], Train Loss : 0.2446, Train Acc : 0.9203, Val Loss : 0.3659, Val Acc : 0.8901
Epoch [10/20], Train Loss : 0.2197, Train Acc : 0.9253, Val Loss : 0.3114, Val Acc : 0.9099
Epoch [11/20], Train Loss : 0.2177, Train Acc : 0.9241, Val Loss : 0.3322, Val Acc : 0.9239
Epoch [12/20], Train Loss : 0.2018, Train Acc : 0.9350, Val Loss : 0.3498, Val Acc : 0.9099
Epoch [13/20], Train Loss : 0.1925, Train Acc : 0.9341, Val Loss : 0.4391, Val Acc : 0.9014
Epoch [14/20], Train Loss : 0.1809, Train Acc : 0.9385, Val Loss : 0.4193, Val Acc : 0.9070
Epoch [15/20], Train Loss : 0.1726, Train Acc : 0.9467, Val Loss : 0.3578, Val Acc : 0.9070
Epoch [16/20], Train Loss : 0.1816, Train Acc : 0.9391, Val Loss : 0.3586, Val Acc : 0.9014
Epoch [17/20], Train Loss : 0.1744, Train Acc : 0.9448, Val Loss : 0.3344, Val Acc : 0.9070
Epoch [18/20], Train Loss : 0.1827, Train Acc : 0.9448, Val Loss : 0.2803, Val Acc : 0.9239
Epoch [19/20], Train Loss : 0.1567, Train Acc : 0.9476, Val Loss : 0.3342, Val Acc : 0.9296
Epoch [20/20], Train Loss : 0.1561, Train Acc : 0.9507, Val Loss : 0.3880, Val Acc : 0.9099
"""