import os
import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from lion_pytorch import Lion
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Assignment_custom_dataset import CustomDataset
import time
import math
from tqdm import tqdm

def train(model, train_loader, val_loader, epochs, device, optimizer, criterion) :
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
        model.train()

        # tqdm
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

            # print the loss
            if i % 10 == 9 : 
                # print(f"Epoch [{epoch+1}/{epochs}], Step[{i+1}/{len(train_loader)}], Loss : {loss.item()}")
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
            torch.save(model.state_dict(), './data/02_best.pt')
            best_val_acc = val_acc

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss : {train_loss:.4f}, "
              f"Train Acc : {train_acc:.4f}, "
              f"Val Loss : {val_loss:.4f}, "
              f"Val Acc : {val_acc:.4f}")

    return model, train_losses, val_losses, train_accs, val_accs
        

def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 65)
    model.to(device)

    # transforms - aug, totensor, normalize
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # datasets
    train_dataset = CustomDataset("./data/Biscuit Wrappers/train", transform=train_transform)
    val_dataset = CustomDataset("./data/Biscuit Wrappers/val", transform=val_transform)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=False)

    # epochs, loss, optimizer
    epochs = 20
    criterion = CrossEntropyLoss().to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
    model, train_losses, val_losses, train_accs, val_accs = train(model, train_loader, val_loader, epochs, device, optimizer, criterion)

    # Plot the traning and validation loss and accuracy curves
    os.makedirs("./result", exist_ok=True)
    
    plt.plot(val_losses, label="Validation loss")
    plt.plot(train_losses, label="Train loss")
    plt.legend()
    plt.savefig('./result/Biscuit Wrappers_Train_Val_loss.png')
    plt.show()
    
    plt.plot(val_accs, label="Validation Accuracy")
    plt.plot(train_accs, label="Train Accuracy")
    plt.legend()
    plt.savefig('./result/Biscuit Wrappers_Train_Val_Accuracy.png')
    plt.show()

if __name__ == "__main__" :
    main()

"""
Epoch [1/20], Train Loss : 3.0886, Train Acc : 0.4369, Val Loss : 1.6460, Val Acc : 0.7928
Epoch [2/20], Train Loss : 0.8262, Train Acc : 0.9568, Val Loss : 0.3360, Val Acc : 0.9730
Epoch [3/20], Train Loss : 0.1652, Train Acc : 0.9914, Val Loss : 0.1009, Val Acc : 0.9970
Epoch [4/20], Train Loss : 0.0463, Train Acc : 0.9985, Val Loss : 0.0459, Val Acc : 0.9955
Epoch [5/20], Train Loss : 0.0230, Train Acc : 0.9981, Val Loss : 0.0514, Val Acc : 0.9925
Epoch [6/20], Train Loss : 0.0193, Train Acc : 0.9974, Val Loss : 0.1041, Val Acc : 0.9820
Epoch [7/20], Train Loss : 0.0181, Train Acc : 0.9977, Val Loss : 0.0556, Val Acc : 0.9895
Epoch [8/20], Train Loss : 0.0117, Train Acc : 0.9985, Val Loss : 0.0488, Val Acc : 0.9895
Epoch [9/20], Train Loss : 0.0143, Train Acc : 0.9974, Val Loss : 0.1156, Val Acc : 0.9835
Epoch [10/20], Train Loss : 0.0095, Train Acc : 0.9992, Val Loss : 0.0892, Val Acc : 0.9865
Epoch [11/20], Train Loss : 0.0121, Train Acc : 0.9981, Val Loss : 0.1213, Val Acc : 0.9745
Epoch [12/20], Train Loss : 0.0076, Train Acc : 0.9989, Val Loss : 0.0861, Val Acc : 0.9865
Epoch [13/20], Train Loss : 0.0171, Train Acc : 0.9962, Val Loss : 0.1149, Val Acc : 0.9790
Epoch [14/20], Train Loss : 0.0157, Train Acc : 0.9974, Val Loss : 0.0985, Val Acc : 0.9775
Epoch [15/20], Train Loss : 0.0151, Train Acc : 0.9966, Val Loss : 0.1005, Val Acc : 0.9760
Epoch [16/20], Train Loss : 0.0238, Train Acc : 0.9962, Val Loss : 0.1235, Val Acc : 0.9760
Epoch [17/20], Train Loss : 0.0114, Train Acc : 0.9970, Val Loss : 0.1026, Val Acc : 0.9745
Epoch [18/20], Train Loss : 0.0172, Train Acc : 0.9962, Val Loss : 0.0948, Val Acc : 0.9850
Epoch [19/20], Train Loss : 0.0130, Train Acc : 0.9959, Val Loss : 0.0988, Val Acc : 0.9805
Epoch [20/20], Train Loss : 0.0141, Train Acc : 0.9970, Val Loss : 0.1034, Val Acc : 0.9805
"""