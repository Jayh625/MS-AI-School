import os
import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg11, VGG11_Weights
from torchvision.models import resnet18, ResNet18_Weights
from lion_pytorch import Lion
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Prac01_custom_dataset import CustomDataset
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
            torch.save(model.state_dict(), 'best.pt')
            best_val_acc = val_acc

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss : {train_loss:.4f}, "
              f"Train Acc : {train_acc:.4f}, "
              f"Val Loss : {val_loss:.4f}, "
              f"Val Acc : {val_acc:.4f}")

    return model, train_losses, val_losses, train_accs, val_accs
        

def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg11(weights=VGG11_Weights.DEFAULT)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 3)
    model.to(device)

    # model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 65)
    # model.to(device)

    # transforms - aug, totensor, normalize
    train_transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor()
    ])

    # datasets
    train_dataset = CustomDataset("./data/GTZAN_data/train", transform=train_transform)
    val_dataset = CustomDataset("./data/GTZAN_data/val", transform=val_transform)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=False)

    # num_workers 조절 
    # test = time.time()
    # math.factorial(100000)
    # for data, t in train_loader :
    #     print(data, t)
    # test01 = time.time() 
    # print(f"{test01 - test :.5f} sec")

    # epochs, loss, optimizer
    epochs = 20
    criterion = CrossEntropyLoss().to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
    model, train_losses, val_losses, train_accs, train_accs = train(model, train_loader, val_loader, epochs, device, optimizer, criterion)

    # Plot the traning and validation loss and accuracy curves
    os.makedirs("./result", exist_ok=True)
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.show()
    plt.savefig('./result/Train_Val_loss.png')
    
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_losses, label="Validation Accuracy")
    plt.legend()
    plt.show()
    plt.savefig('./result/Train_Val_Accuracy.png')

if __name__ == "__main__" :
    main()