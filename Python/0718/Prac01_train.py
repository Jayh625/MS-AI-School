import torch
import torch.nn as nn
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.optim import AdamW
from lion_pytorch import Lion
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from Prac01_custom_dataset import MyFoodDataset
import pandas as pd
import matplotlib.pyplot as plt
import os, glob
from datetime import datetime

def make_dir_by_time () :
    now = datetime.now()
    now = str(now) 
    now = now.split(".")[0] 
    now = now.replace("-","").replace(" ","_").replace(":","")
    dir = os.path.join("./result1", now)
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
            data = data.to(device).float()
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
                data = data.to(device).float()
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
    
    df = pd.DataFrame({
        'Train Loss' : train_losses,
        'Train Accuracy' : train_accs,
        'Validation Loss' : val_losses,
        'Validation Accuracy' : val_accs
    })
    df.to_csv(os.path.join(result_dir, 'train_val_results.csv'), index=False)

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
    
    return model, train_losses, val_losses, train_accs, val_accs

def main() :
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    result_dir = make_dir_by_time()
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = nn.Linear(1280, 20)
    model.to(device)

    # Albumentation augmentation
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=250),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.6),
        A.RandomShadow(),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4),
        A.RandomBrightnessContrast(p=0.5),
        A.Resize(224,224),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=250),
        A.Resize(224,224),
        ToTensorV2()
    ])

    train_dataset = MyFoodDataset("./data/food_dataset/train", transform=train_transform)
    val_dataset = MyFoodDataset("./data/food_dataset/val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=False)

    epochs = 20 
    criterion = CrossEntropyLoss().to(device)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    model, train_losses, val_losses, train_accs, val_accs = train(model, train_loader, val_loader, epochs, device, optimizer, criterion, result_dir)

if __name__ == "__main__" :
    main()