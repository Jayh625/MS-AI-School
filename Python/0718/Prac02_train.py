import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.functional as F 

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from Prac02_custom_dataset import CustomDataset
from tqdm import tqdm
from lion_pytorch import Lion
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import os 
import argparse

class Classifier_US_LicensePlate():
    def __init__(self) :
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.result_dir = ""
        self.make_dir_by_time()

    def make_dir_by_time(self) :
        now = datetime.now()
        now = str(now) 
        now = now.split(".")[0] 
        now = now.replace("-","").replace(" ","_").replace(":","")
        self.result_dir = os.path.join("./result2", now)
        os.makedirs(self.result_dir, exist_ok=True)
    
    def train(self, train_loader, val_loader, epochs, optimizer, criterion, start_epoch=0) :
        best_val_acc = 0.0
        print("Training...")
        for epoch in range(start_epoch, epochs) :
            train_loss = 0.0
            val_loss = 0.0
            val_acc = 0.0
            train_acc = 0.0
            
            self.model.train()
            train_loader_iter = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
            for i, (data, target, _) in enumerate(train_loader_iter) :
                data = data.to(self.device).float()
                target = target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
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
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # eval
            self.model.eval()
            with torch.no_grad() :
                for data, target, _ in val_loader :
                    data = data.to(self.device).float()
                    target = target.to(self.device)
                    outputs = self.model(data)
                    pred = outputs.argmax(dim=1, keepdim=True)
                    val_acc += pred.eq(target.view_as(pred)).sum().item()
                    val_loss += criterion(outputs, target).item()
            
            val_loss /= len(val_loader)
            val_acc /= len(val_loader.dataset)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            print(f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss : {train_loss:.4f}, "
                f"Train Acc : {train_acc:.4f}, "
                f"Val Loss : {val_loss:.4f}, "
                f"Val Acc : {val_acc:.4f}")

            if val_acc > best_val_acc :
                torch.save(self.model.state_dict(), os.path.join(self.result_dir, 'efficientnet_b0_best.pt'))
                best_val_acc = val_acc

            # save the model state and optimizer state to checkpoint after each epoch
            os.makedirs(os.path.join(self.result_dir, "checkpoint"), exist_ok=True)
            torch.save({
                'epoch' : epoch + 1,
                'model_state_dict' : self.model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'train_losses' : self.train_losses,
                'train_accs' : self.train_accs,
                'val_losses' : self.val_losses,
                'val_accs' : self.val_accs
            }, os.path.join("./result2/checkpoint", "efficientnet_b0_checkpoint.pt"))
    
        torch.save(self.model.state_dict(), os.path.join(self.result_dir, "efficientnet_b0_last.pt"))
        self.save_result_to_csv()
        self.plot_loss()
        self.plot_accuracy()

    def save_result_to_csv(self) :
        df = pd.DataFrame({
        'Train Loss' : self.train_losses,
        'Train Accuracy' : self.train_accs,
        'Validation Loss' : self.val_losses,
        'Validation Accuracy' : self.val_accs
        })
        df.to_csv(os.path.join(self.result_dir, 'efficientnet_b0_results.csv'), index=False)

    def plot_loss(self) :
        plt.figure()
        plt.plot(self.train_losses, label="Train loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.result_dir,'efficientnet_b0_loss.png'))

    def plot_accuracy(self) :
        plt.figure()
        plt.plot(self.train_accs, label="Train Accuracy")
        plt.plot(self.val_accs, label="Validation Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.result_dir,'efficientnet_b0_accuracy.png'))

    def run(self, args) :
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
        self.model.classifier[1] = nn.Linear(1280, out_features=50)
        self.model.to(self.device)
        
        # self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        # num_features = 1280
        # self.model.classifier[1] = nn.Linear(num_features, 50)
        # self.model.to(self.device)

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

        train_dataset = CustomDataset(args.train_dir, transform=train_transform)
        val_dataset = CustomDataset(args.val_dir, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

        start_epoch = 0
        epochs = args.epochs
        criterion = CrossEntropyLoss().to(self.device)
        optimizer = Lion(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # optimizer = AdamW(self.model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)

        if args.resume_training : 
            checkpoint = torch.load(args.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.train_accs = checkpoint['train_accs']
            self.val_losses = checkpoint['val_losses']
            self.val_accs = checkpoint['val_accs']
            start_epoch = checkpoint['epoch']

        self.train(train_loader, val_loader, epochs, optimizer, criterion, start_epoch)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./data/US_license_plates_dataset/train", 
                        help="directory path to the training dataset")
    parser.add_argument("--val_dir", type=str, default="./data/US_license_plates_dataset/valid", 
                        help="directory path to the valid dataset")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="batch size for training and validation")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="number of workers for training and validation")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="weight decay for optimizer")
    parser.add_argument("--resume_training", action="store_true", 
                        help="resume training from the last checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="./result2/checkpoint/efficientnet_b0_checkpoint.pt",
                        help="path to the checkpoint file")
    parser.add_argument("--checkpoint_folder_path", type=str, 
                        default="./result2/checkpoint")
    args = parser.parse_args()
    checkpoint_folder_path = args.checkpoint_folder_path
    os.makedirs(checkpoint_folder_path, exist_ok=True)

    classifier = Classifier_US_LicensePlate()
    classifier.run(args)

    # Prac02_train.py --resume_training


    

"""
./result2/20230718_103859
model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
epochs = 20 
criterion = CrossEntropyLoss().to(device)
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)

Epoch [1/20], Train Loss : 3.3297, Train Acc : 0.2181, Val Loss : 1.9493, Val Acc : 0.5480
Epoch [2/20], Train Loss : 1.4139, Train Acc : 0.6569, Val Loss : 0.7274, Val Acc : 0.8120
Epoch [3/20], Train Loss : 0.6661, Train Acc : 0.8304, Val Loss : 0.4339, Val Acc : 0.9080
Epoch [4/20], Train Loss : 0.4294, Train Acc : 0.8895, Val Loss : 0.3137, Val Acc : 0.9280
Epoch [5/20], Train Loss : 0.3267, Train Acc : 0.9124, Val Loss : 0.2981, Val Acc : 0.9440
Epoch [6/20], Train Loss : 0.2437, Train Acc : 0.9318, Val Loss : 0.2981, Val Acc : 0.9440
Epoch [7/20], Train Loss : 0.2393, Train Acc : 0.9326, Val Loss : 0.2457, Val Acc : 0.9560
Epoch [8/20], Train Loss : 0.1963, Train Acc : 0.9450, Val Loss : 0.3167, Val Acc : 0.9520
Epoch [9/20], Train Loss : 0.1779, Train Acc : 0.9512, Val Loss : 0.2591, Val Acc : 0.9600
Epoch [10/20], Train Loss : 0.1630, Train Acc : 0.9511, Val Loss : 0.2830, Val Acc : 0.9560
Epoch [11/20], Train Loss : 0.1437, Train Acc : 0.9620, Val Loss : 0.2735, Val Acc : 0.9560
Epoch [12/20], Train Loss : 0.1334, Train Acc : 0.9630, Val Loss : 0.3660, Val Acc : 0.9600
Epoch [13/20], Train Loss : 0.1210, Train Acc : 0.9670, Val Loss : 0.3536, Val Acc : 0.9520
Epoch [14/20], Train Loss : 0.1107, Train Acc : 0.9674, Val Loss : 0.3119, Val Acc : 0.9640
Epoch [15/20], Train Loss : 0.1140, Train Acc : 0.9664, Val Loss : 0.3173, Val Acc : 0.9640
Epoch [16/20], Train Loss : 0.1004, Train Acc : 0.9717, Val Loss : 0.2468, Val Acc : 0.9640
Epoch [17/20], Train Loss : 0.1049, Train Acc : 0.9690, Val Loss : 0.2059, Val Acc : 0.9640
Epoch [18/20], Train Loss : 0.0970, Train Acc : 0.9698, Val Loss : 0.2069, Val Acc : 0.9720
Epoch [19/20], Train Loss : 0.0951, Train Acc : 0.9717, Val Loss : 0.2786, Val Acc : 0.9600
Epoch [20/20], Train Loss : 0.0815, Train Acc : 0.9765, Val Loss : 0.2554, Val Acc : 0.9640
"""

"""
Epoch [1/20], Train Loss : 3.8151, Train Acc : 0.0655, Val Loss : 3.4743, Val Acc : 0.2280
Epoch [2/20], Train Loss : 3.0880, Train Acc : 0.2868, Val Loss : 2.2993, Val Acc : 0.4360
Epoch [3/20], Train Loss : 2.1739, Train Acc : 0.4838, Val Loss : 1.5915, Val Acc : 0.6080
Epoch [4/20], Train Loss : 1.4756, Train Acc : 0.6378, Val Loss : 1.1780, Val Acc : 0.7120
Epoch [5/20], Train Loss : 1.0079, Train Acc : 0.7404, Val Loss : 0.8299, Val Acc : 0.7840
Epoch [6/20], Train Loss : 0.6694, Train Acc : 0.8262, Val Loss : 0.6349, Val Acc : 0.8480
Epoch [7/20], Train Loss : 0.4640, Train Acc : 0.8788, Val Loss : 0.5359, Val Acc : 0.8720
Epoch [8/20], Train Loss : 0.3413, Train Acc : 0.9069, Val Loss : 0.4074, Val Acc : 0.9000
Epoch [9/20], Train Loss : 0.2579, Train Acc : 0.9293, Val Loss : 0.3855, Val Acc : 0.9120
Epoch [10/20], Train Loss : 0.2255, Train Acc : 0.9370, Val Loss : 0.3008, Val Acc : 0.9360
Epoch [11/20], Train Loss : 0.1875, Train Acc : 0.9457, Val Loss : 0.3678, Val Acc : 0.9160
Epoch [12/20], Train Loss : 0.1561, Train Acc : 0.9561, Val Loss : 0.3508, Val Acc : 0.9200
Epoch [13/20], Train Loss : 0.1430, Train Acc : 0.9582, Val Loss : 0.3071, Val Acc : 0.9440
Epoch [14/20], Train Loss : 0.1271, Train Acc : 0.9647, Val Loss : 0.3592, Val Acc : 0.9240
Epoch [15/20], Train Loss : 0.1275, Train Acc : 0.9632, Val Loss : 0.3810, Val Acc : 0.9400
Epoch [16/20], Train Loss : 0.1150, Train Acc : 0.9651, Val Loss : 0.3255, Val Acc : 0.9480
Epoch [17/20], Train Loss : 0.1098, Train Acc : 0.9712, Val Loss : 0.3144, Val Acc : 0.9520
Epoch [18/20], Train Loss : 0.1001, Train Acc : 0.9721, Val Loss : 0.3595, Val Acc : 0.9480
Epoch [19/20], Train Loss : 0.0944, Train Acc : 0.9720, Val Loss : 0.3512, Val Acc : 0.9600
Epoch [20/20], Train Loss : 0.0929, Train Acc : 0.9723, Val Loss : 0.3132, Val Acc : 0.9480
"""