import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.functional as F 

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

from Prac02_custom_dataset import CustomDataset
from tqdm import tqdm
from lion_pytorch import Lion
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import os 
import argparse

class Food_Classifier():
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
                
            self.plot_loss()
            self.plot_accuracy()

            # save the model state and optimizer state to checkpoint after each epoch\
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
        # self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        # self.model.heads = nn.Sequential(nn.Linear(768, 251))
        # self.model.to(self.device)
        
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
        self.model.classifier[1] = nn.Linear(1280, out_features=251)
        self.model.to(self.device)

        train_transform = transforms.Compose([
            transforms.Resize((255,255)),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomVerticalFlip(p=0.4),
            transforms.RandomRotation(degrees=15),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.2,0.2,0.2))
        ])
        val_transform = transforms.Compose([
            transforms.Resize((255,255)),
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
    parser.add_argument("--train_dir", type=str, default="./data/food_dataset/train", 
                        help="directory path to the training dataset")
    parser.add_argument("--val_dir", type=str, default="./data/food_dataset/val", 
                        help="directory path to the valid dataset")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=115, 
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

    classifier = Food_Classifier()
    classifier.run(args)

    """
    python Prac02_train.py --batch_size=115
    python Prac02_train.py --resume_training
    """

"""
effnetb0
Epoch [1/100], Train Loss : 3.2748, Train Acc : 0.2964, Val Loss : 1.7528, Val Acc : 0.5589
Epoch [2/100], Train Loss : 2.3538, Train Acc : 0.4525, Val Loss : 1.5252, Val Acc : 0.6117
Epoch [3/100], Train Loss : 2.1069, Train Acc : 0.5005, Val Loss : 1.4201, Val Acc : 0.6321
Epoch [4/100], Train Loss : 1.9477, Train Acc : 0.5307, Val Loss : 1.3575, Val Acc : 0.6457
Epoch [5/100], Train Loss : 1.8236, Train Acc : 0.5572, Val Loss : 1.3516, Val Acc : 0.6498
Epoch [6/100], Train Loss : 1.7192, Train Acc : 0.5767, Val Loss : 1.3146, Val Acc : 0.6606
Epoch [7/100], Train Loss : 1.6270, Train Acc : 0.5961, Val Loss : 1.3060, Val Acc : 0.6622
Epoch [8/100], Train Loss : 1.5374, Train Acc : 0.6147, Val Loss : 1.3051, Val Acc : 0.6634
Epoch [9/100], Train Loss : 1.4645, Train Acc : 0.6293, Val Loss : 1.3209, Val Acc : 0.6680
Epoch [10/100], Train Loss : 1.4003, Train Acc : 0.6400, Val Loss : 1.3180, Val Acc : 0.6693
Epoch [11/100], Train Loss : 1.3343, Train Acc : 0.6544, Val Loss : 1.3281, Val Acc : 0.6688
Epoch [12/100], Train Loss : 1.2704, Train Acc : 0.6670, Val Loss : 1.3362, Val Acc : 0.6699
Epoch [13/100], Train Loss : 1.2173, Train Acc : 0.6789, Val Loss : 1.3540, Val Acc : 0.6668
Epoch [14/100], Train Loss : 1.1668, Train Acc : 0.6916, Val Loss : 1.3766, Val Acc : 0.6700
Epoch [15/100], Train Loss : 1.1181, Train Acc : 0.7009, Val Loss : 1.3903, Val Acc : 0.6661
Epoch [16/100], Train Loss : 1.0690, Train Acc : 0.7118, Val Loss : 1.4161, Val Acc : 0.6671
Epoch [17/100], Train Loss : 1.0232, Train Acc : 0.7224, Val Loss : 1.4440, Val Acc : 0.6668
Epoch [18/100], Train Loss : 0.9875, Train Acc : 0.7298, Val Loss : 1.4574, Val Acc : 0.6629
Epoch [19/100], Train Loss : 0.9509, Train Acc : 0.7375, Val Loss : 1.4767, Val Acc : 0.6617
Epoch [20/100], Train Loss : 0.9088, Train Acc : 0.7468, Val Loss : 1.4953, Val Acc : 0.6615
Epoch [21/100], Train Loss : 0.8785, Train Acc : 0.7565, Val Loss : 1.5369, Val Acc : 0.6618
"""