import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.functional as F 

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from Prac01_custom_dataset import CustomDataset
from tqdm import tqdm
from lion_pytorch import Lion
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import os 
import argparse

class Sport_Classifier():
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
        self.result_dir = os.path.join("./result1", now)
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
                torch.save(self.model.state_dict(), os.path.join(self.result_dir, 'efficientnet_v2_s_best.pt'))
                best_val_acc = val_acc

            # save the model state and optimizer state to checkpoint after each epoch
            # os.makedirs(os.path.join(self.result_dir, "checkpoint"), exist_ok=True)
            torch.save({
                'epoch' : epoch + 1,
                'model_state_dict' : self.model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'train_losses' : self.train_losses,
                'train_accs' : self.train_accs,
                'val_losses' : self.val_losses,
                'val_accs' : self.val_accs
            }, os.path.join("./result1/checkpoint", "efficientnet_v2_s_checkpoint.pt"))
    
        torch.save(self.model.state_dict(), os.path.join(self.result_dir, "efficientnet_v2_s_last.pt"))
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
        df.to_csv(os.path.join(self.result_dir, 'efficientnet_v2_s_results.csv'), index=False)

    def plot_loss(self) :
        plt.figure()
        plt.plot(self.train_losses, label="Train loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.result_dir,'efficientnet_v2_s_loss.png'))

    def plot_accuracy(self) :
        plt.figure()
        plt.plot(self.train_accs, label="Train Accuracy")
        plt.plot(self.val_accs, label="Validation Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.result_dir,'efficientnet_v2_s_accuracy.png'))

    def run(self, args) :
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        num_features = 1280
        self.model.classifier[1] = nn.Linear(num_features, 100)
        self.model.to(self.device)

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
    parser.add_argument("--train_dir", type=str, default="./data/sport_dataset/train", 
                        help="directory path to the training dataset")
    parser.add_argument("--val_dir", type=str, default="./data/sport_dataset/val", 
                        help="directory path to the valid dataset")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="batch size for training and validation")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="number of workers for training and validation")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="weight decay for optimizer")
    parser.add_argument("--resume_training", action="store_true", 
                        help="resume training from the last checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="./result1/checkpoint/efficientnet_v2_s_checkpoint.pt",
                        help="path to the checkpoint file")
    parser.add_argument("--checkpoint_folder_path", type=str, 
                        default="./result1/checkpoint")
    args = parser.parse_args()
    checkpoint_folder_path = args.checkpoint_folder_path
    os.makedirs(checkpoint_folder_path, exist_ok=True)

    classifier = Sport_Classifier()
    classifier.run(args)