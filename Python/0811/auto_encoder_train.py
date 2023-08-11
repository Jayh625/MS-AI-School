# 오토 인코더 (MNIST) 예제를 통한 오토인코더의 학습
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader
from auto_encoder_model import AutoEncoder

# hyperparameter
batch_size = 256
lr = 0.001
num_epochs = 100 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transforms
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# dataset
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# model load
autoencoder = AutoEncoder().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=1e-4)

# train
for epoch in range(num_epochs) :
    start_time = time.time()
    for data in train_loader :
        img, _ = data
        img = img.to(device)
        img = img.view(img.size(0), -1).to(device)
        optimizer.zero_grad()
        outputs = autoencoder(img)
        loss = criterion(outputs, img)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"EPOCH [{epoch + 1} / {num_epochs}],"
          f" Loss : {loss.item():.4f} epoch time : {epoch_time:.2f} seconds")
torch.save(autoencoder.state_dict(), 'autoencoder_model.pt')