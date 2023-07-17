import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.functional as F 
import torch.optim as optim

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from Prac01_custom_dataset import CustomDataset
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
    dir = os.path.join("./result2", now)
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
            for data, target,_ in val_loader :
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
    # model.classifier[1] = nn.Linear(num_features, 15)
    # model.to(device)

    # model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 15)
    # model.to(device)

    # model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # num_features = 1280
    # model.classifier[1] = nn.Linear(num_features, 15)
    # model.to(device)

    # model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    # num_features = 1280
    # model.classifier[1] = nn.Linear(num_features, 15)
    # model.to(device)

    # 실험을 위한 resnet50, pt 파일
    # 1:07
    # Train Loss : 1.0988, Train Acc : 0.6943, Val Loss : 0.3268, Val Acc : 0.9200
    # model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 15)
    # model.to(device)
    
    # 1:03
    # Train Loss : 0.3405, Train Acc : 0.9044, Val Loss : 0.1096, Val Acc : 0.9615
    # model = resnet50(weights=None)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 15)
    # model.load_state_dict(torch.load(f="./data/ex01_0714_resnet50_epoch_30.pt"))
    # model.to(device)

    # 1:04
    # Train Loss : 0.3609, Train Acc : 0.8996, Val Loss : 0.0667, Val Acc : 0.9778
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 15)
    model.load_state_dict(torch.load(f="./data/ex01_0714_resnet50_epoch_30.pt"))
    model.to(device)
    

    train_transform = transforms.Compose([
        transforms.CenterCrop((244,244)),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(degrees=15),
        transforms.RandAugment(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.CenterCrop((244,244)),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset("./data/dataset/train", transform=train_transform)
    val_dataset = CustomDataset("./data/dataset/val", transform=val_transform)

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
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)

Epoch [1/20], Train Loss : 2.0913, Train Acc : 0.4756, Val Loss : 1.1719, Val Acc : 0.7304
Epoch [2/20], Train Loss : 0.7510, Train Acc : 0.8021, Val Loss : 0.3829, Val Acc : 0.8830
Epoch [3/20], Train Loss : 0.2921, Train Acc : 0.9091, Val Loss : 0.1879, Val Acc : 0.9393
Epoch [4/20], Train Loss : 0.1417, Train Acc : 0.9544, Val Loss : 0.1427, Val Acc : 0.9526
Epoch [5/20], Train Loss : 0.1104, Train Acc : 0.9635, Val Loss : 0.1160, Val Acc : 0.9630
Epoch [6/20], Train Loss : 0.0932, Train Acc : 0.9719, Val Loss : 0.1516, Val Acc : 0.9615
Epoch [7/20], Train Loss : 0.0731, Train Acc : 0.9768, Val Loss : 0.1189, Val Acc : 0.9689
Epoch [8/20], Train Loss : 0.0700, Train Acc : 0.9774, Val Loss : 0.0956, Val Acc : 0.9733
Epoch [9/20], Train Loss : 0.0684, Train Acc : 0.9804, Val Loss : 0.1222, Val Acc : 0.9659
Epoch [10/20], Train Loss : 0.0692, Train Acc : 0.9794, Val Loss : 0.0962, Val Acc : 0.9719
Epoch [11/20], Train Loss : 0.0588, Train Acc : 0.9816, Val Loss : 0.1182, Val Acc : 0.9689
Epoch [12/20], Train Loss : 0.0597, Train Acc : 0.9804, Val Loss : 0.1208, Val Acc : 0.9674
Epoch [13/20], Train Loss : 0.0606, Train Acc : 0.9806, Val Loss : 0.1230, Val Acc : 0.9630
Epoch [14/20], Train Loss : 0.0554, Train Acc : 0.9824, Val Loss : 0.1269, Val Acc : 0.9719
Epoch [15/20], Train Loss : 0.0542, Train Acc : 0.9827, Val Loss : 0.1306, Val Acc : 0.9704
Epoch [16/20], Train Loss : 0.0550, Train Acc : 0.9829, Val Loss : 0.1545, Val Acc : 0.9644
Epoch [17/20], Train Loss : 0.0536, Train Acc : 0.9834, Val Loss : 0.1464, Val Acc : 0.9719
Epoch [18/20], Train Loss : 0.0485, Train Acc : 0.9840, Val Loss : 0.1229, Val Acc : 0.9689
Epoch [19/20], Train Loss : 0.0453, Train Acc : 0.9865, Val Loss : 0.1150, Val Acc : 0.9748
Epoch [20/20], Train Loss : 0.0498, Train Acc : 0.9860, Val Loss : 0.1239, Val Acc : 0.9630
"""

"""
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

Epoch [1/20], Train Loss : 2.2877, Train Acc : 0.4132, Val Loss : 1.6579, Val Acc : 0.6800
Epoch [2/20], Train Loss : 1.1751, Train Acc : 0.7490, Val Loss : 0.6659, Val Acc : 0.8341
Epoch [3/20], Train Loss : 0.5676, Train Acc : 0.8635, Val Loss : 0.3833, Val Acc : 0.8919
Epoch [4/20], Train Loss : 0.3391, Train Acc : 0.9119, Val Loss : 0.2498, Val Acc : 0.9244
Epoch [5/20], Train Loss : 0.2278, Train Acc : 0.9424, Val Loss : 0.1869, Val Acc : 0.9422
Epoch [6/20], Train Loss : 0.1782, Train Acc : 0.9541, Val Loss : 0.1733, Val Acc : 0.9511
Epoch [7/20], Train Loss : 0.1431, Train Acc : 0.9633, Val Loss : 0.1450, Val Acc : 0.9556
Epoch [8/20], Train Loss : 0.1121, Train Acc : 0.9700, Val Loss : 0.1242, Val Acc : 0.9704
Epoch [9/20], Train Loss : 0.1010, Train Acc : 0.9720, Val Loss : 0.1181, Val Acc : 0.9659
Epoch [10/20], Train Loss : 0.0782, Train Acc : 0.9799, Val Loss : 0.1071, Val Acc : 0.9704
Epoch [11/20], Train Loss : 0.0713, Train Acc : 0.9822, Val Loss : 0.1053, Val Acc : 0.9689
Epoch [12/20], Train Loss : 0.0665, Train Acc : 0.9824, Val Loss : 0.1025, Val Acc : 0.9719
Epoch [13/20], Train Loss : 0.0566, Train Acc : 0.9852, Val Loss : 0.1015, Val Acc : 0.9689
Epoch [14/20], Train Loss : 0.0545, Train Acc : 0.9857, Val Loss : 0.0972, Val Acc : 0.9674
Epoch [15/20], Train Loss : 0.0435, Train Acc : 0.9891, Val Loss : 0.0926, Val Acc : 0.9748
Epoch [16/20], Train Loss : 0.0442, Train Acc : 0.9898, Val Loss : 0.0909, Val Acc : 0.9689
Epoch [17/20], Train Loss : 0.0441, Train Acc : 0.9893, Val Loss : 0.0924, Val Acc : 0.9704
Epoch [18/20], Train Loss : 0.0387, Train Acc : 0.9903, Val Loss : 0.0835, Val Acc : 0.9763
Epoch [19/20], Train Loss : 0.0361, Train Acc : 0.9916, Val Loss : 0.0825, Val Acc : 0.9807
Epoch [20/20], Train Loss : 0.0286, Train Acc : 0.9929, Val Loss : 0.0913, Val Acc : 0.9733
"""

"""
result2/20230714_155028
model = resnet18(weights=ResNet18_Weights.DEFAULT)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
Epoch [1/20], Train Loss : 1.2311, Train Acc : 0.6486, Val Loss : 0.3123, Val Acc : 0.9037
Epoch [2/20], Train Loss : 0.3678, Train Acc : 0.8958, Val Loss : 0.1976, Val Acc : 0.9437
Epoch [3/20], Train Loss : 0.2235, Train Acc : 0.9371, Val Loss : 0.2072, Val Acc : 0.9348
Epoch [4/20], Train Loss : 0.1511, Train Acc : 0.9612, Val Loss : 0.1976, Val Acc : 0.9437
Epoch [5/20], Train Loss : 0.1205, Train Acc : 0.9667, Val Loss : 0.1125, Val Acc : 0.9659
Epoch [6/20], Train Loss : 0.0957, Train Acc : 0.9735, Val Loss : 0.1078, Val Acc : 0.9644
Epoch [7/20], Train Loss : 0.0814, Train Acc : 0.9776, Val Loss : 0.1536, Val Acc : 0.9481
Epoch [8/20], Train Loss : 0.0740, Train Acc : 0.9781, Val Loss : 0.0986, Val Acc : 0.9689
Epoch [9/20], Train Loss : 0.0558, Train Acc : 0.9873, Val Loss : 0.0779, Val Acc : 0.9763
Epoch [10/20], Train Loss : 0.0526, Train Acc : 0.9857, Val Loss : 0.0742, Val Acc : 0.9763
Epoch [11/20], Train Loss : 0.0512, Train Acc : 0.9863, Val Loss : 0.0923, Val Acc : 0.9748
Epoch [12/20], Train Loss : 0.0505, Train Acc : 0.9850, Val Loss : 0.0892, Val Acc : 0.9748
Epoch [13/20], Train Loss : 0.0555, Train Acc : 0.9835, Val Loss : 0.0960, Val Acc : 0.9748
Epoch [14/20], Train Loss : 0.0417, Train Acc : 0.9880, Val Loss : 0.0863, Val Acc : 0.9778
Epoch [15/20], Train Loss : 0.0383, Train Acc : 0.9898, Val Loss : 0.0732, Val Acc : 0.9748
Epoch [16/20], Train Loss : 0.0324, Train Acc : 0.9918, Val Loss : 0.0679, Val Acc : 0.9763
Epoch [17/20], Train Loss : 0.0325, Train Acc : 0.9916, Val Loss : 0.0974, Val Acc : 0.9748
Epoch [18/20], Train Loss : 0.0342, Train Acc : 0.9903, Val Loss : 0.0937, Val Acc : 0.9748
Epoch [19/20], Train Loss : 0.0313, Train Acc : 0.9918, Val Loss : 0.0925, Val Acc : 0.9748
Epoch [20/20], Train Loss : 0.0259, Train Acc : 0.9936, Val Loss : 0.0931, Val Acc : 0.9733
"""

"""
result2/20230714_161238
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
Epoch [1/20], Train Loss : 1.9620, Train Acc : 0.4899, Val Loss : 0.7870, Val Acc : 0.8030
Epoch [2/20], Train Loss : 0.5300, Train Acc : 0.8632, Val Loss : 0.2054, Val Acc : 0.9348
Epoch [3/20], Train Loss : 0.1750, Train Acc : 0.9437, Val Loss : 0.1137, Val Acc : 0.9570
Epoch [4/20], Train Loss : 0.1162, Train Acc : 0.9630, Val Loss : 0.1007, Val Acc : 0.9704
Epoch [5/20], Train Loss : 0.0983, Train Acc : 0.9700, Val Loss : 0.0960, Val Acc : 0.9733
Epoch [6/20], Train Loss : 0.0847, Train Acc : 0.9735, Val Loss : 0.0808, Val Acc : 0.9674
Epoch [7/20], Train Loss : 0.0794, Train Acc : 0.9722, Val Loss : 0.1024, Val Acc : 0.9763
Epoch [8/20], Train Loss : 0.0626, Train Acc : 0.9789, Val Loss : 0.1245, Val Acc : 0.9733
Epoch [9/20], Train Loss : 0.0620, Train Acc : 0.9817, Val Loss : 0.1145, Val Acc : 0.9733
Epoch [10/20], Train Loss : 0.0646, Train Acc : 0.9788, Val Loss : 0.0731, Val Acc : 0.9748
Epoch [11/20], Train Loss : 0.0550, Train Acc : 0.9816, Val Loss : 0.0554, Val Acc : 0.9807
Epoch [12/20], Train Loss : 0.0553, Train Acc : 0.9812, Val Loss : 0.0748, Val Acc : 0.9704
Epoch [13/20], Train Loss : 0.0548, Train Acc : 0.9834, Val Loss : 0.0737, Val Acc : 0.9793
Epoch [14/20], Train Loss : 0.0497, Train Acc : 0.9829, Val Loss : 0.1151, Val Acc : 0.9778
Epoch [15/20], Train Loss : 0.0444, Train Acc : 0.9862, Val Loss : 0.0893, Val Acc : 0.9674
Epoch [16/20], Train Loss : 0.0410, Train Acc : 0.9858, Val Loss : 0.0802, Val Acc : 0.9778
Epoch [17/20], Train Loss : 0.0395, Train Acc : 0.9877, Val Loss : 0.1095, Val Acc : 0.9689
Epoch [18/20], Train Loss : 0.0457, Train Acc : 0.9855, Val Loss : 0.0988, Val Acc : 0.9763
Epoch [19/20], Train Loss : 0.0462, Train Acc : 0.9865, Val Loss : 0.0944, Val Acc : 0.9793
Epoch [20/20], Train Loss : 0.0425, Train Acc : 0.9873, Val Loss : 0.1298, Val Acc : 0.9748
"""

"""
result2/
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
Epoch [1/20], Train Loss : 1.0023, Train Acc : 0.7284, Val Loss : 0.2022, Val Acc : 0.9407
Epoch [2/20], Train Loss : 0.1816, Train Acc : 0.9417, Val Loss : 0.1158, Val Acc : 0.9674
Epoch [3/20], Train Loss : 0.1382, Train Acc : 0.9585, Val Loss : 0.1205, Val Acc : 0.9659
Epoch [4/20], Train Loss : 0.1167, Train Acc : 0.9631, Val Loss : 0.1214, Val Acc : 0.9704
Epoch [5/20], Train Loss : 0.1106, Train Acc : 0.9640, Val Loss : 0.1209, Val Acc : 0.9719
Epoch [6/20], Train Loss : 0.1034, Train Acc : 0.9676, Val Loss : 0.0864, Val Acc : 0.9763
Epoch [7/20], Train Loss : 0.0929, Train Acc : 0.9695, Val Loss : 0.0855, Val Acc : 0.9704
Epoch [8/20], Train Loss : 0.0943, Train Acc : 0.9727, Val Loss : 0.0947, Val Acc : 0.9719
Epoch [9/20], Train Loss : 0.0805, Train Acc : 0.9738, Val Loss : 0.0716, Val Acc : 0.9807
Epoch [10/20], Train Loss : 0.0739, Train Acc : 0.9751, Val Loss : 0.0633, Val Acc : 0.9807
Epoch [11/20], Train Loss : 0.0654, Train Acc : 0.9791, Val Loss : 0.0821, Val Acc : 0.9733
Epoch [12/20], Train Loss : 0.0629, Train Acc : 0.9778, Val Loss : 0.0772, Val Acc : 0.9793
Epoch [13/20], Train Loss : 0.0728, Train Acc : 0.9765, Val Loss : 0.0851, Val Acc : 0.9763
Epoch [14/20], Train Loss : 0.0694, Train Acc : 0.9776, Val Loss : 0.0752, Val Acc : 0.9793
Epoch [15/20], Train Loss : 0.0612, Train Acc : 0.9806, Val Loss : 0.1210, Val Acc : 0.9704
Epoch [16/20], Train Loss : 0.0584, Train Acc : 0.9819, Val Loss : 0.0907, Val Acc : 0.9793
Epoch [17/20], Train Loss : 0.0488, Train Acc : 0.9829, Val Loss : 0.1448, Val Acc : 0.9600
Epoch [18/20], Train Loss : 0.0681, Train Acc : 0.9804, Val Loss : 0.0915, Val Acc : 0.9778
Epoch [19/20], Train Loss : 0.0474, Train Acc : 0.9857, Val Loss : 0.0726, Val Acc : 0.9822
Epoch [20/20], Train Loss : 0.0469, Train Acc : 0.9863, Val Loss : 0.0676, Val Acc : 0.9852
"""