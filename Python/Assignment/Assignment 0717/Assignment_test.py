import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.functional as F 

from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from Assignment_custom_dataset import CustomDataset
from tqdm import tqdm
import cv2
import os

def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    num_features = 1280
    model.classifier[1] = nn.Linear(num_features, 10)
    model.to(device)
    model.load_state_dict(torch.load(f="./assignment/20230717_152931/best.pt"))
    
    val_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.2,0.2,0.2))
    ])
    
    test_dataset = CustomDataset("./data/metal_damaged/val/", transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()
    correct = 0 
    data_dir = "./data/metal_damaged/val/"
    dict = {}
    folders = os.listdir(data_dir)
    for i, folder in enumerate(folders) :
        dict[folder] = i
    label_dict = {v: k for k, v in dict.items()}
    os.makedirs("./assignment/imgs", exist_ok=True)
    with torch.no_grad() :
        for i, (data, target, path) in tqdm(enumerate(test_loader)) :
            target_ = target.item()
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            img = cv2.imread(path[0])
            img = cv2.resize(img, (500,500))

            target_label = label_dict[target_]
            target_text = f"target : {target_label}"
            
            pred_label = label_dict[pred.item()]
            pred_text = f"pred : {pred_label}"

            img = cv2.rectangle(img, (0,0), (500,100), (0,0,0), -1)
            img = cv2.putText(img, pred_text, (0,30), cv2.FONT_ITALIC, 1, (255,255,255), 2)
            img = cv2.putText(img, target_text, (0, 75), cv2.FONT_ITALIC, 1, (255,255,255), 2)
            cv2.imwrite(os.path.join("./assignment/imgs", f"output_{str(i).zfill(4)}.png"),img)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print("Test Set : Acc [{}/{}] {:.0f}%".format(correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))            

if __name__ == "__main__" :
    main()

"""
Test Set : Acc [333/355] 94%
"""