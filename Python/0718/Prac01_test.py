import torch
import torch.nn as nn
import torch.functional as F 
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from Prac01_custom_dataset import MyFoodDataset
from tqdm import tqdm
import cv2
import os

def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = nn.Linear(1280, 20)
    model.to(device)
    model.load_state_dict(torch.load(f="./result1/20230718_102804/best.pt"))
    
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=250),
        A.Resize(224,224),
        ToTensorV2()
    ])
    
    test_dataset = MyFoodDataset("./data/food_dataset/test/", transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()
    correct = 0 
    # data_dir = "./data/food_dataset/test/"
    # dict = {}
    # folders = os.listdir(data_dir)
    # for i, folder in enumerate(folders) :
    #     dict[folder] = i
    # label_dict = {v: k for k, v in dict.items()}
    # os.makedirs("./result1/imgs", exist_ok=True)
    with torch.no_grad() :
        for i, (data, target, _) in tqdm(enumerate(test_loader)) :
            target_ = target.item()
            data, target = data.to(device).float(), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            # img = cv2.imread(path[0])
            # img = cv2.resize(img, (500,500))

            # target_label = label_dict[target_]
            # target_text = f"target : {target_label}"
            
            # pred_label = label_dict[pred.item()]
            # pred_text = f"pred : {pred_label}"

            # img = cv2.rectangle(img, (0,0), (500,100), (0,0,0), -1)
            # img = cv2.putText(img, pred_text, (0,30), cv2.FONT_ITALIC, 1, (255,255,255), 2)
            # img = cv2.putText(img, target_text, (0, 75), cv2.FONT_ITALIC, 1, (255,255,255), 2)
            # cv2.imwrite(os.path.join("./result1/imgs", f"output_{str(i).zfill(4)}.png"),img)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print("Test Set : Acc [{}/{}] {:.0f}%".format(correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))            

if __name__ == "__main__" :
    main()
"""
Test Set : Acc [945/997] 95%
"""