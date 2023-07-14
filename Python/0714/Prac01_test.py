import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.functional as F 

from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from Prac01_custom_dataset import CustomDataset
from tqdm import tqdm
import cv2

def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model.to(device)

    # .pt load 
    model.load_state_dict(torch.load(f="./data/art_best.pt"))
    # print(list(model.parameters()))
    
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    
    test_dataset = CustomDataset("./data/Paintings Image/val/", transform=val_transforms)
    # for i in test_dataset :
    #     print(i)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # for image, label in test_loader :
    #     print(image, label)

    model.to(device)
    model.eval()
    correct = 0 

    label_dict = {0 : "Abstract", 1 : "Cubist", 2 : "Expressionist",
                  3 : "Impressionist", 4 : "Landscape", 5 : "Pop Art",
                  6 : "Portrait", 7 : "Realist", 8 : "Still Life",
                  9 : "Surrealist"}
    
    with torch.no_grad() :
        for data, target, path in test_loader :
            target_ = target.item()
            data, target = data.to(device), target.to(device)
            output = model(data) # tensor([[ 1.3802, -6.1765,  7.7883,  3.9245, -1.4606, -2.7372, -5.4860, -5.9663, -3.0546, -6.6901]], device='cuda:0')
            # print(output.cpu().numpy()) # 라벨마다 일치하는 확률 -> 값 뽑아내기위해서 numpy로 변환
            pred = output.argmax(dim=1, keepdim=True)
            # print(f"pred : {pred.item()} {path}")
            img = cv2.imread(path[0])
            img = cv2.resize(img, (500,500))

            target_label = label_dict[target_]
            target_text = f"target : {target_label}"
            
            pred_label = label_dict[pred.item()]
            pred_text = f"pred : {pred_label}"

            img = cv2.rectangle(img, (0,0), (500,100), (0,0,0), -1)
            img = cv2.putText(img, pred_text, (0,30), cv2.FONT_ITALIC, 1, (255,255,255), 2)
            img = cv2.putText(img, target_text, (0, 75), cv2.FONT_ITALIC, 1, (255,255,255), 2)
            cv2.imshow("test", img)
            if cv2.waitKey(0) & 0xff == ord("q") :
                exit()
            correct += pred.eq(target.view_as(pred)).sum().item()
    print("Test Set : Acc [{}/{}] {:.0f}%".format(correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))            

if __name__ == "__main__" :
    main()