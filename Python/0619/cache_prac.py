import torch
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def is_grayscale(img) :
    return img.mode == 'L'
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = glob.glob(os.path.join(image_paths,"*","*","*.jpg"))
        self.transform = transform
        self.label_dict = {"dew":0, "fogsmog":1, "frost":2, "glaze":3, "hail":4, "lightning":5, 
                           "rain":6, "rainbow":7, "rime":8, "sandstorm":9, "snow":10}
        self.cache = {}
    def __getitem__(self, index):
        if index in self.cache:
            image, label = self.cache[index]
        else:
            image_path: str = self.image_paths[index]
            image = Image.open(image_path).convert('RGB')

            if not is_grayscale(image):
                folder_name = image_path.split("\\") 
                folder_name = folder_name[2]
                label = self.label_dict[folder_name]
                self.cache[index] = (image, label)
            else :
                print(f"흑백 이미지 : {image_path}")
                return None, None
            
        if self.transform:
            image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.image_paths)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    image_paths = './sample_data_01/'
    dataset = CustomImageDataset(image_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for images, labels in data_loader:
        print(images, labels)
