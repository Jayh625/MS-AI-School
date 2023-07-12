import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2

class HandWriteDataSet(Dataset):
    def __init__(self, img_path, transform=None, mode='train'):
        self.img_root = img_path
        self.img_dir_list = os.listdir(img_path)
        self.transform = transform
        self.mode = mode

    def __getitem__(self, idx):
        filename = self.img_dir_list[idx]
        img = Image.open(os.path.join(self.img_root, 
                                      filename)).convert("L")
        
        if len(filename.split("_")) > 1:
            label = int(filename.split("_")[-1].split(".")[0])
        else:
            label = int(filename.split(".")[0][-1])

        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.img_dir_list)

if __name__ == "__main__":
    dset = HandWriteDataSet("C:\\Users\\user\\Desktop\\msAIschool\\230703\\손글씨_데이터")
    for item in dset:
        pass