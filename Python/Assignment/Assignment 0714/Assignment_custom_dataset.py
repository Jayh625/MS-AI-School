import os
import glob
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDataset(Dataset) :
    def __init__(self, data_dir, transform=None):
        # data_dir = ./data_art/train/
        self.data_dir = glob.glob(os.path.join(data_dir, "*", "*.png"))
        self.transform = transform
        self.label_dict = {}
        folders = os.listdir(data_dir)
        for i, folder in enumerate(folders) :
            self.label_dict[folder] = i

    def __getitem__(self, item):
        image_path = self.data_dir[item]
        image = Image.open(image_path)
        image = image.convert("RGB")
        label_name = image_path.split("\\")[1]
        label = self.label_dict[label_name]

        if self.transform is not None :
            image = self.transform(image)

        return image ,label, image_path

    def __len__(self):
        return len(self.data_dir)