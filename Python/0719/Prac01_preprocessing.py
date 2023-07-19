import os
import glob
import shutil
from sklearn.model_selection import train_test_split

class ImagePreprocessing : 
    def __init__(self, src, dst) :
        self.src = src
        self.dst = dst
        folders = os.listdir(self.src)
        for folder in folders : 
            os.makedirs(os.path.join('./data/data/', folder), exist_ok=True)
            
        for folder in folders : 
            os.makedirs(os.path.join(self.dst, "train", folder), exist_ok=True)
            os.makedirs(os.path.join(self.dst, "val", folder), exist_ok=True)

        self.file_copy()
        self.file_rename()
        self.file_split()
        self.folder_delete()

    def file_copy(self) :
        files = glob.glob(os.path.join(self.src, "*", "*"))
        for file in files :
            folder_path = file.split("\\")[1]
            file_name = file.split("\\")[2]
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')) :
                shutil.copy2(file, os.path.join('./data/data', folder_path))
        print("File Copied")

    def file_rename(self) :
        files = glob.glob(os.path.join("./data/data", "*", "*"))
        for file in files :
            folder_path = file.split("\\")[1]
            file_name = file.split("\\")[2]
            file_type = file.split("\\")[2].split('.')
            os.rename(file, os.path.join("./data/data", folder_path, file_type[0]+'.png'))
        print("Renaming Files Completed")
                         
    def file_split(self) :
        files = glob.glob(os.path.join("./data/data", "*", "*"))
        train_list, val_list = train_test_split(files, test_size=0.2)
        for file in train_list : 
            folder_path = file.split("\\")[1]
            shutil.copy2(file, os.path.join(self.dst, "train", folder_path))
        for file in val_list : 
            folder_path = file.split("\\")[1]
            shutil.copy2(file, os.path.join(self.dst, "val", folder_path))
        print("File Split Completed") 

    def folder_delete(self) :
        shutil.rmtree("./data/data")
        shutil.rmtree(self.src)
        print("Deleted Folders")

src = "./data/pneumonia_dataset/"
dst = "./data/pneumonia_data/"
file = ImagePreprocessing(src, dst)