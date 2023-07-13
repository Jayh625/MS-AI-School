import os
import glob
import shutil
from sklearn.model_selection import train_test_split

class ImageMove : 
    def __init__(self, src, dst) :
        self.src = src
        self.dst = dst
        os.makedirs(self.dst, exist_ok=True)
        os.makedirs(os.path.join(self.dst, "MelSepctrogram"), exist_ok=True)
        os.makedirs(os.path.join(self.dst, "STFT"), exist_ok=True)
        os.makedirs(os.path.join(self.dst, "waveshow"), exist_ok=True)
    
    def move_images(self) :
        files = glob.glob(os.path.join(self.src, "*", "*", "*.png"))
        for file in files :
            folder_name = file.split("\\")[1]
            if folder_name == "MelSepctrogram" :
                shutil.move(file, os.path.join(self.dst, "MelSepctrogram"))
            elif folder_name == "STFT" :
                shutil.move(file, os.path.join(self.dst, "STFT"))
            elif folder_name == "waveshow" :
                shutil.move(file, os.path.join(self.dst, "waveshow"))

    def file_split(self, name) :
        files = glob.glob(os.path.join("./data/data", name, "*.png"))
        train_list, val_list = train_test_split(files, test_size=0.2)
        for file in train_list :
            os.makedirs(os.path.join("./data/GTZAN_data/train", name), exist_ok=True)
            shutil.move(file, os.path.join("./data/GTZAN_data/train", name))
        for file in val_list :
            os.makedirs(os.path.join("./data/GTZAN_data/val", name), exist_ok=True)
            shutil.move(file, os.path.join("./data/GTZAN_data/val", name))
            
    def __del__(self):
        shutil.rmtree(self.src)
        shutil.rmtree(self.dst)

src = "./data/final_data"
dst = "./data/data"

move = ImageMove(src, dst)
move.move_images()

move.file_split("MelSepctrogram")
move.file_split("STFT")
move.file_split("waveshow")

del move