import os
import random
import shutil
import glob
from sklearn.model_selection import train_test_split

src = "./data/data"
dst = "./data/dataset"

# train, val folder_path 
train_path = os.path.join(dst, "train")
val_path = os.path.join(dst, "val")
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

folders = os.listdir(src)
for folder in folders :
    full_path = os.path.join(src, folder)
    images = os.listdir(full_path)
    random.shuffle(images) # image random shuffle
    train = os.path.join(train_path, folder)
    val = os.path.join(val_path, folder)
    os.makedirs(train, exist_ok=True)
    os.makedirs(val, exist_ok=True)

    # image -> train folder copy
    split_index = int(len(images) * 0.9)
    for image in images[:split_index] :
        src_path = os.path.join(full_path, image)
        dst_path = os.path.join(train, image)
        shutil.copyfile(src_path, dst_path)

    for image in images[split_index:] :
        src_path = os.path.join(full_path, image)
        dst_path = os.path.join(val, image)
        shutil.copyfile(src_path, dst_path)
        
print("Finished")
