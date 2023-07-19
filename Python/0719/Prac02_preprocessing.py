import os
import cv2
import csv
from PIL import Image
import glob
from tqdm import tqdm
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def csv_index_to_label(csv_file) :
    label_dict = {}
    with open("./data/food_dataset/class_list.txt") as f:
        for line in f:
            (key, val) = line.split()
            label_dict[int(key)] = val
    df = pd.read_csv(csv_file)
    labels = df['label']
    for i, label in enumerate(labels) :
        df.loc[i, 'label'] = label_dict[label]
    df.to_csv(csv_file, index=False)    

def csv_label_file_copy(csv_file, org_folder) :
    folder_name = os.path.join("./data/food_dataset/", os.path.basename(org_folder).split('_')[0])
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            label = row['label']  
            file_name = row['img_name'] 
            org_path = os.path.join(org_folder, file_name)
            if os.path.exists(org_path):
                dst_folder = os.path.join(folder_name, label)
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder, exist_ok=True)
                shutil.copy2(org_path, os.path.join(dst_folder, os.path.basename(org_path)))
            else:
                print(f"경고: {org_path} 파일이 존재하지 않습니다.")
    print("Complete File Copy")

def expend2square(pil_image, background_color):
    width, height = pil_image.size
    if width == height:
        return pil_image
    elif width > height :
        result = Image.new(pil_image.mode, (width, width), background_color)
        result.paste(pil_image, (0, (width-height) // 2))
        return result
    else:
        result = Image.new(pil_image.mode, (height, height), background_color)
        result.paste(pil_image, ((height-width) // 2, 0))
        return result 

def resize_with_padding(pil_image, new_size, background_color):
    img = expend2square(pil_image, background_color)
    img = img.resize((new_size[0], new_size[1]), Image.ANTIALIAS)
    return img

def resize_images(data_path) :
    img_path_list = glob.glob(os.path.join(data_path, "*","*.png"))
    for img_path in tqdm(img_path_list):
        try:
            dir, file = os.path.split(img_path)
            folder_name = dir.rsplit('\\')[1]
            os.makedirs(os.path.join("./data/food_dataset/resized", folder_name), exist_ok=True)
            name = os.path.basename(img_path).rsplit('.png')[0]
            img = Image.open(img_path).convert('RGB')
            img_new = resize_with_padding(img, (224,224), (0,0,0)) 
            save_file_name = f"./data/food_dataset/resized/{folder_name}/{name}.png"
            img_new.save(save_file_name, "png")
        except Exception as ex:
            print(f"Error occurs on : {file} with the reason of {ex}")

train_csv = "./data/food_dataset/train_labels.csv"
val_csv = "./data/food_dataset/val_labels.csv"
train_folder = "./data/food_dataset/train_set"
val_folder = "./data/food_dataset/val_set"

csv_index_to_label(train_csv)
csv_index_to_label(val_csv)    

csv_label_file_copy(train_csv, train_folder)
csv_label_file_copy(val_csv, val_folder)

# resized_train = "./data/food_dataset/train"
# resized_val = "./data/food_dataset/val"
# resize_images(resized_train)
# resize_images(resized_val)