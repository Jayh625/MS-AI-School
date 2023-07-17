import json
import os
import cv2
from PIL import Image
import glob
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

def bbox_crop_images(json_data_path, images_path) : 
    with open(json_data_path, 'r', encoding='utf-8') as j :
        json_data = json.load(j)

    files = os.listdir(images_path)
    for file in tqdm(files) :
        json_infos = json_data[file]
        annotations = json_infos['anno']
        for i, annots in enumerate(annotations) : 
            label = annots['label']
            bbox = annots['bbox']
            os.makedirs(os.path.join("./data/metal_damged", 'data', label), exist_ok=True)
            os.makedirs(os.path.join("./data/metal_damged", 'train', label), exist_ok=True)
            os.makedirs(os.path.join("./data/metal_damged", 'val', label), exist_ok=True)
            x, y, w, h = bbox
            img = cv2.imread(os.path.join(images_path, file))
            cropped_img = img[y:y+h, x:x+w]
            file_name = file.split(".")[0]
            new_file_path = os.path.join("./data/metal_damaged", "data", label, f"{file_name}_cropped_{str(i).zfill(1)}.png")
            print(new_file_path)
            cv2.imwrite(new_file_path, cropped_img)

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
            os.makedirs(os.path.join("./data/metal_damaged/resized", folder_name), exist_ok=True)
            name = os.path.basename(img_path).rsplit('.png')[0]
            img = Image.open(img_path).convert('RGB')
            img_new = resize_with_padding(img, (256,256), (0,0,0)) 
            save_file_name = f"./data/metal_damaged/resized/{folder_name}/{name}.png"
            img_new.save(save_file_name, "png")
        except Exception as ex:
            print(f"Error occurs on : {file} with the reason of {ex}")

def file_split(src, dst) :
    files = glob.glob(os.path.join(src, "*", "*"))
    train_list, val_list = train_test_split(files, test_size=0.1)
    for file in train_list : 
        folder_path = file.split("\\")[1]
        shutil.copy2(file, os.path.join(dst, "train", folder_path))
    for file in val_list : 
        folder_path = file.split("\\")[1]
        shutil.copy2(file, os.path.join(dst, "val", folder_path))
    print("File Split Completed") 

json_data_path = "./data/data/anno/annotation.json"
images_path = "./data/data/images"
bbox_crop_images(json_data_path, images_path)

resized_data_path = "./data/metal_damaged/data"
resize_images(resized_data_path)

src = "./data/metal_damaged/resized"
dst = "./data/metal_damaged/"
file_split(src, dst)