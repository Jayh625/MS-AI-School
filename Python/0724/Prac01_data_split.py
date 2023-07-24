import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# new folder 
train_folder = "./data/train"
val_folder = "./data/val"
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# data folder path
image_folder_path = "./dataset/images/"
annotation_folder_path = "./dataset/annotations/"

# csv path 
csv_file_path = os.path.join(annotation_folder_path, "annotations.csv")

# csv file -> pandas DataFrame read
annotation_df = pd.read_csv(csv_file_path)

# unique() -> 중복되지 않는 고유한 값 출력 
image_names = annotation_df['filename'].unique()
train_names, val_names = train_test_split(image_names, test_size=0.2)
# print(f"{len(train_names)}, {len(val_names)}")

# train data copy and bounding box info save
train_annotations = pd.DataFrame(columns=annotation_df.columns)
for image_name in train_names : 
    # print(f"image_name value : {image_name}")
    img_path = os.path.join(image_folder_path, image_name)
    new_image_path = os.path.join(train_folder, image_name)
    shutil.copy2(img_path, new_image_path)

    # annotation csv
    annotation = annotation_df.loc[annotation_df['filename']==image_name].copy()
    annotation['filename'] = image_name
    train_annotations = train_annotations._append(annotation)

train_annotations.to_csv(os.path.join("./data/", "train_annotations.csv"), index=False)

# val data copy and bounding box info save
val_annotations = pd.DataFrame(columns=annotation_df.columns)
for image_name in val_names : 
    # print(f"image_name value : {image_name}")
    img_path = os.path.join(image_folder_path, image_name)
    new_image_path = os.path.join(val_folder, image_name)
    shutil.copy2(img_path, new_image_path)

    # annotation csv
    annotation = annotation_df.loc[annotation_df['filename']==image_name].copy()
    annotation['filename'] = image_name
    val_annotations = val_annotations._append(annotation)

val_annotations.to_csv(os.path.join("./data/", "val_annotations.csv"), index=False)