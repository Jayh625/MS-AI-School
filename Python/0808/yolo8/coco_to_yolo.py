import json
import os
from shutil import copy

def convert_coco_to_yolo(coco_json_file, org_images_folder, images_folder, labels_folder):
    with open(coco_json_file, 'r') as f:
        data = json.load(f)
        
    images = {img['id']: img for img in data['images']}

    for ann in data['annotations']:
        class_id = ann['category_id'] - 1
        img_id = ann['image_id']
        img = images[img_id]
        img_path = os.path.join(org_images_folder, img['file_name'])
        copy(img_path, images_folder)

        width, height = img['width'], img['height']
        bbox = ann['bbox']
        
        x, y, w, h = bbox

        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w /= width
        h /= height

        yolo_ann = f"{class_id} {x_center} {y_center} {w} {h}\n"

        txt_file_path = os.path.join(labels_folder, f"{os.path.splitext(img['file_name'])[0]}.txt")
        with open(txt_file_path, 'a') as txt_file:
            txt_file.write(yolo_ann)

train_path = "./ultralytics/cfg/flow_chart_dataset/train/"
valid_path = "./ultralytics/cfg/flow_chart_dataset/valid/"
test_path = "./ultralytics/cfg/flow_chart_dataset/test/"

train_yolo_path = "./ultralytics/cfg/flow_chart_yolo_dataset/train/"
valid_yolo_path = "./ultralytics/cfg/flow_chart_yolo_dataset/valid/"
test_yolo_path = "./ultralytics/cfg/flow_chart_yolo_dataset/test/"

train_coco_json_file_path = os.path.join(train_path, "_annotations.coco.json")
valid_coco_json_file_path = os.path.join(valid_path, "_annotations.coco.json")
test_coco_json_file_path = os.path.join(test_path, "_annotations.coco.json")

train_images_path = os.path.join(train_yolo_path, "images")
train_labels_path = os.path.join(train_yolo_path, "labels")

valid_images_path = os.path.join(valid_yolo_path, "images")
valid_labels_path = os.path.join(valid_yolo_path, "labels")

test_images_path = os.path.join(test_yolo_path, "images")
test_labels_path = os.path.join(test_yolo_path, "labels")

os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)

os.makedirs(valid_images_path, exist_ok=True)
os.makedirs(valid_labels_path, exist_ok=True)

os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

convert_coco_to_yolo(train_coco_json_file_path, train_path, train_images_path, train_labels_path)
convert_coco_to_yolo(valid_coco_json_file_path, valid_path, valid_images_path, valid_labels_path)
convert_coco_to_yolo(test_coco_json_file_path, test_path, test_images_path, test_labels_path)