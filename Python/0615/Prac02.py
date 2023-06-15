import json
import os
import cv2
import glob
import numpy as np

# json data
json_dir = "./anno/"
json_paths = glob.glob(os.path.join(json_dir, "*.json"))

# label dict
label_dict = {"수각류" : 0}

for json_path in json_paths : 
    # json 읽기
    with open(json_path, 'r', encoding='utf-8') as f :
        json_data = json.load(f)
        
    # images, annotations
    images_info = json_data['images']
    annotations_info = json_data['annotations']
    
    filename = images_info['filename']
    image_id = images_info['id']
    image_width = images_info['width']
    image_height = images_info['height']
    
    # 4:3
    new_width = 1024
    new_height = 768
    for ann_info in annotations_info :
        if image_id == ann_info['image_id'] :
            # image read 
            image_path = os.path.join("./images/", filename)
            image = cv2.imread(image_path)
            
            # image 스케일 
            scale_x = new_width / image.shape[1] # X 축 스케일 비율
            scale_y = new_height / image.shape[0] # Y 축 스케일 비율
            
            # 리사이즈
            resized_image = cv2.resize(image, (new_width, new_height))
            
            category_name = ann_info['category_name']
            polygons = ann_info['polygon']
            
            # 폴리곤의 좌표 생성
            points = []
            for polygon_info in polygons :
                x = polygon_info['x']
                y = polygon_info['y']
                
                resized_x = int(x * scale_x)
                resized_y = int(y * scale_y)
                
                points.append((resized_x, resized_y))
            
            # 폴리곤 그리기 
            cv2.polylines(resized_image, [np.array(points, np.int32).reshape((-1,1,2))], isClosed=True, color=(0,255,0), thickness=2)
            
            # 폴리곤 좌표를 이용한 바운딩 박스 만들기(xyxy format)
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            
            # Draw bounding box
            cv2.rectangle(resized_image, (x_min,y_min), (x_max, y_max), (0,0,255), 2)
            
            # bounding box yolo 양식으로 저장하기 
            center_x = ((x_max+x_min) / (2*new_width))
            center_y = ((y_max+y_min) / (2*new_height))
            yolo_w = (x_max-x_min) / new_width
            yolo_h = (y_max-y_min) / new_height
            
            # file name
            image_name_temp = filename.replace(".jpg", "")
            
            # label_number
            label_number = label_dict[category_name]

        # os.makedirs('./yolo_label_data', exist_ok=True)
        # with open(f"./yolo_label_data/{image_name_temp}.txt", "a") as f :
        #     f.write(f"{label_number} {center_x} {center_y} {yolo_w} {yolo_h}\n")

        cv2.imshow("Polygon", resized_image)

        if cv2.waitKey(0) & 0xFF == ord('q') : 
            exit()