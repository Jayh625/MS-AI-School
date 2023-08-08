import os
import glob 
import cv2
from ultralytics import YOLO

model = YOLO("./runs/detect/train4/weights/best.pt")
data_path = "./ultralytics/cfg/flow_chart_yolo_dataset/test/images"
data_path_list = glob.glob(os.path.join(data_path, "*.jpg"))
for path in data_path_list :
    file_name = os.path.basename(path)
    # image read
    image = cv2.imread(path)
    
    # class name info
    names = model.names
    results = model.predict(path, save=False, imgsz=640, conf=0.7)
    boxes = results[0].boxes
    results_info = boxes
    cls_numbers = results_info.cls # 클래스 넘버
    conf_numbers = results_info.conf # 바운딩 박스 점수
    box_xyxy = results_info.xyxy
    
    for bbox, cls_idx, conf_idx in zip(box_xyxy, cls_numbers, conf_numbers) :
        class_number = int(cls_idx.item())
        class_name = names[class_number]
        
        x1 = int(bbox[0].item())
        y1 = int(bbox[1].item())
        x2 = int(bbox[2].item())
        y2 = int(bbox[3].item())
        print(class_name, class_number, x1,y1,x2,y2)
        rect = cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(rect, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(f"./results/{file_name}", rect)
    # cv2.imshow("test", rect)
    # if cv2.waitKey(0) == ord('q') :
    #     exit()