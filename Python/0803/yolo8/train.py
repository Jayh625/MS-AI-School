from ultralytics import YOLO

if __name__ == "__main__" :
    model = YOLO("yolov8s.pt")
    model.train(data="data.yaml", epochs=100, batch=64, lrf=0.025)

# Augmentation 수정
# ./ultralytics/cfg/default.yaml