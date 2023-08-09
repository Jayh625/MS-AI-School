from ultralytics import YOLO

if __name__ == "__main__" :
    model = YOLO("yolov8m.pt")
    model.train(data="glass_data.yaml", epochs=100, batch=32, degrees=5, lrf=0.0025)

    # model = YOLO("./runs/detect/train/weights/last.pt")
    # model.train(resume=True)