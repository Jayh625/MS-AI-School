from ultralytics import YOLO

if __name__ == "__main__" :
    model = YOLO("yolov8s.pt")
    model.train(data="swim_data.yaml", epochs=100, batch=32, lrf=0.0025)

    # model = YOLO("./runs/detect/train3/weights/last.pt")
    # model.train(resume=True)