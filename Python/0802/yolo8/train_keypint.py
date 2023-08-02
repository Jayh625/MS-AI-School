from ultralytics import YOLO

if __name__ == "__main__" :
    model = YOLO("yolov8s-pose.pt")
    model.train(data="keypoint_data.yaml", epochs=100, batch=64, lrf=0.025, hsv_h=0.0, degrees=5)