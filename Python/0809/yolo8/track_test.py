import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load the yolov8 model
model = YOLO('yolov8m.pt')

# open the video file
video_path = "./MOT17-04-SDP-raw.webm"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda : [])

while cap.isOpened():
    # Read a frame from video
    success, frame = cap.read()
    if success :
        # Run yolo8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot()
        for box, track_id in zip(boxes, track_ids) :
            x1,y1,x2,y2 = box
            track = track_history[track_id]
            track.append((float(x1), float(y1)))
            if len(track) > 30 :
                track.pop(0) # retain 90 tracks for 90 frames

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(210,150,180), thickness=5)
            
        # Display the annotated frame
        cv2.imshow("Tracking..", annotated_frame)
        if cv2.waitKey(30) & 0xFF == ord('q') :
            exit()