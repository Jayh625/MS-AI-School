import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load the yolov8 model
model = YOLO('./runs/detect/train5/weights/best.pt')

# open the video file
video_path = "./swimming_pool.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda : [])

# Create a directory to save frames
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)
frame_count = 0

while cap.isOpened() :
    # Read a frame from video
    success, frame = cap.read()

    if success :
        # Run yolov8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        boxes = results[0].boxes.xyxy.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot()

        for box, track_id in zip(boxes, track_ids) :
            x1,y1,x2,y2 = box
            track = track_history[track_id]

            track.append((float(x1), float(y1)))
            if len(track) > 30 :
                track.pop(0) # retain 90 tracks for 90 frames

            # Drew the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
            img = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        
        cv2.imshow("Tracking..", img)

        # Save the annotated frame
        frame_name = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_name, img)
        frame_count += 1

        if cv2.waitKey(30) & 0xFF == ord('q') :
            exit()
    else :
        break

# cap.release()
# cv2.destroyAllWindows()
