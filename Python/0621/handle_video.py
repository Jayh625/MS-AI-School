import cv2
import os

cap = cv2.VideoCapture("./data/홈페이지 배경 샘플 영상 - 바다.mp4")

# print(f"동영상 프레임 수 : {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
# print(f"동영상 가로 길이 : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
# print(f"동영상 세로 길이 : {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
# print(f"FPS : {cap.get(cv2.CAP_PROP_FPS)}")

os.makedirs("./data/video_frame_dataset", exist_ok=True)
img_count = 0
while True : 
    ret, frame = cap.read()
    if not ret :
        break
    if img_count % 15 == 0 :
        img_filename = f"./data/video_frame_dataset/frame_{img_count:04d}.png"
        cv2.imwrite(img_filename, frame)
    
    img_count += 1
cap.release()