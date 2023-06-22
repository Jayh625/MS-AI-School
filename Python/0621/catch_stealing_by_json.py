import cv2
import os
import json

json_path = "data\\raw_data\\json\\Stealing_Courier\\C041_A30_SY32_P07_S06_02DAS.json"

folder_name = json_path.split("\\")[2]
file_name = json_path.split("\\")[-1]
file_name = file_name.replace(".json", "")
os.makedirs(f"./data/AI_hub_final_data/{folder_name}/{file_name}", exist_ok=True)

with open(json_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

metadata_info = json_data["metadata"]
categories_info = json_data["categories"]

crime_info = categories_info["crime"]
action_info = categories_info["action"]
sympton_info = categories_info["symptom"]

cap = cv2.VideoCapture("data\\raw_data\\video\\Stealing_Courier\\C041_A30_SY32_P07_S06_02DAS.mp4")
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

file_info = json_data["file"]

for item in file_info :
    videos_info = item["videos"]
    block_information = videos_info["block_information"]

    count = 0 
    for block in block_information :
        if block["block_detail"] == "A30":
            start_time = block["start_time"]
            end_time = block["end_time"]
            start_frame_index = block["start_frame_index"]
            end_frame_index = block["end_frame_index"]

            for frame_idx in range(int(start_frame_index), int(end_frame_index), 30):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret : 
                    img_name = f"./data/AI_hub_final_data/{folder_name}/{file_name}/frame_{str(count).zfill(4)}.png"
                    cv2.imwrite(img_name, frame)
                    count += 1
cap.release()                    