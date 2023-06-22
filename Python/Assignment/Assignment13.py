
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from PIL import Image
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

import time
import os
import urllib, requests

import cv2
import json

class CrawlingandResizing :
    def __init__(self) :
        pass
    def google_crawling(search_words, counts) :
        print("Download starts..")
        service = Service('./chromedriver.exe')
        driver = webdriver.Chrome(service=service)
        driver.implicitly_wait(3)
        driver.get("https://www.google.co.kr/imghp?h1=ko")
        os.makedirs('./imgs/original', exist_ok=True)
        current_path = os.getcwd()
        current_path += '\\data\\imgs\\original\\'
        for search_word in search_words:
            elem = driver.find_element(By.NAME, value="q")
            elem.clear()
            elem.send_keys(search_word)
            elem.send_keys(Keys.RETURN)
            SCROLL_PAUSE_TIME = 1
            last_height = driver.execute_script("return document.body.scrollHeight") 

            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  
                time.sleep(SCROLL_PAUSE_TIME)  
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:  
                    try:
                        driver.find_elements(By.CSS_SELECTOR, value=".mye4qd").click()  
                    except:
                        break
                last_height = new_height
                
            images = driver.find_elements(By.CSS_SELECTOR, value=".rg_i.Q4LuWd")
            folder_name = search_word
            if not os.path.isdir(current_path +folder_name):  
                os.makedirs(current_path +folder_name, exist_ok=True)
                
            count = 1
            for image in tqdm(images):
                if count > counts:
                    break
                try:
                    image.click()
                    time.sleep(3)
                    imgUrl = driver.find_element(By.XPATH, value='//*[@id="Sva75c"]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div/div/div/div[3]/div[1]/a/img[1]').get_attribute("src")
                    urllib.request.urlretrieve(
                        imgUrl,
                        current_path  + folder_name + "/" + str(count).zfill(3)+ "_" + search_word + ".png")
                    count += 1
                except:
                    pass
        
            driver.back()
        print("Download completed")
        driver.close()

    def expend2square(self, pil_image, background_color):
            width, height = pil_image.size
            if width == height:
                return pil_image
            elif width > height :
                result = Image.new(pil_image.mode, (width, width), background_color)
                result.paste(pil_image, (0, (width-height) // 2))
                return result
            else:
                result = Image.new(pil_image.mode, (height, height), background_color)
                result.paste(pil_image, ((height-width) // 2, 0))
                return result 

    def resize_with_padding(self, pil_image, new_size, background_color):
        img = self.expend2square(pil_image, background_color)
        img = img.resize((new_size[0], new_size[1]), Image.ANTIALIAS)
        return img

    def img_aug(self, img) :
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        angle = 30
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        flipped_right_and_left = cv2.flip(image, 1)
        flipped_up_and_down = cv2.flip(image, 0)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation_factor = 0.8
        img_hsv[:, :, 1] = img_hsv[:, :, 1] * saturation_factor
        img_saturated = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        img_saturated = cv2.cvtColor(img_saturated, cv2.COLOR_BGR2RGB)
        return [image, rotated, flipped_right_and_left, flipped_up_and_down, img_saturated]
    
    def get_files_paths(self, input_path):
            file_list = []
            for (path, dir, files) in os.walk(input_path):
                for f in files:
                    file_list.append(os.path.join(path, f))
            return file_list
    def make_dirs(self, input_path):
        dir_list = []
        for (path, dir, files) in os.walk(input_path):
            for d in dir:
                dir_list.append(d)
        for dir in dir_list:
            os.makedirs(f"./data/imgs/resized/{dir}", exist_ok=True)
            os.makedirs(f"./data/imgs/augmented/{dir}", exist_ok=True)
            
class CatchThief :
    def __init__(self) :
        pass
    def catch_thief_by_json(self, json_path, video_path):
        folder_name = json_path.split("\\")[0].split('/')[-1]
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

        cap = cv2.VideoCapture(video_path)
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
    def get_files_paths(self, input_path):
            file_list = []
            for (path, dir, files) in os.walk(input_path):
                for f in files:
                    file_list.append(os.path.join(path, f))
            return file_list   
if __name__ == "__main__" :
    # Crawling
    cr = CrawlingandResizing()
    input_count = int(input("다운받을 목록 개수를 입력하세요: "))
    input_list = []
    for i in range(input_count):
        input_value = input("목록 {} 입력하세요: ".format(i + 1))
        input_list.append(input_value)
    print("목록 :", input_list)
    counts = int(input("다운받을 이미지 개수를 입력하세요: "))
    cr.google_crawling(input_list, counts)

    # Original to Resized (255,255) 
    input_path = './data/imgs/original/'
    cr.make_dirs(input_path)
    img_path_list = glob.glob(os.path.join(input_path, "*", "*.jpg"))
    for img_path in tqdm(img_path_list):
        dir, file = os.path.split(img_path)
        folder_name = dir.rsplit('\\')[1]
        name = os.path.basename(img_path).rsplit('.png')[0]
        img = Image.open(img_path)
        img_new = cr.resize_with_padding(img, (255,255), (0,0,0)) 
        save_file_name = f"./data/imgs/resized/{folder_name}/resized_{name}.png"
        img_new.save(save_file_name, "png")

    # Resized to Augmented
    resized_img_path = './data/imgs/resized/'
    resized_img_path_list = glob.glob(os.path.join(resized_img_path, "*", "*.png"))
    for img_path in tqdm(resized_img_path_list):
        dir, file = os.path.split(img_path)
        folder_name = dir.rsplit('\\')[-1]
        name = os.path.basename(img_path).rsplit('.png')[0]
        img = cv2.imread(img_path)
        for i, aug_img in enumerate(cr.img_aug(img)) :
            aug_img_name = f"augmented{i}_{name}.png"
            save_file = f"./data/imgs/augmented/{folder_name}/{aug_img_name}"
            aug_image = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_file, aug_image)
    
    # Catch Thief
    ct = CatchThief()
    json_path = "./data/raw_data/json/"
    video_path = "./data/raw_data/video/"
    json_list = ct.get_files_paths(json_path)
    video_list = ct.get_files_paths(video_path)
    count = 0
    for i in range(len(json_list)) :
        ct.catch_thief_by_json(json_list[i], video_list[i])