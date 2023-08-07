import cv2 
import json
import os 
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.builder import DATASETS
from mmdet.apis import set_random_seed
from mmcv import Config
from mmdet.datasets.coco import CocoDataset
from mmdet.models import build_detector

@DATASETS.register_module(force=True)
class ParkingDataset(CocoDataset) :
    CLASSES = ('세단(승용차)', 'suv', '승합차', '버스', '학원차량(통학버스)', 
               '트럭', '택시', '성인', '어린이', '오토바이', '전동킥보드', 
               '자전거', '유모차', '쇼핑카트')
    
# Config setting == train config setting
config_file = "./configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_file)

# Learning rate setting
# sing gpu -> 0.0025
cfg.optimizer.lr = 0.0025

# dataset에 대한 환경 파라미터 수정
cfg.dataset_type = 'ParkingDataset'
cfg.data_root = './data/'

# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정
cfg.data.train.type = 'ParkingDataset'
cfg.data.train.ann_file = './data/train.json'
cfg.data.train.img_prefix = './data/images/'

cfg.data.val.type = 'ParkingDataset'
cfg.data.val.ann_file = './data/val.json'
cfg.data.val.img_prefix = './data/images/'

cfg.data.test.type = 'ParkingDataset'
cfg.data.test.ann_file = './data/test.json'
cfg.data.test.img_prefix = './data/images/'

# class의 갯수 수정
cfg.model.roi_head.bbox_head.num_classes = 14

# small car -> change anchor -> size 8 -> 4
cfg.model.rpn_head.anchor_generator.scales = [4]

# pretrained 모델
cfg.load_from = './dynamic_rcnn_r50_fpn_1x-62a3f276.pth'

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정
cfg.work_dir = './work_dirs/0807'

# 학습율 변경 환경 파라미터 설정
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# CocoDataset의 경우 metric을 bbox로 설정해야함.(mAP아님. bbox로 설정하면 mAP를 iou threshold를 0.5 - 0.95까지 변경하면서 측정)
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 6
cfg.checkpoint_config.interval = 6

cfg.runner.max_epochs = 10
cfg.seed = 0 
cfg.data.samples_per_gpu = 6
cfg.data.workers_per_gpu = 2
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
set_random_seed(0, deterministic=False)
# print("cfg show >>", cfg.pretty_text)

checkpoint_file_path = "./epoch_15.pth"
model = init_detector(cfg, checkpoint_file_path, device='cuda:0')

with open("./data/test.json", 'r', encoding='utf-8') as f :
    image_infos = json.load(f)

# Confidence Score default -> 0.5
score_threshold = 0.5
for img_info in image_infos['images']:
    file_name = img_info['file_name']
    image_path = os.path.join("./data/images/", file_name)
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = inference_detector(model, img)
    # category_dict = image_infos["categories"]
    
    for number, result in enumerate(results) :
        if len(result) == 0 :
            continue
        category_id = number + 1 
        result_filtered = result[np.where(result[:, 4] > score_threshold)]
        if len(result_filtered) == 0 :
            continue
        for i in range(len(result_filtered)) :
            tmp_dict = dict()
            x_min = int(result_filtered[i, 0])
            y_min = int(result_filtered[i, 1])
            x_max = int(result_filtered[i, 2])
            y_max = int(result_filtered[i, 3])
            # print(x_min, y_min, x_max, y_max, float(result_filtered[i,4]))
            
            # width = x_max - x_min
            # width = float(width)
            # height = y_max - y_min
            # height = float(height)

            rect = cv2.rectangle(img, (x_min,y_min), (x_max,y_max), (0,255,0), 2)

    cv2.imshow("test", rect)
    if cv2.waitKey(0) == ord('q') :
        exit()