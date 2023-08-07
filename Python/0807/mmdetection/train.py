import mmcv

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import set_random_seed
from mmdet.utils import collect_env, get_root_logger, setup_multi_processes

@DATASETS.register_module(force=True)
class ParkingDataset(CocoDataset) :
    CLASSES = ('세단(승용차)', 'suv', '승합차', '버스', '학원차량(통학버스)', 
               '트럭', '택시', '성인', '어린이', '오토바이', '전동킥보드', 
               '자전거', '유모차', '쇼핑카트')

# Config file setting
# Dynamic RCNN model load
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
cfg.data.val.ann_file = './data/valild.json'
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

# Epochs 설정
# 8*12 -> 96
cfg.runner.max_epochs = 10
cfg.seed = 0 
cfg.data.samples_per_gpu = 6
cfg.data.workers_per_gpu = 2
print("Test", cfg.data)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
set_random_seed(0, deterministic=False)
print("cfg show >>", cfg.pretty_text)

datasets = [build_dataset(cfg.data.train)]
print(datasets[0])

# datasets[0].__dict__로 모든 self variables의 key와 value값을 볼 수 있음
datasets[0].__dict__.keys()

model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES
print(model.CLASSES)

if __name__ == "__main__":
    # epochs는 config의 runner 파라미터로 지정됨. 기본 12회
    train_detector(model, datasets, cfg, distributed=False, validate=True)