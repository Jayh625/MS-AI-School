from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import set_random_seed
from mmcv.runner import get_dist_info, init_dist
from mmdet.utils import collect_env, get_root_logger, setup_multi_processes
import os

@DATASETS.register_module(force=True)
class CarDamageDataset(CocoDataset) :
    CLASSES = ('headlamp', 'rear_bumper', 'door', 'hood', 'front_bumper')

# config
config_file = "./configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_file)

# Learning rate Setting
cfg.optimizer.lr = 0.0025
cfg.lr_config.warmup = None
# cfg.lr_config.step = [80, 95]

# dataset
cfg.dataset_type = "CarDamageDataset"
cfg.data_root = "./car_damage_dataset/"

# train, val, test dataset path
cfg.data.train.type = 'CarDamageDataset'
cfg.data.train.ann_file = './car_damage_dataset/train/COCO_mul_train_annos.json'
cfg.data.train.img_prefix = './car_damage_dataset/train/'
cfg.data.train.pipeline[2].img_scale=(512,512)
cfg.data.train.pipeline[3].flip_ratio=0.3

cfg.data.val.type = 'CarDamageDataset'
cfg.data.val.ann_file = './car_damage_dataset/val/COCO_mul_val_annos.json'
cfg.data.val.img_prefix = './car_damage_dataset/val/'
cfg.data.val.pipeline[1].img_scale=(512,512)

cfg.data.test.type = 'CarDamageDataset'
cfg.data.test.ann_file = './car_damage_dataset/val/COCO_mul_val_annos.json'
cfg.data.test.img_prefix = './car_damage_dataset/val/'
cfg.data.test.pipeline[1].img_scale=(512,512)

cfg.model.roi_head.bbox_head[0].num_classes = 5
cfg.model.roi_head.bbox_head[1].num_classes = 5
cfg.model.roi_head.bbox_head[2].num_classes = 5
cfg.model.roi_head.mask_head.num_classes = 5
cfg.model.rpn_head.anchor_generator.scales = [4]

# Pretrained model
cfg.load_from = "./cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth"

# weight file save path
cfg.work_dir = "./work_dirs/0810_cascade_rcnn_r50_fpn_1x_coco_multi"

# train setting
cfg.log_config.interval = 3
cfg.evaluation.interval = 3
cfg.checkpoint_config.interval = 3

cfg.runner.max_epochs = 96
cfg.seed = 777
cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu = 2
cfg.gpu_ids = range(1)
cfg.device = "cuda"
set_random_seed(777, deterministic=False)
# print(cfg.pretty_text)
# exit()

datasets = [build_dataset(cfg.data.train)]
# print(datasets[0])

if __name__ == "__main__" :
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    train_detector(model, datasets, cfg, distributed=False, validate=True)