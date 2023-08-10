import os
import matplotlib.pyplot as plt
import random 

from pycocotools.coco import COCO
from PIL import Image

annfile_path = "./car_damage_dataset/train/COCO_train_annos.json"
mul_annfile_path = "./car_damage_dataset/train/COCO_mul_train_annos.json"
img_path = "./car_damage_dataset/img/"

coco = COCO(annfile_path)
mul_coco = COCO(mul_annfile_path)

# Class info
cats = coco.loadCats(coco.getCatIds())
coco_class_name = [cat['name'] for cat in cats]
print("COCO categories for damages \n{}\n".format(', '.join(coco_class_name)))

mul_cats = mul_coco.loadCats(mul_coco.getCatIds())
mul_class_name = [cat['name'] for cat in mul_cats]
print("COCO categories for damages \n{}\n".format(', '.join(mul_class_name)))

catIds = coco.getCatIds(catNms=['damage'])
imgIds = coco.getImgIds(catIds=catIds)
random_img_id = random.choice(imgIds)

imgIds = coco.getImgIds(imgIds=[random_img_id])
print(imgIds)
img = coco.loadImgs(imgIds)[0]
image_path = os.path.join(img_path, img['file_name'])
print(image_path)
image = Image.open(image_path)
image_org = image.copy()
plt.axis('off')
plt.imshow(image)
plt.show()

# get damage annotations
annIds = coco.getAnnIds(imgIds=imgIds, iscrowd=None)
anns = coco.loadAnns(annIds)
print(anns)
plt.imshow(image)
plt.axis('off')
coco.showAnns(anns, draw_bbox=True)
plt.show()

mul_annIds = mul_coco.getAnnIds(imgIds=imgIds, iscrowd=None)
mul_anns = mul_coco.loadAnns(mul_annIds)
category_map = dict()
for ele in list(mul_coco.cats.values()) :
    category_map.update({ele['id']:ele['name']})
# {1: 'headlamp', 2: 'rear_bumper', 3: 'door', 4: 'hood', 5: 'front_bumper'}

# create a list of parts in the image
parts = []
for region in mul_anns :
    parts.append(category_map[region['category_id']])

# plot parts
plt.imshow(image_org)
plt.axis('off')
mul_coco.showAnns(mul_anns, draw_bbox=True)
plt.show()