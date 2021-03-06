from pycocotools.coco import COCO
import random
import cv2
import numpy as np
import os
from tqdm import tqdm
def generateOutput(annotations, newimage, coco=None):
    """
    Args:
        annotations (List):
            a list of coco annotaions for the current image
        coco (`optional`, defaults to `False`):
            COCO annotation object instance. If set, this function will
            convert the loaded annotation category ids to category names
            set in COCO.categories
    """
    
    for ele in annotations:

        x, y, w, h = ele['bbox']
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        cv2.rectangle(newimage, (x, y), (x + w, y + h), int(ele["category_id"]), -1)
        # print(np.array(ele['segmentation'], dtype = int))
        # cv2.polylines(newimage, np.array(ele['segmentation'], dtype = int), True, (0, 255, 0), 2)

    return newimage

#/home/praveen_venkatesh/data/labels/publaynet
COCO_IMG_PATH = r'/home/praveen_venkatesh/data/train-0/publaynet/train'
COCO_ANNO_PATH  = r'/home/praveen_venkatesh/data/labels/publaynet/train.json'

coco = COCO(COCO_ANNO_PATH)

color_map = {
    'text':   'red',
    'title':  'blue',
    'list':   'green',
    'table':  'purple',
    'figure': 'pink',
}

os.makedirs(f'{COCO_IMG_PATH}/annotations/', exist_ok=True)
for image_id in tqdm(coco.imgs.keys()):
    image_info = coco.imgs[image_id]
    annotations = coco.loadAnns(coco.getAnnIds([image_id]))
    image = cv2.imread(f'{COCO_IMG_PATH}/{image_info["file_name"]}')
    if image is None:
        continue
    newimage = np.zeros(image.shape[:2], dtype=np.uint8)
    newimage = generateOutput(annotations, newimage, coco)
    cv2.imwrite(f'{COCO_IMG_PATH}/annotations/{image_info["file_name"]}', newimage)
