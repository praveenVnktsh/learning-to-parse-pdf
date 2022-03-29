import glob
import cv2
import numpy as np


for path in glob.glob('data/train-0/publaynet/annotations/*.jpg'):

    image = cv2.imread(path)
    newimg = np.zeros(image.shape[:2])
    
    (255, 0, 0), #text
        (0, 255, 0), #title
        (0, 0, 255), #list
        (255, 255, 0), #table
        (0, 255, 255), #figure
        (255, 0, 255)] # mask colors