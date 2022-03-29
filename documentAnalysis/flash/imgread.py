import cv2
import glob

import numpy as np
for path in glob.glob('/home/praveen_venkatesh/data/train-0/publaynet/annotations/*.jpg'):

    img = cv2.imread(path, 0)

    print(np.unique(img))