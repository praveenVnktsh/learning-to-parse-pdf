import glob
import cv2
import numpy as np



for path in glob.glob(r'E:\Google Drive\Acads\Notes\final sem\ML\project\learning-to-parse-pdf\documentAnalysis\data\primalayoutanal\Images\*.tif'):

    img = cv2.imread(path)
    img = cv2.resize(img, (600, 800))
    cv2.imshow('image', img)
    cv2.imwrite(r'E:\Google Drive\Acads\Notes\final sem\ML\project\learning-to-parse-pdf\documentAnalysis\data\primalayoutanal\downsize/' + path.split('\\')[-1].split('.')[0] + '.png', img)
    cv2.waitKey(1)
    print(path)