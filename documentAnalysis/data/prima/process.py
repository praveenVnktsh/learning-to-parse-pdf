import xml.etree.ElementTree as ET
import numpy as np
import cv2
tree = ET.parse('annotation.xml')
root = tree.getroot()
# img = cv2.imread('00000086.tif')
for child in root:
    if "Page" in child.tag:
        img = np.zeros((int(child.attrib['imageHeight']), int(child.attrib['imageWidth'])), dtype=np.uint8)
        for region in child:
            if 'TextRegion' in region.tag:
                print(region.attrib)
                coords = region[0]
                coordslist = []
                for coord in coords:
                    coord = coord.attrib
                    coordslist.append([int(coord['x']), int(coord['y'])])

                cv2.fillPoly(img, np.array([coordslist]), 255, )

        break

temp = cv2.resize(img, (480, 640))
cv2.imshow('image', temp)
# cv2.imshow('imadge', cv2.resize(, (480, 640)))
cv2.waitKey(0)
