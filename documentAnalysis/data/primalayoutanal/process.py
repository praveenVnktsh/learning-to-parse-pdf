import xml.etree.ElementTree as ET
import numpy as np
import glob
import cv2

for path in glob.glob(r'E:\Google Drive\Acads\Notes\final sem\ML\project\learning-to-parse-pdf\documentAnalysis\data\primalayoutanal\XML/*.xml'):
    tree = ET.parse(path)
    root = tree.getroot()
    for child in root:
        if "Page" in child.tag:
            filename = child.attrib['imageFilename'].split('.')[0]

            print(filename, 'Dims:',int(child.attrib['imageHeight']), int(child.attrib['imageWidth']) )
            try:
                img = np.zeros((int(child.attrib['imageHeight']), int(child.attrib['imageWidth'])), dtype=np.uint8)
                for region in child:
                    if 'TextRegion' in region.tag:
                        coords = region[0]
                        coordslist = []
                        for coord in coords:
                            coord = coord.attrib
                            coordslist.append([int(coord['x']), int(coord['y'])])
                        cv2.fillPoly(img, np.array([coordslist]), 255, )

            except:
                print('FAILL!')
                pass

            break
    

    img = cv2.resize(img, (600, 800), interpolation= cv2.INTER_NEAREST)
    try:
        cv2.imwrite(r'E:\Google Drive\Acads\Notes\final sem\ML\project\learning-to-parse-pdf\documentAnalysis\data\primalayoutanal/finaldataset/annotations/' + filename + '.png',   img)
        
        imm = cv2.imread(r'E:\Google Drive\Acads\Notes\final sem\ML\project\learning-to-parse-pdf\documentAnalysis\data\primalayoutanal/downsize/' + filename + '.png')
        cv2.imwrite(r'E:\Google Drive\Acads\Notes\final sem\ML\project\learning-to-parse-pdf\documentAnalysis\data\primalayoutanal/finaldataset/images/' + filename + '.png', imm)
        imm[:, :, 0][img == 255] = 0
        cv2.imwrite(r'E:\Google Drive\Acads\Notes\final sem\ML\project\learning-to-parse-pdf\documentAnalysis\data\primalayoutanal/finaldataset/viz/' + filename + '.png', imm)
    except:
        pass