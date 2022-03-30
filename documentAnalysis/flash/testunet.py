from flash.image import SemanticSegmentation
from flash.image import SemanticSegmentationData


dm = SemanticSegmentationData.from_folders(
    predict_folder=r"E:\Google Drive\Acads\Notes\final sem\ML\project\learning-to-parse-pdf\documentAnalysis\data\images/",
    transform_kwargs=dict(image_size=(800, 640)),
    batch_size = 1,
    )

model = SemanticSegmentation.load_from_checkpoint("semantic_segmentation_model.pt")

from flash import Trainer

trainer = Trainer(max_epochs=1, gpus = 1)
predictions = trainer.predict(model, dm)
import cv2
import numpy as np
import torch

inp = (predictions[0][0]['input']).numpy()
inp = np.transpose(inp, (1, 2, 0))
pred = (predictions[0][0]['preds'])
pred = torch.softmax(pred, dim = 0)
pred = torch.argmax(pred, dim = 0).numpy()
print(pred.shape, inp.shape)
inp[:, :, 0][pred == 1] = 255
inp[:, :, 1][pred == 2] = 255
inp[:, :, 2][pred == 3] = 255
inp[:, :, 1][pred == 4] = 128
inp[:, :, 2][pred == 5] = 128
inp[:, :, 0][pred == 6] = 128
inp[:, :, 1][pred == 7] = 0


cv2.imshow('image', inp)
cv2.imshow('segm', pred)
cv2.waitKey(0)