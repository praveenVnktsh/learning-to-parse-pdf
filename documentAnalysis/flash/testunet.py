from flash.image import SemanticSegmentation
from flash.image import SemanticSegmentationData


dm = SemanticSegmentationData.from_folders(
    predict_folder="/home/praveen_venkatesh/learning-to-parse-pdf/documentAnalysis/data/images",
    transform_kwargs=dict(image_size=(800, 640)),
    )

model = SemanticSegmentation.load_from_checkpoint("/home/praveen_venkatesh/learning-to-parse-pdf/documentAnalysis/flash/lightning_logs/version_10/checkpoints/epoch=9-step=86319.ckpt")

from flash import Trainer

trainer = Trainer(max_epochs=1, gpus = 1)
predictions = trainer.predict(model, dm)

print(predictions)