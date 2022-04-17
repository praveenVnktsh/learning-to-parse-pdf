from flash.image import SemanticSegmentation
from flash.image import SemanticSegmentationData

dm = SemanticSegmentationData.from_folders(
    train_folder="/content/learning-to-parse-pdf/documentAnalysis/data/primalayoutanal/finaldataset/images",
    train_target_folder="/content/learning-to-parse-pdf/documentAnalysis/data/primalayoutanal/finaldataset/annotations",
    val_split=0.1,
    batch_size = 5,
    transform_kwargs=dict(image_size=(800, 640)),
    num_classes=2,
    num_workers = 4,
)

model = SemanticSegmentation(
  head="unet", 
  backbone='efficientnet-b0', 
  num_classes=dm.num_classes  
)

from flash import Trainer

trainer = Trainer(max_epochs=100, gpus = 0)
trainer.fit(model, datamodule=dm)
trainer.save_checkpoint("semantic_segmentation_model.pt")