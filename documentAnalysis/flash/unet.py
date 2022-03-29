from flash.image import SemanticSegmentation
from flash.image import SemanticSegmentationData

dm = SemanticSegmentationData.from_folders(
    train_folder="/home/praveen_venkatesh/data/train-0/publaynet/train",
    train_target_folder="/home/praveen_venkatesh/data/train-0/publaynet/annotations",
    val_split=0.1,
    batch_size = 5,
    transform_kwargs=dict(image_size=(800, 640)),
    num_classes=8,
    num_workers = 4,
)

model = SemanticSegmentation(
  head="unet", 
  backbone='efficientnet-b0', 
  num_classes=dm.num_classes  
)

from flash import Trainer

trainer = Trainer(max_epochs=100, gpus = 1)
trainer.fit(model, datamodule=dm)
trainer.save_checkpoint("semantic_segmentation_model.pt")