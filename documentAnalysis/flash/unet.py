from flash.image import SemanticSegmentation
from flash.image import SemanticSegmentationData

# dm = SemanticSegmentationData.from_folders(
#     train_folder="data/CameraRGB",
#     train_target_folder="data/CameraSeg",
#     val_split=0.1,
#     image_size=(256, 256),
#     num_classes=5,
# )
print(SemanticSegmentation.available_heads())

print(sorted(SemanticSegmentation.available_backbones('unet')))

# print(SemanticSegmentation.available_pretrained_weights('unet'))


# model = SemanticSegmentation(
#   head="unet", backbone='efficientnet-b0', pretrained="advprop", num_classes=dm.num_classes)