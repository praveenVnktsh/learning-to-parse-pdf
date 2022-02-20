from model import LitModel
from models.fcn16 import FCN16s
from dataloader import CustomDataset, LitDataLoader
import pytorch_lightning as pl
import torch.nn as nn

if __name__ == '__main__':
    hparams = {
        'lr': 0.001
    }

    baseModel = FCN16s(n_class = 3)
    lossFunc = nn.CrossEntropyLoss()
    model = LitModel(baseModel, lossFunc, hparams)

    dataset = LitDataLoader("datasets/")

    trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(model, dataset)