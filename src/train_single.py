import pandas as pd
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from ranzcr.utils import seed
from ranzcr.globals import *
from ranzcr.config import config
from ranzcr.data import RanzcrDataModule
from ranzcr.model import RanzcrClassifier


if __name__ == '__main__':

    FOLD = 0

    # Seed everything
    seed(config['seed'])

    # Read data
    train_k_df = pd.read_csv(os.path.join(INPUT_DIR, 'annotated_train_k.csv'))
    train_df = train_k_df[train_k_df.fold != FOLD]
    val_df = train_k_df[train_k_df.fold == FOLD]

    # Data module
    dm = RanzcrDataModule(TRAIN_IMAGES, train_df, val_df, config)
    classifier = RanzcrClassifier(config)

    # Create trainer
    filename = f"{config['base_model']}-{{epoch:02d}}{{val_score:.3f}}"

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=MODELS_DIR,
                                          mode='min',
                                          filename=filename)

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    trainer = Trainer(gpus=1,
                      max_epochs=config['epochs'],
                      precision=config['precision'],
                      num_sanity_val_steps=0,
                      callbacks=[checkpoint_callback, early_stopping]
                      )

    # Find best learning rate
    tune = False
    if tune:
        lr_finder = trainer.tuner.lr_find(classifier, dm)
        print(f'Suggested lr={lr_finder.suggestion()}')
        fig = lr_finder.plot(suggest=True)
        fig.show()

    # Train
    train = True
    torch.cuda.empty_cache()
    if train:
        trainer.fit(classifier, datamodule=dm)