import pandas as pd
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ranzcr.utils import seed
from ranzcr.globals import *
from ranzcr.config import config
from ranzcr.data import RanzcrDataModule
from ranzcr.model import RanzcrClassifier

def train_fold(experiment, version, fold):

    # Seed everything
    seed(config['fold_seed'][fold])

    # Read data
    train_k_df = pd.read_csv(os.path.join(INPUT_DIR, 'train_k.csv'))
    train_df = train_k_df[train_k_df.fold != fold]
    val_df = train_k_df[train_k_df.fold == fold]

    # Data module
    dm = RanzcrDataModule(TRAIN_IMAGES, train_df, val_df, config)
    classifier = RanzcrClassifier(config)

    # Logger
    logger = TensorBoardLogger(save_dir=LOGS_DIR,
                               name=f'{experiment}_{fold}, version={version}')


    # Create trainer
    filename = f"{config['base_model']}-{{epoch:02d}}-fold_{fold}-{{val_score:.3f}}"
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=MODELS_DIR,
                                          mode='min',
                                          filename=filename)

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    trainer = Trainer(gpus=1,
                      max_epochs=config['epochs'],
                      precision=config['precision'],
                      num_sanity_val_steps=0,
                      logger=logger,
                      callbacks=[checkpoint_callback, early_stopping]
                      )
    # Free cache
    torch.cuda.empty_cache()

    # Train
    trainer.fit(classifier, datamodule=dm)

if __name__ == '__main__':

    for fold in range(NUM_FOLDS):
        train_fold(experiment='apprentice', version='1', fold=fold)