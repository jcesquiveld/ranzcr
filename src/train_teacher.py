import pandas as pd
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from ranzcr.utils import seed
from ranzcr.globals import *
from ranzcr.config import config
from ranzcr.data import RanzcrDataset
from ranzcr.model import RanzcrClassifier
from ranzcr.augmentations import get_augmentations


if __name__ == '__main__':

    FOLD = 0

    # Seed everything
    seed(config['seed'])

    # Read data
    # Read data
    annotated_df = pd.read_csv(os.path.join(INPUT_DIR, 'train_k.csv'))
    train_df = annotated_df.loc[annotated_df.fold != FOLD]
    val_df = annotated_df.loc[annotated_df.fold == FOLD]

    # Data - Train with mixed images but validate with normal images
    train_images = [os.path.join(MIXED_IMAGES, imageId + '.jpg') for imageId in
                    train_df[IMAGE_ID_COLUMN].values]
    val_images = [os.path.join(TRAIN_IMAGES, imageId + '.jpg') for imageId in val_df[IMAGE_ID_COLUMN].values]

    train_labels = train_df[TARGET_COLUMNS].values
    valid_labels = val_df[TARGET_COLUMNS].values

    train_dataset = RanzcrDataset(train_images, train_labels,
                                  transform=get_augmentations(config['train_aug'], config['img_size']))

    valid_dataset = RanzcrDataset(val_images, valid_labels,
                                  transform=get_augmentations(config['val_aug'], config['img_size']))

    train_data_loader = DataLoader(train_dataset,
                          batch_size=config['train_batch_size'],
                          num_workers=config['num_workers'],
                          shuffle=True,
                          pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset,
                                   batch_size=config['val_batch_size'],
                                   num_workers=config['num_workers'],
                                   shuffle=False,
                                   pin_memory=True)
    # Teacher
    teacher = RanzcrClassifier(config)

    # Create trainer
    filename = f"teacher-{config['base_model']}-{{epoch:02d}}{{val_score:.3f}}"

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
        lr_finder = trainer.tuner.lr_find(teacher, train_dataloader=train_data_loader)
        print(f'Suggested lr={lr_finder.suggestion()}')
        fig = lr_finder.plot(suggest=True)
        fig.show()

    # Train
    train = True
    torch.cuda.empty_cache()
    if train:
        trainer.fit(teacher, train_dataloader=train_data_loader, val_dataloaders=valid_data_loader)