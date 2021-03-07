from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torch
import pandas as pd
from ranzcr.globals import *
import os
from ranzcr.augmentations import get_augmentations
from ranzcr.data import RanzcrDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from ranzcr.config import config
from ranzcr.model import RanzcrClassifier, RanzcrStudentClassifier
from ranzcr.data import TeacherStudentDataset


if __name__ == '__main__':

    FOLD = 0
    TEACHER_CHECKPOINT = '../models/teacher/teacher-resnest101e-epoch=06val_score=0.998.ckpt'

    # Read data
    annotated_df = pd.read_csv(os.path.join(INPUT_DIR, 'annotated_train_k.csv'))
    train_df = annotated_df.loc[annotated_df.fold != FOLD]
    val_df = annotated_df.loc[annotated_df.fold == FOLD]

    # Data - Images are the same for both, but annotated for the teacher and not annotated for the student
    teacher_train_images = [os.path.join(ANNOTATED_IMAGES, imageId + '.jpg') for imageId in train_df[IMAGE_ID_COLUMN].values]
    student_train_images = [os.path.join(TRAIN_IMAGES, imageId + '.jpg') for imageId in train_df[IMAGE_ID_COLUMN].values]
    student_val_images = [os.path.join(TRAIN_IMAGES, imageId + '.jpg') for imageId in val_df[IMAGE_ID_COLUMN].values]
    train_labels = train_df[TARGET_COLUMNS].values
    valid_labels = val_df[TARGET_COLUMNS].values


    train_aug = get_augmentations('training_2_bis', 512)
    valid_aug = get_augmentations('validation', 512)

    train_dataset = TeacherStudentDataset(teacher_train_images, student_train_images, train_labels, transform=train_aug)
    valid_dataset = RanzcrDataset(student_val_images, valid_labels, transform=valid_aug)

    train_data_loader = DataLoader(train_dataset,
                             batch_size=config['train_batch_size'],
                             shuffle=True,
                             num_workers=20,
                             pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset,
                                   batch_size=config['val_batch_size'],
                                   shuffle=False,
                                   num_workers=20,
                                   pin_memory=True)


    filename = f"student-{config['base_model']}-{{epoch:02d}}{{val_score:.3f}}"

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=MODELS_DIR,
                                          mode='min',
                                          filename=filename)

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)

    # Create trainer


    trainer = Trainer(gpus=1,
                      max_epochs=config['epochs'],
                      precision=config['precision'],
                      num_sanity_val_steps=0,
                      callbacks=[checkpoint_callback, early_stopping]
                      )

    teacher = RanzcrClassifier.load_from_checkpoint(TEACHER_CHECKPOINT)
    student = RanzcrStudentClassifier(config, teacher)
    # Find best learning rate
    tune = False
    if tune:
        lr_finder = trainer.tuner.lr_find(student, train_dataloader=train_data_loader)
        print(f'Suggested lr={lr_finder.suggestion()}')
        fig = lr_finder.plot(suggest=True)
        fig.show()

    # Train
    train = True
    torch.cuda.empty_cache()
    if train:
        trainer.fit(student, train_dataloader=train_data_loader, val_dataloaders=valid_data_loader)