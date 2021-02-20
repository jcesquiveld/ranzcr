import numpy as np
from PIL import Image
import os

from torch.utils.data import Dataset, DataLoader
import torch
from pytorch_lightning import LightningDataModule

# Constant definitions
from .globals import *

from .augmentations import get_augmentations

### Classes ###

class RanzcrDataset(Dataset):

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pil_image = Image.open(self.images[idx]).convert('RGB')
        image = np.array(pil_image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(self.labels[idx], dtype=torch.float32)


class RanzcrDataModule(LightningDataModule):

    def __init__(self, path, train_df, val_df, config):
        super().__init__()
        self.path = path
        self.train_df = train_df
        self.val_df = val_df
        self.config = config

    def setup(self, stage=None):

        # Create train dataset
        train_images = [os.path.join(self.path, image_id + '.jpg') for image_id in self.train_df[IMAGE_ID_COLUMN].values]
        train_labels = self.train_df[TARGET_COLUMNS].values
        self.train_dataset = RanzcrDataset(train_images, train_labels,
                                           get_augmentations(self.config['train_aug'], self.config['img_size']))

        # Create val dataset
        val_images = [os.path.join(self.path, image_id + '.jpg') for image_id in self.val_df[IMAGE_ID_COLUMN].values]
        val_labels = self.val_df[TARGET_COLUMNS].values
        self.val_dataset = RanzcrDataset(val_images, val_labels,
                                         get_augmentations(self.config['val_aug'], self.config['img_size']))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config['train_batch_size'],
                          num_workers=self.config['num_workers'],
                          shuffle=True,
                          pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config['val_batch_size'],
                          num_workers=self.config['num_workers'],
                          shuffle=False,
                          pin_memory=False)
