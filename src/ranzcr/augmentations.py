import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentations(name, img_size):

    if name == 'training_none':
        aug = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
    elif name == 'training_dropout':
            aug = A.Compose([
                A.Resize(img_size, img_size),
                A.CoarseDropout(min_height=int(img_size * 0.05), min_width=int(img_size * 0.05),
                                max_height=int(img_size * 0.1), max_width=int(img_size * 0.1),
                                min_holes=1, max_holes=20, p=0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2()
            ])
    elif name == 'training_1':
        aug = A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.9,1), p=1),
            A.ShiftScaleRotate(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, val_shift_limit=10, sat_shift_limit=10, p=0.7),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.3),
            A.OneOf([
                A.ImageCompression(),
                A.Downscale(scale_min=0.1, scale_max=0.15),
            ], p=0.2),
            A.IAAPiecewiseAffine(p=0.2),
            A.IAASharpen(p=0.2),
            A.CoarseDropout(max_height=int(img_size*0.1), max_width=int(img_size*0.1),
                            min_holes=5, max_holes=10, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
    elif name == 'training_2':
        aug = A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.9,1), p=1),
            A.ShiftScaleRotate(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, val_shift_limit=10, sat_shift_limit=10, p=0.7),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.3),
            A.OneOf([
                A.ImageCompression(),
                A.Downscale(scale_min=0.1, scale_max=0.15),
            ], p=0.2),
            A.IAAPiecewiseAffine(p=0.2),
            A.IAASharpen(p=0.2),
            A.CoarseDropout(max_height=int(img_size*0.1), max_width=int(img_size*0.1),
                            min_holes=5, max_holes=10, p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
    elif name == 'training_2_bis':
        aug = A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.9, 1), p=1),
            A.ShiftScaleRotate(rotate_limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, val_shift_limit=10, sat_shift_limit=10, p=0.7),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.OneOf([A.GaussNoise(var_limit=[10, 50]), A.GaussianBlur(), A.MotionBlur(), A.MedianBlur()], p=0.3),
            #A.OneOf([A.OpticalDistortion(distort_limit=1.0), A.GridDistortion(num_steps=5, distort_limit=1.),
            #         A.ElasticTransform(alpha=3)], p=0.3),
            A.OneOf([A.ImageCompression(), A.Downscale(scale_min=0.1, scale_max=0.15)], p=0.2),
            #A.IAAPiecewiseAffine(p=0.2),
            A.IAASharpen(p=0.2),
            A.CoarseDropout(max_height=int(img_size*0.1),max_width=int(img_size*0.1),min_holes=5,max_holes=10,p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
    elif name == 'training_3':
        aug = A.Compose([
            A.Rotate(limit=5),
            A.RandomResizedCrop(img_size, img_size, scale=(0.9, 1), p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.CoarseDropout(min_height=int(img_size * 0.05), min_width=int(img_size * 0.05),
                            max_height=int(img_size * 0.1), max_width=int(img_size * 0.1),
                            min_holes=1, max_holes=10, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
    elif name == 'training_4':
        aug = A.Compose([
            A.Rotate(limit=5, p=1),
            A.RandomResizedCrop(img_size, img_size, scale=(0.9, 1), p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.15,+0.25),
                                       contrast_limit=(-0.15,+0.25), p=1),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(),
            ], p=1),
            A.IAASharpen(p=0.3),
            A.CoarseDropout(min_height=int(img_size*0.05), min_width=int(img_size*0.05),
                    max_height=int(img_size * 0.1), max_width=int(img_size * 0.1),
                    min_holes=1, max_holes=20, p=0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
    elif name == 'validation':
        aug = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        raise ValueError(f"{name} is not a valid augmentations name")

    return aug