config = {
    'seed' : 42,
    'fold_seed': [23, 13, 19, 42, 11],
    'base_model':'resnet200d',
    'base_model_classifier':'fc',
    'classes' : 11,
    'img_size' : 512,
    'precision' : 16,
    'train_batch_size' : 16,
    'val_batch_size' : 64,
    'train_aug': 'training_2_bis',
    'val_aug': 'validation',
    'epochs' : 20,
    'num_workers' : 10,

    # Optimizer and LR scheduling - General
    'weight_decay': 1e-7,
    'lr' : 1e-4,
    'min_lr': 1e-6,
    'scheduler': 'CosineAnnealingLR',

    # CosineAnnealingLR
    't_max': 20,

    # CosineAnnealingWarmupRestarts
    "first_cycle_steps": 1000,
    "warmup_steps": 100,
    "gamma":0.75,

    # ReduceLROnPlateau
    'patience': 2,
    'factor': 0.1

}