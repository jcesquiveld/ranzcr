# Global constants

INPUT_DIR = '../input'
TRAIN_IMAGES = '../input/train'
TEST_IMAGES = '../input/test'
MODELS_DIR = '../models'
LOGS_DIR = '../logs'

TARGET_COLUMNS = ['ETT - Abnormal', 'ETT - Borderline',
       'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
       'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal',
       'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

PATIENT_ID_COLUMN = 'PatientID'
IMAGE_ID_COLUMN = 'StudyInstanceUID'

NUM_FOLDS = 5
GLOBAL_SEED = 42