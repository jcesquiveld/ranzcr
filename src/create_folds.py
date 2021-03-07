from sklearn.model_selection import GroupKFold
import pandas as pd
import os
from ranzcr.globals import *

# I know this should be a parameter, but for now...
#TRAIN_FILE = 'train.csv'
#TRAIN_K_FOLD_FILE = 'train_k.csv'
TRAIN_FILE = 'annotated_train.csv'
TRAIN_K_FOLD_FILE = 'annotated_train_k.csv'


if __name__ == '__main__':

    # Create folds only in 2020 training data
    train_df = pd.read_csv(os.path.join(INPUT_DIR, TRAIN_FILE))

    # Create folds
    gkf = GroupKFold(n_splits=5)
    groups = train_df[PATIENT_ID_COLUMN].values
    for f, (t_, v_) in enumerate(gkf.split(train_df, train_df[TARGET_COLUMNS], groups)):
        train_df.loc[v_, 'fold'] = f
    train_df['fold'] = train_df['fold'].astype(int)

    # Save train dataframe with folds
    train_df.to_csv(os.path.join(INPUT_DIR, TRAIN_K_FOLD_FILE), index=False)
    print(train_df.groupby('fold').size())

