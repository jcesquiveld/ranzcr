from sklearn.model_selection import GroupKFold
import pandas as pd
import os
from ranzcr.globals import *

if __name__ == '__main__':

    # Create folds only in 2020 training data
    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))

    # Create folds
    gkf = GroupKFold(n_splits=5)
    groups = train_df[PATIENT_ID_COLUMN].values
    for f, (t_, v_) in enumerate(gkf.split(train_df, train_df[TARGET_COLUMNS], groups)):
        train_df.loc[v_, 'fold'] = f
    train_df['fold'] = train_df['fold'].astype(int)

    # Save train dataframe with folds
    train_df.to_csv(os.path.join(INPUT_DIR, 'train_k.csv'), index=False)
    print(train_df.groupby('fold').size())

