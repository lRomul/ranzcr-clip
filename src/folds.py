import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from src import config


def make_folds():
    random_state = 42

    random.seed(random_state)
    np.random.seed(random_state)

    train_df = pd.read_csv(config.train_csv_path)

    patient_ids = sorted(train_df.PatientID.unique())
    kf = KFold(n_splits=config.n_folds, random_state=random_state, shuffle=True)

    patient_id2fold = dict()
    for fold, (_, val_index) in enumerate(kf.split(patient_ids)):
        for index in val_index:
            patient_id2fold[patient_ids[index]] = fold

    train_df['fold'] = train_df.PatientID.map(patient_id2fold)

    train_df.to_csv(config.train_folds_path, index=False)
    print(f"Train folds saved to '{config.train_folds_path}'")


if __name__ == '__main__':
    make_folds()
