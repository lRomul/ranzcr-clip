import random
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from src import config


def make_folds():
    random_state = 4656

    random.seed(random_state)
    np.random.seed(random_state)

    train_df = pd.read_csv(config.train_csv_path, index_col=0)

    corrections_df = pd.read_csv(config.corrections_csv_path, index_col=0)

    for study_id, row in corrections_df.iterrows():
        train_df.loc[study_id, row.label] = 0
        train_df.loc[study_id, row.new_label] = 1

    patient_id_lst = []
    patient_target_lst = []
    for patient_id, group in train_df.groupby(by='PatientID'):
        patient_target = group[config.classes].sum(axis=0).values
        patient_target = np.clip(patient_target, 0, 1)
        patient_target = np.dot(patient_target, [2 ** i for i in range(len(config.classes))])

        patient_id_lst.append(patient_id)
        patient_target_lst.append(patient_target)

    skf = StratifiedKFold(n_splits=config.n_folds, random_state=random_state, shuffle=True)

    patient_id2fold = dict()
    for fold, (_, val_index) in enumerate(skf.split(patient_id_lst, patient_target_lst)):
        for index in val_index:
            patient_id2fold[patient_id_lst[index]] = fold

    train_df['fold'] = train_df.PatientID.map(patient_id2fold)

    train_df.to_csv(config.train_folds_path)
    print(f"Train folds saved to '{config.train_folds_path}'")


if __name__ == '__main__':
    make_folds()
