import argparse
import itertools
import numpy as np
import pandas as pd
from pprint import pprint
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score

from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--max', default=7, type=int)
parser.add_argument('--workers', default=32, type=int)
args = parser.parse_args()


def experiments_blend_score(experiments):
    pred_lst = []
    study_ids = None
    for experiment in experiments:
        pred_path = config.predictions_dir / experiment / 'val' / 'preds.npz'
        pred_npz = np.load(pred_path)
        preds = pred_npz['preds']
        if study_ids is not None:
            assert np.all(study_ids == pred_npz['study_ids'])
        study_ids = pred_npz['study_ids']
        pred_lst.append(preds)

    preds = np.mean(pred_lst, axis=0)

    train_df = pd.read_csv(config.train_folds_path, index_col=0)
    train_df = train_df.loc[study_ids].copy()
    scores = roc_auc_score(train_df[config.classes].values,
                           preds, average=None)
    return np.mean(scores)


if __name__ == "__main__":
    experiments = []
    for experiment_dir in config.predictions_dir.iterdir():
        if experiment_dir.is_dir():
            experiments.append(experiment_dir.name)

    combinations = []
    for num_exp in range(1, min(len(experiments), args.max) + 1):
        combinations += itertools.combinations(experiments, num_exp)

    with Pool(processes=args.workers) as pool:
        scores = pool.map(experiments_blend_score, combinations)

    experiments_scores = [(sorted(exp), scr) for exp, scr
                          in zip(combinations, scores)]
    experiments_scores = sorted(experiments_scores, key=lambda x: x[1])

    pprint(experiments_scores)
