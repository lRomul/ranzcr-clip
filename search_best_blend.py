import argparse
import itertools
import numpy as np
import pandas as pd
from pprint import pprint
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score

from src import config
from src.utils import load_and_blend_preds


parser = argparse.ArgumentParser()
parser.add_argument('--max', default=7, type=int)
parser.add_argument('--workers', default=32, type=int)
parser.add_argument('--folder', default='', type=str)
args = parser.parse_args()


def experiments_blend_score(experiments,
                            multipliers=None,
                            folder=args.folder):
    pred_paths = [config.predictions_dir / folder / e / 'val' / 'preds.npz'
                  for e in experiments]
    blend_preds, study_ids = load_and_blend_preds(pred_paths,
                                                  multipliers=multipliers)

    train_df = pd.read_csv(config.train_folds_path, index_col=0)
    train_df = train_df.loc[study_ids].copy()
    scores = roc_auc_score(train_df[config.classes].values,
                           blend_preds, average=None)
    return np.mean(scores)


if __name__ == "__main__":
    experiments = []
    for experiment_dir in (config.predictions_dir / args.folder).iterdir():
        if experiment_dir.is_dir():
            if ',' not in experiment_dir.name:  # filter blend predictions
                experiments.append(experiment_dir.name)

    combinations = []
    for num_exp in range(1, min(len(experiments), args.max) + 1):
        combinations += itertools.combinations(experiments, num_exp)

    with Pool(processes=args.workers) as pool:
        scores = pool.map(experiments_blend_score, combinations)

    experiments_scores = [(','.join(sorted(exp)), scr) for exp, scr
                          in zip(combinations, scores)]
    experiments_scores = sorted(experiments_scores, key=lambda x: x[1])

    pprint(experiments_scores)
