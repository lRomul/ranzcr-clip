import argparse
import numpy as np
from pprint import pprint

from src.utils import get_best_model_path

from src import config


def get_stack_mean_score(path):
    scores = []
    for fold in config.folds:
        fold_path = path / f'fold_{fold}'
        _, score = get_best_model_path(fold_path, return_score=True)
        scores.append(score)

    return np.mean(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, type=str)
    args = parser.parse_args()

    stack_search_dir = config.experiments_dir / args.experiment

    exp_scores = []
    for path in stack_search_dir.iterdir():
        if path.is_dir():
            score = get_stack_mean_score(path)
            exp_scores.append((path, score))

    pprint(sorted(exp_scores, key=lambda x: x[1]))
