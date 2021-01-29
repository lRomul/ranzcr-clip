import numpy as np
from sklearn.metrics import roc_auc_score

import torch

from argus.metrics import Metric

from src import config


class RocAuc(Metric):
    name = 'roc_auc'
    better = 'max'

    def __init__(self):
        self.predictions = []
        self.targets = []

    def reset(self):
        self.predictions = []
        self.targets = []

    @torch.no_grad()
    def update(self, step_output: dict):
        pred = step_output['prediction'].cpu().numpy()
        target = step_output['target'].cpu().numpy()

        self.predictions.append(pred)
        self.targets.append(target)

    def compute(self) -> list:
        y_true = np.concatenate(self.targets, axis=0)
        y_pred = np.concatenate(self.predictions, axis=0)
        score = roc_auc_score(y_true, y_pred, average=None)
        return score

    def epoch_complete(self, state):
        with torch.no_grad():
            scores = self.compute()
        name_prefix = f"{state.phase}_" if state.phase else ''
        state.metrics[f"{name_prefix}{self.name}"] = np.mean(scores)
        for trg, cls in config.target2class.items():
            task = cls.split(' - ')[0].split(' ')[-1]
            cls = cls.split(' - ')[-1].split(' ')[0]
            state.metrics[f"{name_prefix}{self.name}_{task}_{cls}"] = scores[trg]
