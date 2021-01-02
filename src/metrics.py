import numpy as np
from sklearn.metrics import roc_auc_score

import torch

from argus.metrics import Metric


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

    def compute(self):
        y_true = np.concatenate(self.targets, axis=0)
        y_pred = np.concatenate(self.predictions, axis=0)
        score = roc_auc_score(y_true, y_pred)
        return score
