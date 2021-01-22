import numpy as np
from sklearn.metrics import roc_auc_score

import torch

from argus.metrics import Metric
from argus.utils import AverageMeter

EPSILON = 1e-15


def binary_mean_iou(probs: torch.Tensor,
                    targets: torch.Tensor,
                    threshold=0.5) -> torch.Tensor:
    output = (probs > threshold).int()
    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)
    intersection = (targets * output).sum()
    union = targets.sum() + output.sum() - intersection
    result = (intersection + EPSILON) / (union + EPSILON)
    return result.item()


class BinaryIou(Metric):
    name = 'binary_iou'
    better = 'max'

    def __init__(self):
        self.average = AverageMeter()

    def reset(self):
        self.average.reset()

    @torch.no_grad()
    def update(self, step_output: dict):
        pred = step_output['prediction']
        target = step_output['target']
        loss = binary_mean_iou(pred, target, 0.5)
        self.average.update(loss)

    def compute(self):
        return self.average.average
