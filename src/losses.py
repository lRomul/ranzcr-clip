import torch.nn.functional as F
from torch import nn, Tensor


class SoftBCEWithLogitsLoss(nn.Module):
    def __init__(self, smooth_factor=None):
        super().__init__()
        self.smooth_factor = smooth_factor

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.smooth_factor is not None:
            soft_targets = (1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)
        else:
            soft_targets = target

        loss = F.binary_cross_entropy_with_logits(input, soft_targets)
        return loss
