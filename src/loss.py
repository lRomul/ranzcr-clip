from torch import nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        loss = (
            self.kl_loss(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
            )
            * self.temperature ** 2
        )
        return loss
