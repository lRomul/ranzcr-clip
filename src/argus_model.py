import torch
import timm

import argus
from argus.utils import deep_to, deep_detach
from argus.loss import pytorch_losses

from src.losses import SoftBCEWithLogitsLoss


class RanzcrModel(argus.Model):
    nn_module = {
        "timm": timm.create_model
    }
    loss = {
        'SoftBCEWithLogitsLoss': SoftBCEWithLogitsLoss,
        **pytorch_losses
    }
    prediction_transform = torch.nn.Sigmoid

    def __init__(self, params: dict):
        super().__init__(params)
        self.amp = 'amp' in params and params['amp']
        self.clip_grad = 'clip_grad' in params and params['clip_grad']
        self.scaler = torch.cuda.amp.GradScaler()
        self.logger.info(f"amp: {self.amp}, clip_grad: {self.clip_grad}")

    def train_step(self, batch, state) -> dict:
        self.train()
        self.optimizer.zero_grad()
        input, target = deep_to(batch, device=self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.amp):
            prediction = self.nn_module(input)
            loss = self.loss(prediction, target)
        self.scaler.scale(loss).backward()
        if self.clip_grad:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.nn_module.parameters(),
                                           max_norm=2.0, norm_type=2.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item()
        }
