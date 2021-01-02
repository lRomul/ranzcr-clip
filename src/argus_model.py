import torch
import timm

import argus
from argus.utils import deep_to, deep_detach


class RanzcrModel(argus.Model):
    nn_module = {
        "timm": timm.create_model
    }

    def __init__(self, params: dict):
        super().__init__(params)
        self.amp = 'amp' in params and params['amp']
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(self, batch, state) -> dict:
        self.train()
        self.optimizer.zero_grad()
        input, target = deep_to(batch, device=self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.amp):
            prediction = self.nn_module(input)
            loss = self.loss(prediction, target)
        self.scaler.scale(loss).backward()
        self.optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item()
        }
