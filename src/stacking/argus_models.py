import torch

from argus import Model
from argus.utils import deep_to, deep_detach, deep_chunk

from src.stacking.models import FCNet
from src.agc import adaptive_clip_grad


class StackingModel(Model):
    nn_module = {
        'FCNet': FCNet
    }
    prediction_transform = torch.nn.Sigmoid

    def __init__(self, params: dict):
        super().__init__(params)
        self.iter_size = (1 if 'iter_size' not in self.params
                          else int(self.params['iter_size']))
        self.amp = (False if 'amp' not in self.params
                    else bool(self.params['amp']))
        self.clip_grad = (0. if 'clip_grad' not in self.params
                          else float(self.params['clip_grad']))
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None
        self.model_ema = None

    def train_step(self, batch, state) -> dict:
        self.train()
        self.optimizer.zero_grad()

        # Gradient accumulation
        for i, chunk_batch in enumerate(deep_chunk(batch, self.iter_size)):
            input, target = deep_to(chunk_batch, self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                prediction = self.nn_module(input)
                loss = self.loss(prediction, target)
                loss = loss / self.iter_size

            if self.amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        if self.clip_grad:
            if self.amp:
                self.scaler.unscale_(self.optimizer)
            adaptive_clip_grad(self.nn_module.parameters(),
                               clip_factor=self.clip_grad)

        if self.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.model_ema is not None:
            self.model_ema.update(self.nn_module)

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item()
        }

    def val_step(self, batch, state) -> dict:
        self.eval()
        with torch.no_grad():
            input, target = deep_to(batch, device=self.device, non_blocking=True)
            if self.model_ema is None:
                prediction = self.nn_module(input)
            else:
                prediction = self.model_ema.ema(input)
            loss = self.loss(prediction, target)
            prediction = self.prediction_transform(prediction)
            return {
                'prediction': prediction,
                'target': target,
                'loss': loss.item()
            }

    def predict(self, input):
        self._check_predict_ready()
        with torch.no_grad():
            self.eval()
            input = deep_to(input, self.device)
            if self.model_ema is None:
                prediction = self.nn_module(input)
            else:
                prediction = self.model_ema.ema(input)
            prediction = self.prediction_transform(prediction)
            return prediction
