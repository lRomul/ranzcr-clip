import torch
from torch.utils.data import DataLoader, TensorDataset

from argus import load_model


class StackPredictor:
    def __init__(self,
                 model_path,
                 batch_size, device='cuda'):
        self.model = load_model(model_path, device=device)
        self.batch_size = batch_size

    def predict(self, preds):
        dataset = TensorDataset(preds.copy())
        loader = DataLoader(dataset, batch_size=self.batch_size)

        preds_lst = []
        for batch in loader:
            pred_batch = self.model.predict(batch)
            preds_lst.append(pred_batch.cpu().numpy())

        pred = torch.cat(preds_lst, dim=0)
        return pred
