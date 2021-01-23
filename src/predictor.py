import torch
from torch.utils.data import DataLoader

from argus import load_model

from src.datasets import RanzcrDataset


@torch.no_grad()
def predict_data(data, model, batch_size, transform, num_workers=0):

    dataset = RanzcrDataset(data,
                            return_target=False,
                            folds=None,
                            transform=transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers)

    preds_lst = []
    for batch in loader:
        pred = model.predict(batch)
        preds_lst.append(pred)

    pred = torch.cat(preds_lst, dim=0)
    pred = pred.cpu().numpy()

    return pred


class Predictor:
    def __init__(self,
                 model_path,
                 batch_size,
                 transform,
                 device='cuda',
                 num_workers=0):
        self.model = load_model(model_path, device=device)
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

    def predict(self, data):
        pred = predict_data(data,
                            self.model,
                            self.batch_size,
                            self.transform,
                            self.num_workers)
        return pred
