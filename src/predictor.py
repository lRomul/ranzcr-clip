import torch
from torch.utils.data import DataLoader

from argus import load_model

from src.datasets import RanzcrDataset


@torch.no_grad()
def predict_data(data, model, batch_size, transform,
                 tta=False, num_workers=0):

    dataset = RanzcrDataset(data,
                            return_target=False,
                            folds=None,
                            image_transform=transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers)

    preds_lst = []
    for batch in loader:
        pred = model.predict(batch)

        if tta:
            hflip_batch = torch.flip(batch, [2])
            hflip_pred = model.predict(hflip_batch)
            pred = 0.5 * pred + 0.5 * hflip_pred

        preds_lst.append(pred)

    pred = torch.cat(preds_lst, dim=0)
    pred = pred.cpu().numpy()

    return pred


class Predictor:
    def __init__(self,
                 model_path,
                 batch_size,
                 image_transform,
                 device='cuda',
                 tta=False,
                 num_workers=0):
        self.model = load_model(model_path, device=device)
        self.batch_size = batch_size
        self.image_transform = image_transform
        self.tta = tta
        self.num_workers = num_workers

    def predict(self, data):
        pred = predict_data(data,
                            self.model,
                            self.batch_size,
                            self.image_transform,
                            self.tta,
                            self.num_workers)
        return pred
