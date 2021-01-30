import torch
from torch.utils.data import DataLoader

from argus import load_model

from src.datasets import RanzcrDataset
from src.transforms import get_transforms


@torch.no_grad()
def predict_data(data, model, batch_size, transform, tta, num_workers=0):

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

        if tta:
            hflip_batch = torch.flip(batch, [3])
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
                 device='cuda',
                 tta=False,
                 num_workers=0):
        self.model = load_model(model_path, device=device)
        self.batch_size = batch_size
        self.transform = get_transforms(False,
                                        self.model.params['image_size'])
        self.tta = tta
        self.num_workers = num_workers

    @torch.no_grad()
    def predict(self, data):
        dataset = RanzcrDataset(data,
                                return_target=False,
                                folds=None,
                                transform=self.transform)
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)

        preds_lst = []
        for batch in loader:
            pred = self.model.predict(batch)

            if self.tta:
                hflip_batch = torch.flip(batch, [3])
                hflip_pred = self.model.predict(hflip_batch)
                pred = 0.5 * pred + 0.5 * hflip_pred

            preds_lst.append(pred)

        pred = torch.cat(preds_lst, dim=0)
        pred = pred.cpu().numpy()
        return pred
