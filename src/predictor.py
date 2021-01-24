import torch
from torch.utils.data import DataLoader

from argus import load_model

from src.datasets import RanzcrDataset
from src.transforms import get_transforms
from src import config


class Predictor:
    def __init__(self,
                 model_paths,
                 batch_size,
                 device='cuda',
                 num_workers=0):
        self.models = [load_model(p, device=device) for p in model_paths]
        self.tasks = [m.params['task'] for m in self.models]
        assert self.tasks == list(config.sub_classes)
        self.transforms = [get_transforms(False, m.params['image_size'])
                           for m in self.models]
        self.crop_settings = {t: m.params['crop_settings'][t]
                              for t, m in zip(self.tasks, self.models)}
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.no_grad()
    def predict(self, data):
        dataset = RanzcrDataset(data,
                                crop_settings=self.crop_settings,
                                return_target=False,
                                folds=None,
                                transform=self.transforms)
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)

        preds_lst = []
        for batch in loader:
            task_preds_lst = []
            for task_batch, model in zip(batch, self.models):
                task_pred = model.predict(task_batch).cpu()
                task_preds_lst.append(task_pred)
            preds_lst.append(torch.cat(task_preds_lst, dim=1))
        pred = torch.cat(preds_lst, dim=0).numpy()
        return pred
