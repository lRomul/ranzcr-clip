import torch
from torch.utils.data import DataLoader

from argus import load_model

from src.segm.transforms import get_transforms
from src.datasets import RanzcrDataset


class SegmPredictor:
    def __init__(self,
                 model_path,
                 batch_size,
                 device='cuda',
                 num_workers=0):
        self.model = load_model(model_path, device=device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = get_transforms(False,
                                        self.model.params['image_size'])
        self.rgb = self.model.params['nn_module'][1]['in_channels'] == 3

    @torch.no_grad()
    def predict(self, data):
        dataset = RanzcrDataset(data,
                                return_target=False,
                                folds=None,
                                transform=self.transform,
                                segm=True,
                                rgb=self.rgb)
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)

        for batch in loader:
            pred = self.model.predict(batch)
            pred = pred.cpu().numpy()
            for i in range(pred.shape[0]):
                yield pred[i][0]
