import cv2
import time
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from src import config


def get_folds_data():
    train_df = pd.read_csv(config.train_folds_path)
    train_dict = train_df.to_dict(orient='index')
    folds_data = []
    for _, sample in train_dict.items():
        sample['image_path'] = str(config.train_dir /
                                   (sample['StudyInstanceUID'] + '.jpg'))
        folds_data.append(sample)
    return folds_data


class RanzcrDataset(Dataset):
    def __init__(self,
                 data,
                 folds=None,
                 image_transform=None):
        self.data = data
        self.folds = folds
        self.image_transform = image_transform
        if folds is not None:
            self.data = [s for s in self.data if s['fold'] in folds]

    def __len__(self):
        return len(self.data)

    def _set_random_seed(self, index):
        seed = int(time.time() * 1000.0) + index
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    def _get_sample(self, index):
        sample = self.data[index]
        image = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)

        target = torch.zeros(config.n_classes, dtype=torch.float32)
        for cls in config.classes:
            target[config.class2target[cls]] = sample[cls]
        return image, target

    def __getitem__(self, index):
        self._set_random_seed(index)
        image, target = self._get_sample(index)
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image, target
