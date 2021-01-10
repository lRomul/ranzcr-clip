import cv2
import time
import glob
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src import config


def get_folds_data():
    train_df = pd.read_csv(config.train_folds_path)
    train_dict = train_df.to_dict(orient='index')
    folds_dict = dict()
    for _, sample in train_dict.items():
        sample['image_path'] = str(config.train_dir /
                                   (sample['StudyInstanceUID'] + '.jpg'))
        sample['annotations'] = list()
        folds_dict[sample['StudyInstanceUID']] = sample
    train_annotations_df = pd.read_csv(config.train_annotations_csv_path)
    for ann_sample in train_annotations_df.to_dict(orient='index').values():
        sample = folds_dict[ann_sample['StudyInstanceUID']]
        sample['annotations'].append({
            'label': ann_sample['label'],
            'data': ann_sample['data']
        })
    folds_data = list(folds_dict.values())
    return folds_data


def get_test_data():
    test_data = []
    for image_path in glob.glob(str(config.test_dir / "*.jpg")):
        test_data.append({
            'image_path': image_path,
            'StudyInstanceUID': Path(image_path).stem
        })
    return test_data


class RanzcrDataset(Dataset):
    def __init__(self,
                 data,
                 folds=None,
                 image_transform=None,
                 return_target=True):
        self.data = data
        self.folds = folds
        self.image_transform = image_transform
        self.return_target = return_target
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

        if not self.return_target:
            return image, None

        target = torch.zeros(config.n_classes, dtype=torch.float32)
        for cls in config.classes:
            target[config.class2target[cls]] = sample[cls]
        return image, target

    def __getitem__(self, index):
        self._set_random_seed(index)
        image, target = self._get_sample(index)
        if self.image_transform is not None:
            image = self.image_transform(image)
        if target is not None:
            return image, target
        else:
            return image
