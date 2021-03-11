import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset

from src.folds import make_folds
from src.datasets import set_random_seed
from src.utils import load_and_concat_preds
from src import config


def get_stacking_folds_data(experiments):
    if not config.train_folds_path.exists():
        make_folds()

    pred_paths = [config.predictions_dir / e / 'val' / 'preds.npz'
                  for e in experiments]
    concat_preds, study_ids = load_and_concat_preds(pred_paths)
    study_id2concat_pred = {s: p for s, p in zip(study_ids, concat_preds)}

    train_df = pd.read_csv(config.train_folds_path)
    train_dict = train_df.to_dict(orient='index')
    folds_data = list()
    for _, sample in train_dict.items():
        study_id = sample['StudyInstanceUID']
        image_name = study_id + '.jpg'
        sample['image_path'] = str(config.train_dir / image_name)
        sample['concat_preds'] = study_id2concat_pred[study_id]

        folds_data.append(sample)
    return folds_data


def get_stacking_test_data(experiments):
    test_data = []

    pred_paths = [config.predictions_dir / exp / 'test' / 'preds.npz'
                  for exp in experiments]
    concat_preds, study_ids = load_and_concat_preds(pred_paths)
    study_id2concat_pred = {s: p for s, p in zip(study_ids, concat_preds)}

    for image_path in sorted(glob.glob(str(config.test_dir / "*.jpg"))):
        study_id = Path(image_path).stem
        sample = {
            'image_path': image_path,
            'StudyInstanceUID': study_id,
            'concat_preds': study_id2concat_pred[study_id]
        }
        test_data.append(sample)
    return test_data


class StackingDataset(Dataset):
    def __init__(self,
                 data,
                 folds=None,
                 size=None,
                 return_target=True):
        super().__init__()
        self.folds = folds
        self.size = size
        self.return_target = return_target

        if folds is None:
            self.data = data
        else:
            self.data = [s for s in data if s['fold'] in folds]

    def __len__(self):
        if self.size is None:
            return len(self.data)
        else:
            return self.size

    def get_sample(self, idx):
        sample = self.data[idx]

        preds = sample['concat_preds'].copy()
        preds = torch.from_numpy(preds)

        if not self.return_target:
            return preds

        target = torch.zeros(config.n_classes, dtype=torch.float32)
        for cls in config.classes:
            target[config.class2target[cls]] = sample[cls]

        return preds, target

    def __getitem__(self, idx):
        set_random_seed(idx)
        idx = np.random.randint(len(self.data))
        preds, target = self.get_sample(idx)
        return preds, target
