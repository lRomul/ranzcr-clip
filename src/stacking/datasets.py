import glob
import torch
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset

from src.folds import make_folds
from src.datasets import load_pseudo_label
from src.utils import load_and_concat_preds
from src import config


def get_stacking_folds_data(experiments, pseudo_label_path=None):
    if not config.train_folds_path.exists():
        make_folds()

    pred_paths = [config.predictions_dir / e / 'val' / 'preds.npz'
                  for e in experiments]
    concat_preds, study_ids = load_and_concat_preds(pred_paths)
    study_id2concat_pred = {s: p for s, p in zip(study_ids, concat_preds)}

    pseudo_label_dict = dict()
    if pseudo_label_path is not None:
        pseudo_label_dict = load_pseudo_label(pseudo_label_path)

    train_df = pd.read_csv(config.train_folds_path)
    train_dict = train_df.to_dict(orient='index')
    folds_data = list()
    for _, sample in train_dict.items():
        study_id = sample['StudyInstanceUID']
        image_name = study_id + '.jpg'
        sample['image_path'] = str(config.train_dir / image_name)
        sample['concat_preds'] = study_id2concat_pred[study_id]

        if study_id in pseudo_label_dict:
            sample['pseudo_label'] = pseudo_label_dict[study_id]
        else:
            sample['pseudo_label'] = None

        folds_data.append(sample)
    return folds_data


def get_stacking_test_data(experiments, pseudo_label_path=None):
    test_data = []

    pred_paths = [config.predictions_dir / exp / 'test' / 'preds.npz'
                  for exp in experiments]
    concat_preds, study_ids = load_and_concat_preds(pred_paths)
    study_id2concat_pred = {s: p for s, p in zip(study_ids, concat_preds)}

    pseudo_label_dict = dict()
    if pseudo_label_path is not None:
        pseudo_label_dict = load_pseudo_label(pseudo_label_path)

    for image_path in sorted(glob.glob(str(config.test_dir / "*.jpg"))):
        study_id = Path(image_path).stem
        sample = {
            'image_path': image_path,
            'StudyInstanceUID': study_id,
            'concat_preds': study_id2concat_pred[study_id]
        }
        if study_id in pseudo_label_dict:
            sample['pseudo_label'] = pseudo_label_dict[study_id]
        else:
            sample['pseudo_label'] = None
        test_data.append(sample)
    return test_data


class StackingDataset(Dataset):
    def __init__(self,
                 data,
                 folds=None,
                 return_target=True,
                 pseudo_label=False,
                 pseudo_threshold=None):
        super().__init__()
        self.folds = folds
        self.return_target = return_target
        self.pseudo_label = pseudo_label
        self.pseudo_threshold = pseudo_threshold

        if folds is None:
            self.data = data
        else:
            self.data = [s for s in data if s['fold'] in folds]

    def __len__(self):
        return len(self.data)

    def get_sample(self, idx):
        sample = self.data[idx]

        preds = sample['concat_preds'].copy()
        preds = torch.from_numpy(preds)

        if not self.return_target:
            return preds

        if self.pseudo_label:
            target = sample['pseudo_label']
            if self.pseudo_threshold is not None:
                target = target > self.pseudo_threshold
            target = torch.from_numpy(target.astype('float32'))
        else:
            target = torch.zeros(config.n_classes, dtype=torch.float32)
            for cls in config.classes:
                target[config.class2target[cls]] = sample[cls]

        return preds, target

    def __getitem__(self, idx):
        return self.get_sample(idx)
