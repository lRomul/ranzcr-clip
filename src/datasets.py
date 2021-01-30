import cv2
import time
import glob
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.folds import make_folds
from src import config


def get_folds_data(lung_masks_dir=config.segm_train_lung_masks_dir,
                   pseudo_label_path=None):
    lung_masks_dir = Path(lung_masks_dir)

    if not config.train_folds_path.exists():
        make_folds()

    pseudo_label_dict = dict()
    if pseudo_label_path is not None:
        pseudo_label = np.load(pseudo_label_path)
        for study_id, pred in zip(pseudo_label['study_ids'],
                                  pseudo_label['logits']):
            pseudo_label_dict[study_id] = pred

    train_df = pd.read_csv(config.train_folds_path)
    train_dict = train_df.to_dict(orient='index')
    folds_dict = dict()
    for _, sample in train_dict.items():
        study_id = sample['StudyInstanceUID']
        image_name = study_id + '.jpg'
        sample['image_path'] = str(config.train_dir / image_name)
        sample['lung_mask_path'] = str(lung_masks_dir / image_name)
        sample['annotations'] = list()
        if study_id in pseudo_label_dict:
            sample['pseudo_label'] = pseudo_label_dict[study_id]
        else:
            sample['pseudo_label'] = None

        folds_dict[sample['StudyInstanceUID']] = sample
    train_annotations_df = pd.read_csv(config.train_annotations_csv_path)
    for ann_sample in train_annotations_df.to_dict(orient='index').values():
        sample = folds_dict[ann_sample['StudyInstanceUID']]
        sample['annotations'].append({
            'label': ann_sample['label'],
            'data': json.loads(ann_sample['data'])
        })
    folds_data = list(folds_dict.values())
    return folds_data


def draw_annotations(image, annotations):
    image = image.copy()
    for annotation in annotations:
        color = config.class2color[annotation['label']]
        point_lst = annotation['data']
        for i in range(len(point_lst) - 1):
            cv2.line(image, (point_lst[i][0], point_lst[i][1]),
                     (point_lst[i + 1][0], point_lst[i + 1][1]), color, 10)


def draw_mask(image, mask):
    mask = cv2.resize(mask, image.shape[:2][::-1])
    image[:, :, 0] = cv2.addWeighted(image[:, :, 0], 0.9, mask, 0.1, 0)


def draw_visualization(sample):
    image = cv2.imread(sample['image_path'])
    image = draw_annotations(image, sample['annotations'])
    image = np.concatenate([image, image], axis=1)

    annotations_set = set()
    for annotation in sample['annotations']:
        annotations_set.add(annotation['label'])

    for cls, trg in config.class2target.items():
        if cls in annotations_set:
            color = config.class2color[cls]
        else:
            color = (255, 255, 255)
        if sample[cls]:
            cls += ' *'
        cv2.putText(image, cls, (50, 70 + trg * 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def get_test_data(lung_masks_dir=None):
    test_data = []
    for image_path in sorted(glob.glob(str(config.test_dir / "*.jpg"))):
        sample = {
            'image_path': image_path,
            'StudyInstanceUID': Path(image_path).stem,
        }
        if lung_masks_dir is not None:
            sample['lung_mask_path'] = str(Path(lung_masks_dir)
                                           / Path(image_path).name)
        test_data.append(sample)
    return test_data


class RanzcrDataset(Dataset):
    def __init__(self,
                 data,
                 folds=None,
                 transform=None,
                 return_target=True,
                 segm=False,
                 annotations=False,
                 pseudo_label=False,
                 rgb=False):
        self.data = data
        self.folds = folds
        self.transform = transform
        self.return_target = return_target
        self.segm = segm
        self.annotations = annotations
        self.rgb = rgb
        self.pseudo_label = pseudo_label
        if folds is not None:
            self.data = [s for s in self.data if s['fold'] in folds]
        if self.annotations:
            assert self.rgb

    def __len__(self):
        return len(self.data)

    def _set_random_seed(self, index):
        seed = int(time.time() * 1000.0) + index
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    def _get_sample(self, index):
        sample = self.data[index]
        if self.rgb:
            image = cv2.imread(sample['image_path'], cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)

        if self.annotations:
            mask = cv2.imread(sample['lung_mask_path'], cv2.IMREAD_GRAYSCALE)
            draw_mask(image, mask)
            draw_annotations(image, sample['annotations'])

        if not self.return_target:
            return image, None

        if self.segm:
            target = cv2.imread(sample['lung_mask_path'], cv2.IMREAD_GRAYSCALE)
            target = (target > 128).astype('float32')
            target = target[..., np.newaxis]
        elif self.pseudo_label:
            target = torch.from_numpy(sample['pseudo_label'].astype('float32'))
        else:
            target = torch.zeros(config.n_classes, dtype=torch.float32)
            for cls in config.classes:
                target[config.class2target[cls]] = sample[cls]

        return image, target

    def __getitem__(self, index):
        self._set_random_seed(index)
        image, target = self._get_sample(index)
        if self.transform is not None:
            if self.segm and self.return_target:
                image, target = self.transform(image, target)
            else:
                image = self.transform(image)
        if target is not None:
            return image, target
        else:
            return image
