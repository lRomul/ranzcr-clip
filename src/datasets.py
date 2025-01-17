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


def load_pseudo_label(path):
    pseudo_label_dict = dict()
    if path is not None:
        pseudo_label = np.load(path)
        for study_id, pred in zip(pseudo_label['study_ids'],
                                  pseudo_label['preds']):
            pseudo_label_dict[study_id] = pred
    return pseudo_label_dict


def get_folds_data(pseudo_label_path=None):
    if not config.train_folds_path.exists():
        make_folds()

    pseudo_label_dict = dict()
    if pseudo_label_path is not None:
        pseudo_label_dict = load_pseudo_label(pseudo_label_path)

    train_df = pd.read_csv(config.train_folds_path)
    train_dict = train_df.to_dict(orient='index')
    folds_dict = dict()
    for _, sample in train_dict.items():
        study_id = sample['StudyInstanceUID']
        image_name = study_id + '.jpg'
        sample['image_path'] = str(config.train_dir / image_name)
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


def get_test_data(pseudo_label_path=None):
    test_data = []

    pseudo_label_dict = dict()
    if pseudo_label_path is not None:
        pseudo_label_dict = load_pseudo_label(pseudo_label_path)

    for image_path in sorted(glob.glob(str(config.test_dir / "*.jpg"))):
        study_id = Path(image_path).stem
        sample = {
            'image_path': image_path,
            'StudyInstanceUID': study_id,
        }
        if study_id in pseudo_label_dict:
            sample['pseudo_label'] = pseudo_label_dict[study_id]
        else:
            sample['pseudo_label'] = None
        test_data.append(sample)
    return test_data


def get_chest_xrays_data(classes=None, pseudo_label_path=None):
    test_data = []
    if classes is None:
        classes = []

    pseudo_label_dict = dict()
    if pseudo_label_path is not None:
        pseudo_label_dict = load_pseudo_label(pseudo_label_path)

    with open(config.nih_chest_xrays_dir / 'test_list.txt', 'r') as file:
        test_set = {s.strip().split('.')[0] for s in file.readlines()}

    train_df = pd.read_csv(config.nih_chest_xrays_dir / 'Data_Entry_2017.csv')
    id2labels = dict()
    for i, row in train_df.iterrows():
        id2labels[row['Image Index'].split('.')[0]] = row['Finding Labels']

    dir_path = config.nih_chest_xrays_dir / 'images_*' / 'images' /"*.png"
    for image_path in sorted(glob.glob(str(dir_path))):
        study_id = Path(image_path).stem

        sample = {
            'image_path': image_path,
            'StudyInstanceUID': study_id,
            'fold': 0 if study_id in test_set else 1
        }
        for cls in classes:
            sample[cls] = int(cls in id2labels[study_id])
        if study_id in pseudo_label_dict:
            sample['pseudo_label'] = pseudo_label_dict[study_id]
        else:
            sample['pseudo_label'] = None
        test_data.append(sample)
    return test_data


def set_random_seed(index):
    seed = int(time.time() * 1000.0) + index
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


class RanzcrDataset(Dataset):
    def __init__(self,
                 data,
                 folds=None,
                 transform=None,
                 return_target=True,
                 annotations=False,
                 pseudo_label=False,
                 pseudo_threshold=None,
                 length=None,
                 classes=None):
        self.data = data
        self.folds = folds
        self.transform = transform
        self.return_target = return_target
        self.annotations = annotations
        self.pseudo_label = pseudo_label
        self.pseudo_threshold = pseudo_threshold
        self.length = length
        self.classes = classes
        if folds is not None:
            self.data = [s for s in self.data if s['fold'] in folds]

    def __len__(self):
        if self.length is None:
            return len(self.data)
        else:
            return self.length

    def _get_sample(self, index):
        sample = self.data[index]

        if self.annotations:
            image = cv2.imread(sample['image_path'], cv2.IMREAD_COLOR)
            draw_annotations(image, sample['annotations'])
        else:
            if self.transform.n_channels == 3:
                image = cv2.imread(sample['image_path'], cv2.IMREAD_COLOR)
            elif self.transform.n_channels == 1:
                image = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)
            else:
                raise ValueError

        if not self.return_target:
            return image, None

        if self.pseudo_label:
            target = sample['pseudo_label']
            if self.pseudo_threshold is not None:
                target = target > self.pseudo_threshold
            target = torch.from_numpy(target.astype('float32'))
        elif self.classes is not None:
            target = torch.zeros(len(self.classes), dtype=torch.float32)
            for trg, cls in enumerate(self.classes):
                target[trg] = sample[cls]
        else:
            target = torch.zeros(config.n_classes, dtype=torch.float32)
            for cls in config.classes:
                target[config.class2target[cls]] = sample[cls]

        return image, target

    def __getitem__(self, index):
        set_random_seed(index)
        if self.length is not None:
            index = np.random.randint(len(self.data))
        image, target = self._get_sample(index)
        if self.transform is not None:
            image = self.transform(image)
        if target is not None:
            return image, target
        else:
            return image


class RandomDataset(Dataset):
    def __init__(self, datasets, length, probs=None):
        self.datasets = datasets
        self.probs = probs
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        set_random_seed(index)
        dataset_idx = np.random.choice(
            range(len(self.datasets)),
            p=self.probs
        )
        dataset = self.datasets[dataset_idx]
        index = random.randint(0, len(dataset) - 1)
        return dataset[index]
