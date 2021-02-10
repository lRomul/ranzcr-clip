import json
import argparse

import torch
from torch.utils.data import DataLoader, ConcatDataset

from argus.callbacks import (
    MonitorCheckpoint,
    LoggingToFile,
    LoggingToCSV,
    CosineAnnealingLR,
    LambdaLR,
    EarlyStopping
)

from src.datasets import (
    RanzcrDataset,
    RandomDataset,
    get_folds_data,
    get_test_data,
    get_chest_xrays_data
)
from src.transforms import get_transforms
from src.argus_model import RanzcrModel
from src.ema import EmaMonitorCheckpoint, ModelEma
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--folds', default='all', type=str)
args = parser.parse_args()

PSEUDO_EXPERIMENT = 'b4_002'
PSEUDO_THRESHOLD = None
PSEUDO_XRAYS_PROB = 0.2
BATCH_SIZE = 8
IMAGE_SIZE = 1024
NUM_WORKERS = 12
NUM_EPOCHS = [2, 16]  # , 3]
STAGE = ['warmup', 'train']  # , 'cooldown']
BASE_LR = 5e-4
MIN_BASE_LR = 5e-6
USE_AMP = True
USE_EMA = True
EMA_DECAY = 0.9997
SAVE_DIR = config.experiments_dir / args.experiment

if PSEUDO_EXPERIMENT:
    PSEUDO = config.predictions_dir / PSEUDO_EXPERIMENT / 'val' / 'preds.npz'
    TEST_PSEUDO = config.predictions_dir / PSEUDO_EXPERIMENT / 'test'
    XRAYS_PSEUDO = config.predictions_dir / PSEUDO_EXPERIMENT / 'chest_xrays'
else:
    PSEUDO = None
    TEST_PSEUDO = None
    XRAYS_PSEUDO = None
N_CHANNELS = 1


def get_lr(base_lr, batch_size):
    return base_lr * (batch_size / 16)


PARAMS = {
    'nn_module': ('TimmModel', {
        'model_name': 'tf_efficientnet_b3_ns',
        'pretrained': True,
        'num_classes': config.n_classes,
        'in_chans': N_CHANNELS,
        'drop_rate': 0.3,
        'drop_path_rate': 0.2,
        'attention': None
    }),
    'loss': 'BCEWithLogitsLoss',
    'optimizer': ('AdamW', {'lr': get_lr(BASE_LR, BATCH_SIZE)}),
    'device': [f'cuda:{i}' for i in range(torch.cuda.device_count())],
    'amp': USE_AMP,
    'clip_grad': False,
    'image_size': IMAGE_SIZE,
    'draw_annotations': False
}


def train_fold(save_dir, train_folds, val_folds, folds_data):
    model = RanzcrModel(PARAMS)
    if 'pretrained' in model.params['nn_module'][1]:
        model.params['nn_module'][1]['pretrained'] = False

    if USE_EMA:
        print(f"EMA decay: {EMA_DECAY}")
        model.model_ema = ModelEma(model.nn_module, decay=EMA_DECAY)
        checkpoint = EmaMonitorCheckpoint
    else:
        checkpoint = MonitorCheckpoint

    for num_epochs, stage in zip(NUM_EPOCHS, STAGE):
        train_transfrom = get_transforms(train=True, size=IMAGE_SIZE,
                                         n_channels=N_CHANNELS)
        val_transform = get_transforms(train=False, size=IMAGE_SIZE,
                                       n_channels=N_CHANNELS)

        pseudo = stage != 'cooldown' and PSEUDO is not None
        print(f"Pseudo label: {pseudo}")

        train_datasets = []
        if pseudo:
            test_data = get_test_data(
                pseudo_label_path=TEST_PSEUDO / f'fold_{val_folds[0]}' / 'preds.npz'
            )
            test_dataset = RanzcrDataset(test_data,
                                         transform=train_transfrom,
                                         pseudo_label=pseudo,
                                         pseudo_threshold=PSEUDO_THRESHOLD)
            train_datasets.append(test_dataset)
        train_folds_dataset = RanzcrDataset(folds_data,
                                      folds=train_folds,
                                      transform=train_transfrom,
                                      pseudo_label=pseudo,
                                      pseudo_threshold=PSEUDO_THRESHOLD)
        train_datasets.append(train_folds_dataset)
        train_dataset = ConcatDataset(train_datasets)

        if PSEUDO_XRAYS_PROB and pseudo:
            xrays_data = get_chest_xrays_data(
                pseudo_label_path=XRAYS_PSEUDO / f'fold_{val_folds[0]}' / 'preds.npz'
            )
            xrays_dataset = RanzcrDataset(xrays_data,
                                          transform=train_transfrom,
                                          pseudo_label=pseudo,
                                          pseudo_threshold=PSEUDO_THRESHOLD)
            train_dataset = RandomDataset(
                [train_dataset, xrays_dataset],
                len(train_dataset),
                probs=[1.0 - PSEUDO_XRAYS_PROB, PSEUDO_XRAYS_PROB]
            )

        val_dataset = RanzcrDataset(folds_data,
                                    folds=val_folds,
                                    transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=True,
                                  num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                                shuffle=False, num_workers=NUM_WORKERS)

        callbacks = [
            LoggingToFile(save_dir / 'log.txt', append=True),
            LoggingToCSV(save_dir / 'log.csv', append=True)
        ]

        num_iterations = (len(train_dataset) // BATCH_SIZE) * num_epochs
        if stage == 'warmup':
            callbacks += [
                LambdaLR(lambda x: x / num_iterations,
                         step_on_iteration=True)
            ]
        elif stage == 'train':
            callbacks += [
                CosineAnnealingLR(T_max=num_iterations,
                                  eta_min=get_lr(MIN_BASE_LR, BATCH_SIZE),
                                  step_on_iteration=True),
                checkpoint(save_dir, monitor='val_roc_auc',
                           max_saves=1, better='max'),
                EarlyStopping(monitor='val_roc_auc', patience=2)
            ]
        elif stage == 'cooldown':
            callbacks += [
                checkpoint(save_dir, monitor=f'val_roc_auc',
                           max_saves=1, better='max')
            ]

        model.fit(train_loader,
                  val_loader=val_loader,
                  num_epochs=num_epochs,
                  callbacks=callbacks,
                  metrics=['roc_auc'])


if __name__ == "__main__":
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder {SAVE_DIR} already exists.")

    with open(SAVE_DIR / 'source.py', 'w') as outfile:
        outfile.write(open(__file__).read())

    print("Model params", PARAMS)
    with open(SAVE_DIR / 'params.json', 'w') as outfile:
        json.dump(PARAMS, outfile)

    folds_data = get_folds_data(pseudo_label_path=PSEUDO)

    if args.folds == 'all':
        folds = config.folds
    else:
        folds = [int(fold) for fold in args.folds.split(',')]

    for fold in folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds, folds_data)
