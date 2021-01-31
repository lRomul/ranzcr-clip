import json
import argparse

import torch
from torch.utils.data import DataLoader

from argus.callbacks import (
    MonitorCheckpoint,
    LoggingToFile,
    LoggingToCSV,
    CosineAnnealingLR,
    LambdaLR
)

from src.datasets import RanzcrDataset, get_folds_data
from src.transforms import get_transforms
from src.argus_model import RanzcrModel
from src.ema import EmaMonitorCheckpoint, ModelEma
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--folds', default='', type=str)
args = parser.parse_args()

SEGM_EXPERIMENT = 'segm_003'
PSEUDO_EXPERIMENT = ''
BATCH_SIZE = 16
IMAGE_SIZE = 768
NUM_WORKERS = 8
NUM_EPOCHS = [2, 16]  # , 3]
STAGE = ['warmup', 'train']  # , 'cooldown']
BASE_LR = 1e-3
MIN_BASE_LR = 1e-5
USE_AMP = True
USE_EMA = True
DRAW_ANNOTATIONS = False
EMA_DECAY = 0.9997
SAVE_DIR = config.experiments_dir / args.experiment

if PSEUDO_EXPERIMENT:
    PSEUDO = config.predictions_dir / PSEUDO_EXPERIMENT / 'val' / 'preds.npz'
else:
    PSEUDO = None
N_CHANNELS = 3 if DRAW_ANNOTATIONS else 1


def get_lr(base_lr, batch_size):
    return base_lr * (batch_size / 16)


PARAMS = {
    'nn_module': ('timm', {
        'model_name': 'tf_efficientnet_b3_ns',
        'pretrained': True,
        'num_classes': config.n_classes,
        'in_chans': N_CHANNELS,
        'drop_rate': 0.3,
        'drop_path_rate': 0.2
    }),
    'loss': 'BCEWithLogitsLoss',
    'optimizer': ('AdamW', {'lr': get_lr(BASE_LR, BATCH_SIZE)}),
    'device': [f'cuda:{i}' for i in range(torch.cuda.device_count())],
    'amp': USE_AMP,
    'clip_grad': False,
    'image_size': IMAGE_SIZE,
    'draw_annotations': DRAW_ANNOTATIONS
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
        print(f"Pseudo label: {pseudo}, draw annotations: {DRAW_ANNOTATIONS}")
        train_dataset = RanzcrDataset(folds_data,
                                      folds=train_folds,
                                      transform=train_transfrom,
                                      annotations=DRAW_ANNOTATIONS,
                                      pseudo_label=pseudo)
        val_dataset = RanzcrDataset(folds_data,
                                    folds=val_folds,
                                    transform=val_transform,
                                    annotations=DRAW_ANNOTATIONS)
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
                checkpoint(save_dir, monitor=f'val_roc_auc',
                           max_saves=1, better='max')
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

    segm_predictions_dir = config.segm_predictions_dir / 'val' / SEGM_EXPERIMENT
    folds_data = get_folds_data(lung_masks_dir=segm_predictions_dir,
                                pseudo_label_path=PSEUDO)

    if args.folds:
        folds = [int(fold) for fold in args.folds.split(',')]
    else:
        folds = config.folds

    for fold in folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds, folds_data)
