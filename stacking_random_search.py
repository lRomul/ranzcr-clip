import json
import time
import random
import argparse
import numpy as np
from pprint import pprint

from argus.callbacks import (
    EarlyStopping,
    LoggingToFile,
    MonitorCheckpoint,
    CosineAnnealingLR
)

import torch
from torch.utils.data import DataLoader

from src.ema import EmaMonitorCheckpoint, ModelEma
from src.stacking.datasets import StackingDataset, get_stacking_folds_data
from src.stacking.argus_models import StackingModel
from src import config

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--folds', default='all', type=str)
args = parser.parse_args()

EXPERIMENTS = 'kdb3v3_b71_001,kdb4v3_b61_002,kdb4v3_b71_001'
NUM_WORKERS = 2
SAVE_DIR = config.experiments_dir / args.experiment


def train_folds(save_dir, folds_data):
    random_params = {
        'base_size': int(np.random.choice([64, 128, 256, 512])),
        'reduction_scale': int(np.random.choice([2, 4, 8, 16])),
        'p_dropout': float(np.random.uniform(0.0, 0.5)),
        'lr': float(np.random.uniform(0.0001, 0.00001)),
        'epochs': int(np.random.randint(10, 80)),
        'eta_min_scale': float(np.random.uniform(0.1, 0.01)),
        'batch_size': int(np.random.choice([32, 64, 128])),
        'use_ema': bool(np.random.choice([False, True])),
        'ema_decay': float(np.random.uniform(0.9995, 0.9999))
    }
    pprint(random_params)

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'random_params.json', 'w') as outfile:
        json.dump(random_params, outfile)

    params = {
        'nn_module': ('FCNet', {
            'in_channels': len(EXPERIMENTS.split(',')) * config.n_classes,
            'num_classes': config.n_classes,
            'base_size': random_params['base_size'],
            'reduction_scale': random_params['reduction_scale'],
            'p_dropout': random_params['p_dropout']
        }),
        'loss': 'BCEWithLogitsLoss',
        'optimizer': ('AdamW', {'lr': random_params['lr']}),
        'device': 'cuda',
        'experiments': EXPERIMENTS,
        'amp': False,
        'clip_grad': 0.0,
    }

    for fold in config.folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = save_dir / f'fold_{fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")

        train_dataset = StackingDataset(folds_data, train_folds)
        val_dataset = StackingDataset(folds_data, val_folds)

        train_loader = DataLoader(train_dataset, batch_size=random_params['batch_size'],
                                  shuffle=True, drop_last=True,
                                  num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=random_params['batch_size'] * 2,
                                shuffle=False, num_workers=NUM_WORKERS)

        model = StackingModel(params)

        if random_params['use_ema']:
            print(f"EMA decay: {random_params['use_ema']}")
            model.model_ema = ModelEma(model.nn_module, decay=random_params['ema_decay'])
            checkpoint = EmaMonitorCheckpoint
        else:
            checkpoint = MonitorCheckpoint

        callbacks = [
            checkpoint(save_fold_dir, monitor='val_roc_auc', max_saves=1),
            CosineAnnealingLR(T_max=random_params['epochs'],
                              eta_min=random_params['lr'] * random_params['eta_min_scale']),
            EarlyStopping(monitor='val_roc_auc', patience=30),
            LoggingToFile(save_fold_dir / 'log.txt'),
        ]

        model.fit(train_loader,
                  val_loader=val_loader,
                  num_epochs=random_params['epochs'],
                  callbacks=callbacks,
                  metrics=['roc_auc'])


if __name__ == "__main__":
    experiments = sorted(EXPERIMENTS.split(','))

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVE_DIR / 'source.py', 'w') as outfile:
        outfile.write(open(__file__).read())

    folds_data = get_stacking_folds_data(experiments)

    while True:
        num = random.randint(0, 2 ** 32 - 1)
        np.random.seed(num)
        random.seed(num)

        save_dir = SAVE_DIR / f'{num:011}'
        train_folds(save_dir, folds_data)

        time.sleep(5.0)
        torch.cuda.empty_cache()
        time.sleep(5.0)
