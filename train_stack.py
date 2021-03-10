import json
import argparse

from argus.callbacks import (
    EarlyStopping,
    LoggingToFile,
    MonitorCheckpoint,
    CosineAnnealingLR
)

from torch.utils.data import DataLoader

from src.ema import EmaMonitorCheckpoint, ModelEma
from src.stacking.datasets import StackingDataset, get_stacking_folds_data
from src.stacking.argus_models import StackingModel
from src import config

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--folds', default='all', type=str)
args = parser.parse_args()

STACKING_EXPERIMENT = args.experiment
EXPERIMENTS = 'kdb3v3_b71_001,kdb4v3_b61_002,kdb4v3_b71_001'
USE_EMA = True
USE_AMP = False
EMA_DECAY = 0.9997
RS_PARAMS = {
    "base_size": 256,
    "reduction_scale": 2,
    "p_dropout": 0.025,
    "lr": 4e-04,
    "epochs": 12,
    "eta_min_scale": 0.01,
    "batch_size": 32
}
BATCH_SIZE = RS_PARAMS['batch_size']
NUM_WORKERS = 2

SAVE_DIR = config.experiments_dir / STACKING_EXPERIMENT
PARAMS = {
    'nn_module': ('FCNet', {
        'in_channels': len(EXPERIMENTS.split(',')) * config.n_classes,
        'num_classes': config.n_classes,
        'base_size': RS_PARAMS['base_size'],
        'reduction_scale': RS_PARAMS['reduction_scale'],
        'p_dropout': RS_PARAMS['p_dropout']
    }),
    'loss': 'BCEWithLogitsLoss',
    'optimizer': ('AdamW', {'lr': RS_PARAMS['lr']}),
    'device': 'cuda',
    'experiments': EXPERIMENTS,
    'amp': USE_AMP,
    'clip_grad': 0.0,
}


def train_fold(save_dir, train_folds, val_folds, folds_data):
    train_dataset = StackingDataset(folds_data, train_folds)
    val_dataset = StackingDataset(folds_data, val_folds)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = StackingModel(PARAMS)

    if USE_EMA:
        print(f"EMA decay: {EMA_DECAY}")
        model.model_ema = ModelEma(model.nn_module, decay=EMA_DECAY)
        checkpoint = EmaMonitorCheckpoint
    else:
        checkpoint = MonitorCheckpoint

    callbacks = [
        checkpoint(save_dir, monitor='val_roc_auc', max_saves=1),
        CosineAnnealingLR(T_max=RS_PARAMS['epochs'],
                          eta_min=RS_PARAMS['lr'] * RS_PARAMS['eta_min_scale']),
        EarlyStopping(monitor='val_roc_auc', patience=30),
        LoggingToFile(save_dir / 'log.txt'),
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              num_epochs=RS_PARAMS['epochs'],
              callbacks=callbacks,
              metrics=['roc_auc'])


if __name__ == "__main__":
    experiments = sorted(EXPERIMENTS.split(','))
    assert experiments
    print("Batch size", BATCH_SIZE)
    print("Experiments", experiments)

    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder {SAVE_DIR} already exists.")

    with open(SAVE_DIR / 'source.py', 'w') as outfile:
        outfile.write(open(__file__).read())

    print("Model params", PARAMS)
    with open(SAVE_DIR / 'params.json', 'w') as outfile:
        json.dump(PARAMS, outfile)

    if args.folds == 'all':
        folds = config.folds
    else:
        folds = [int(fold) for fold in args.folds.split(',')]

    folds_data = get_stacking_folds_data(experiments)

    for fold in folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds, folds_data)
