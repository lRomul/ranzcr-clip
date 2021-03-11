import json
import argparse

from argus.callbacks import (
    EarlyStopping,
    LoggingToFile,
    MonitorCheckpoint,
    CosineAnnealingLR
)

from torch.utils.data import DataLoader, ConcatDataset

from src.ema import EmaMonitorCheckpoint, ModelEma
from src.stacking.datasets import (
    StackingDataset,
    get_stacking_folds_data,
    get_stacking_test_data
)
from src.stacking.argus_models import StackingModel
from src import config

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--folds', default='all', type=str)
args = parser.parse_args()

EXPERIMENTS = 'kdb3v3_b71_001,kdb4v3_b61_002,kdb4v3_b71_001'
EXPERIMENTS = sorted(EXPERIMENTS.split(','))
PSEUDO_EXPERIMENT = 'b7v3_001'
USE_AMP = False
RS_PARAMS = {
    "base_size": 512,
    "reduction_scale": 4,
    "p_dropout": 0.2132065192973704,
    "lr": 7.929567216714842e-05,
    "epochs": 70,
    "eta_min_scale": 0.09535456983407244,
    "batch_size": 32,
    "use_ema": False,
    "ema_decay": 0.9996726803610275
}
BATCH_SIZE = RS_PARAMS['batch_size']
NUM_WORKERS = 2

SAVE_DIR = config.experiments_dir / args.experiment

PSEUDO_THRESHOLD = None
if PSEUDO_EXPERIMENT:
    PSEUDO = config.predictions_dir / PSEUDO_EXPERIMENT / 'val' / 'preds.npz'
    TEST_PSEUDO = config.predictions_dir / PSEUDO_EXPERIMENT / 'test'
else:
    PSEUDO = None
    TEST_PSEUDO = None

PARAMS = {
    'nn_module': ('FCNet', {
        'in_channels': len(EXPERIMENTS) * config.n_classes,
        'num_classes': config.n_classes,
        'base_size': RS_PARAMS['base_size'],
        'reduction_scale': RS_PARAMS['reduction_scale'],
        'p_dropout': RS_PARAMS['p_dropout']
    }),
    'loss': 'BCEWithLogitsLoss',
    'optimizer': ('AdamW', {'lr': RS_PARAMS['lr']}),
    'device': 'cuda',
    'experiments': ','.join(EXPERIMENTS),
    'amp': USE_AMP,
    'clip_grad': 0.0,
}


def train_fold(save_dir, train_folds, val_folds, folds_data):
    train_datasets = []
    if PSEUDO:
        test_data = get_stacking_test_data(
            EXPERIMENTS,
            pseudo_label_path=TEST_PSEUDO / f'fold_{val_folds[0]}' / 'preds.npz'
        )
        test_dataset = StackingDataset(test_data,
                                       pseudo_label=PSEUDO,
                                       pseudo_threshold=PSEUDO_THRESHOLD)
        train_datasets.append(test_dataset)
    train_folds_dataset = StackingDataset(folds_data,
                                          folds=train_folds,
                                          pseudo_label=PSEUDO,
                                          pseudo_threshold=PSEUDO_THRESHOLD)
    train_datasets.append(train_folds_dataset)
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = StackingDataset(folds_data, val_folds)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = StackingModel(PARAMS)

    if RS_PARAMS['use_ema']:
        print(f"EMA decay: {RS_PARAMS['ema_decay']}")
        model.model_ema = ModelEma(model.nn_module, decay=RS_PARAMS['ema_decay'])
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
    print("Batch size", BATCH_SIZE)
    print("Experiments", EXPERIMENTS)

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

    folds_data = get_stacking_folds_data(EXPERIMENTS, pseudo_label_path=PSEUDO)

    for fold in folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds, folds_data)
