import json
import argparse

from torch.utils.data import DataLoader

from argus.callbacks import (
    MonitorCheckpoint,
    EarlyStopping,
    LoggingToFile,
    LoggingToCSV,
    CosineAnnealingLR
)

from src.datasets import RanzcrDataset, get_folds_data
from src.transforms import get_transforms
from src.argus_model import RanzcrModel
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--folds', default='', type=str)
args = parser.parse_args()

BATCH_SIZE = 16
NUM_EPOCHS = 24
IMAGE_SIZE = 512
NUM_WORKERS = 8
SAVE_DIR = config.experiments_dir / args.experiment
PARAMS = {
    'nn_module': ('timm', {
        'model_name': 'tf_efficientnet_b0_ns',
        'pretrained': True,
        'num_classes': config.n_classes,
        'in_chans': 1,
        'drop_rate': 0.2,
        'drop_path_rate': 0.2
    }),
    'loss': 'BCEWithLogitsLoss',
    'optimizer': ('Adam', {'lr': 0.001}),
    'device': 'cuda',
    'amp': True
}


def train_fold(save_dir, train_folds, val_folds, folds_data):
    train_transfrom = get_transforms(train=True, size=IMAGE_SIZE)
    val_transform = get_transforms(train=False, size=IMAGE_SIZE)

    train_dataset = RanzcrDataset(folds_data, folds=train_folds,
                                  image_transform=train_transfrom)
    val_dataset = RanzcrDataset(folds_data, folds=val_folds,
                                image_transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = RanzcrModel(PARAMS)
    if 'pretrained' in model.params['nn_module'][1]:
        model.params['nn_module'][1]['pretrained'] = False

    num_iterations = (len(train_dataset) // BATCH_SIZE) * NUM_EPOCHS
    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_roc_auc', max_saves=1),
        CosineAnnealingLR(T_max=num_iterations, eta_min=0, step_on_iteration=True),
        EarlyStopping(monitor='val_roc_auc', patience=12),
        LoggingToFile(save_dir / 'log.txt'),
        LoggingToCSV(save_dir / 'log.csv')
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              num_epochs=NUM_EPOCHS,
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

    folds_data = get_folds_data()

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
