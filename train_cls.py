import os
import json
import argparse

import torch
import torch.distributed as dist
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from argus.callbacks import (
    MonitorCheckpoint,
    LoggingToFile,
    LoggingToCSV,
    CosineAnnealingLR,
    LambdaLR,
    EarlyStopping,
    on_epoch_complete
)

from src.datasets import (
    RanzcrDataset,
    get_folds_data,
    get_test_data
)
from src.transforms import get_transforms
from src.argus_model import RanzcrModel
from src.ema import EmaMonitorCheckpoint, ModelEma
from src import config

torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--folds', default='all', type=str)
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

PSEUDO_EXPERIMENT = ''
PSEUDO_THRESHOLD = None
BATCH_SIZE = 3
ITER_SIZE = 2
IMAGE_SIZE = 1024
NUM_WORKERS = 6
NUM_EPOCHS = [2, 16]  # , 3]
STAGE = ['warmup', 'train']  # , 'cooldown']
BASE_LR = 5e-4
MIN_BASE_LR = 5e-6
USE_AMP = True
USE_EMA = True
EMA_DECAY = 0.9997
SAVE_DIR = config.experiments_dir / args.experiment

if args.distributed:
    WORLD_BATCH_SIZE = BATCH_SIZE * dist.get_world_size()
else:
    WORLD_BATCH_SIZE = BATCH_SIZE
print("World batch size:", WORLD_BATCH_SIZE)

if PSEUDO_EXPERIMENT:
    PSEUDO = config.predictions_dir / PSEUDO_EXPERIMENT / 'val' / 'preds.npz'
    TEST_PSEUDO = config.predictions_dir / PSEUDO_EXPERIMENT / 'test'
else:
    PSEUDO = None
    TEST_PSEUDO = None
N_CHANNELS = 1


def get_lr(base_lr, batch_size):
    return base_lr * (batch_size / 16)


PARAMS = {
    'nn_module': ('TimmModel', {
        'model_name': 'tf_efficientnet_b7_ns',
        'pretrained': True,
        'num_classes': config.n_classes,
        'in_chans': N_CHANNELS,
        'drop_rate': 0.5,
        'drop_path_rate': 0.2,
        'attention': None
    }),
    'loss': 'BCEWithLogitsLoss',
    'optimizer': ('AdamW', {
        'lr': get_lr(BASE_LR, WORLD_BATCH_SIZE)
    }),
    'device': 'cuda',
    'amp': USE_AMP,
    'iter_size': ITER_SIZE,
    'clip_grad': False,
    'image_size': IMAGE_SIZE,
    'draw_annotations': False
}


def train_fold(save_dir, train_folds, val_folds, folds_data,
               local_rank=0, distributed=False):
    model = RanzcrModel(PARAMS)
    if 'pretrained' in model.params['nn_module'][1]:
        model.params['nn_module'][1]['pretrained'] = False

    if distributed:
        model.nn_module = SyncBatchNorm.convert_sync_batchnorm(model.nn_module)
        model.nn_module = DistributedDataParallel(model.nn_module.to(local_rank),
                                                  device_ids=[local_rank],
                                                  output_device=local_rank)
        if local_rank:
            model.logger.disabled = True
    else:
        model.set_device('cuda')

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
            train_datasets += [test_dataset]
        train_dataset = RanzcrDataset(folds_data,
                                      folds=train_folds,
                                      transform=train_transfrom,
                                      pseudo_label=pseudo,
                                      pseudo_threshold=PSEUDO_THRESHOLD)
        train_datasets += [train_dataset]

        train_dataset = ConcatDataset(train_datasets)

        train_sampler = None
        if distributed:
            train_sampler = DistributedSampler(train_dataset,
                                               num_replicas=dist.get_world_size(),
                                               rank=local_rank,
                                               shuffle=True)

        val_dataset = RanzcrDataset(folds_data,
                                    folds=val_folds,
                                    transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=train_sampler is None,
                                  drop_last=True,
                                  num_workers=NUM_WORKERS,
                                  sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                                shuffle=False, num_workers=NUM_WORKERS)

        callbacks = []
        if local_rank == 0:
            callbacks += [
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
                EarlyStopping(monitor='val_roc_auc', patience=1)
            ]
            if local_rank == 0:
                callbacks += [
                    checkpoint(save_dir, monitor='val_roc_auc',
                               max_saves=1, better='max')
                ]
        elif stage == 'cooldown':
            if local_rank == 0:
                callbacks += [
                    checkpoint(save_dir, monitor=f'val_roc_auc',
                               max_saves=1, better='max')
                ]

        if distributed:
            @on_epoch_complete
            def schedule_sampler(state):
                train_sampler.set_epoch(state.epoch + 1)
            callbacks += [schedule_sampler]

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
        train_fold(save_fold_dir, train_folds, val_folds, folds_data,
                   local_rank=args.local_rank, distributed=args.distributed)
