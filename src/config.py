import os
from pathlib import Path


kernel_mode = False
if 'KERNEL_MODE' in os.environ and os.environ['KERNEL_MODE'] == 'predict':
    kernel_mode = True

if kernel_mode:
    input_data_dir = Path('/kaggle/input/ranzcr-clip-catheter-line-classification/')
    output_data_dir = Path('/kaggle/working/data')
else:
    input_data_dir = Path('/workdir/data/')
    output_data_dir = Path('/workdir/data/')

train_csv_path = input_data_dir / 'train.csv'
train_annotations_csv_path = input_data_dir / 'train_annotations.csv'
train_dir = input_data_dir / 'train'
test_dir = input_data_dir / 'test'
sample_submission_path = input_data_dir / 'sample_submission.csv'

train_folds_path = output_data_dir / 'train_folds_v1.csv'
experiments_dir = output_data_dir / 'experiments'
predictions_dir = output_data_dir / 'predictions'

classes = [
    'ETT - Abnormal',
    'ETT - Borderline',
    'ETT - Normal',
    'NGT - Abnormal',
    'NGT - Borderline',
    'NGT - Incompletely Imaged',
    'NGT - Normal',
    'CVC - Abnormal',
    'CVC - Borderline',
    'CVC - Normal',
    'Swan Ganz Catheter Present'
]

target2class = {trg: cls for trg, cls in enumerate(classes)}
class2target = {cls: trg for trg, cls in enumerate(classes)}
n_classes = len(classes)
n_folds = 5
folds = list(range(n_folds))
