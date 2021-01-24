import cv2
import shutil
import argparse
import numpy as np
import pandas as pd

from src.predictor import Predictor
from src.segm.predictor import SegmPredictor
from src.datasets import get_test_data
from src.utils import get_best_model_path

from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--segm', required=True, type=str)
parser.add_argument('--ett', required=True, type=str)
parser.add_argument('--ngt', required=True, type=str)
parser.add_argument('--cvc', required=True, type=str)
parser.add_argument('--folds', default='', type=str)
args = parser.parse_args()

SEGM_EXPERIMENT = args.segm
ETT_EXPERIMENT = args.ett
NGT_EXPERIMENT = args.ngt
CVC_EXPERIMENT = args.cvc
EXPERIMENT = "-".join([SEGM_EXPERIMENT, ETT_EXPERIMENT, NGT_EXPERIMENT, CVC_EXPERIMENT])
SEGM_PREDICTION_DIR = config.segm_predictions_dir / 'test' / SEGM_EXPERIMENT
BATCH_SIZE = 4
DEVICE = 'cuda'
if args.folds:
    FOLDS = [int(fold) for fold in args.folds.split(',')]
else:
    FOLDS = config.folds


def segmentation_pred():
    test_data = get_test_data()
    segm_experiment_dir = config.segm_experiments_dir / SEGM_EXPERIMENT
    if SEGM_PREDICTION_DIR.exists():
        shutil.rmtree(SEGM_PREDICTION_DIR)
    print(f"Segm experiment dir: {segm_experiment_dir}")
    print(f"Segm prediction dir: {SEGM_PREDICTION_DIR}")

    for fold in FOLDS:
        print("Predict fold", fold)
        fold_experiment_dir = segm_experiment_dir / f'fold_{fold}'
        model_path = get_best_model_path(fold_experiment_dir)
        print("Model paths:", model_path)
        predictor = SegmPredictor(model_path, BATCH_SIZE, DEVICE,
                                  num_workers=2)

        study_ids = [s['StudyInstanceUID'] for s in test_data]

        fold_segm_pred_dir = SEGM_PREDICTION_DIR / f'fold_{fold}'
        fold_segm_pred_dir.mkdir(parents=True)
        for mask, study_id in zip(predictor.predict(test_data), study_ids):
            mask = (mask * 255).astype(np.uint8)
            mask_path = str(fold_segm_pred_dir / (study_id + '.jpg'))
            cv2.imwrite(mask_path, mask)


def classification_pred():
    print(f"Start predict: {EXPERIMENT}")

    pred_lst = []
    for fold in FOLDS:
        print("Predict fold", fold)
        model_paths = [get_best_model_path(config.experiments_dir / e / f'fold_{fold}')
                       for e in [ETT_EXPERIMENT, NGT_EXPERIMENT, CVC_EXPERIMENT]]
        print("Model path", model_paths)

        predictor = Predictor(model_paths, BATCH_SIZE,
                              device=DEVICE, num_workers=2)
        test_data = get_test_data(lung_masks_dir=SEGM_PREDICTION_DIR / f'fold_{fold}')

        fold_pred = predictor.predict(test_data)
        pred_lst.append(fold_pred)

    pred = np.mean(pred_lst, axis=0)
    return pred


def make_submission(pred):
    test_data = get_test_data()
    test_prediction_dir = config.predictions_dir / EXPERIMENT / 'test'
    test_prediction_dir.mkdir(parents=True, exist_ok=True)
    study_ids = [s['StudyInstanceUID'] for s in test_data]
    subm_df = pd.DataFrame(index=study_ids, columns=config.classes)
    subm_df.index.name = 'StudyInstanceUID'
    subm_df.values[:] = pred
    if config.kernel_mode:
        subm_df.to_csv('submission.csv')
    else:
        subm_df.to_csv(test_prediction_dir / 'submission.csv')


if __name__ == "__main__":
    print("Device", DEVICE)
    print("Batch size", BATCH_SIZE)

    # segmentation_pred()
    pred = classification_pred()
    make_submission(pred)
