import cv2
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.predictor import Predictor
from src.segm.predictor import SegmPredictor
from src.datasets import get_folds_data
from src.utils import get_best_model_path, remove_than_make_dir

from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--segm', default='', type=str)
parser.add_argument('--cls', default='', type=str)
args = parser.parse_args()

SEGM_EXPERIMENT = args.segm
CLS_EXPERIMENT = args.cls
SEGM_PREDICTION_DIR = config.segm_predictions_dir / 'val' / SEGM_EXPERIMENT
VAL_PREDICTION_DIR = config.predictions_dir / CLS_EXPERIMENT / 'val'
BATCH_SIZE = 8
DEVICE = 'cuda'


def segmentation_pred():
    segm_experiment_dir = config.segm_experiments_dir / SEGM_EXPERIMENT
    print(f"Segm experiment dir: {segm_experiment_dir}")
    print(f"Segm prediction dir: {SEGM_PREDICTION_DIR}")

    for fold in config.folds:
        print("Predict fold", fold)
        folds_data = get_folds_data()
        folds_data = [s for s in folds_data if s['fold'] == fold]
        fold_experiment_dir = segm_experiment_dir / f'fold_{fold}'
        model_path = get_best_model_path(fold_experiment_dir)
        print("Model path", model_path)
        predictor = SegmPredictor(model_path, BATCH_SIZE, DEVICE,
                                  num_workers=2)

        study_ids = [s['StudyInstanceUID'] for s in folds_data]

        for mask, study_id in zip(predictor.predict(folds_data), study_ids):
            mask = (mask * 255).astype(np.uint8)
            mask_path = str(SEGM_PREDICTION_DIR / (study_id + '.jpg'))
            cv2.imwrite(mask_path, mask)


def classification_pred():
    print(f"Start predict: {CLS_EXPERIMENT}")

    pred_dict = dict()
    for fold in config.folds:
        print("Predict fold", fold)
        model_path = get_best_model_path(
            config.experiments_dir / CLS_EXPERIMENT / f'fold_{fold}'
        )
        print("Model path", model_path)

        predictor = Predictor(model_path, BATCH_SIZE,
                              device=DEVICE, num_workers=8, tta=False)
        folds_data = get_folds_data(lung_masks_dir=SEGM_PREDICTION_DIR)
        folds_data = [s for s in folds_data if s['fold'] == fold]
        study_ids = [s['StudyInstanceUID'] for s in folds_data]

        fold_pred = predictor.predict(folds_data)
        for study_id, row_pred in zip(study_ids, fold_pred):
            pred_dict[study_id] = row_pred

    study_ids = [s['StudyInstanceUID'] for s in get_folds_data()]
    np.savez(
        VAL_PREDICTION_DIR / 'preds.npz',
        preds=np.stack([pred_dict[sid] for sid in study_ids]),
        study_ids=study_ids,
    )

    return pred_dict


def make_submission(pred_dict):
    folds_data = get_folds_data()
    study_ids = [s['StudyInstanceUID'] for s in folds_data]
    pred = np.stack([pred_dict[s] for s in study_ids])
    subm_df = pd.DataFrame(index=study_ids, columns=config.classes)
    subm_df.index.name = 'StudyInstanceUID'
    subm_df.values[:] = pred
    subm_df.to_csv(VAL_PREDICTION_DIR / 'submission.csv')

    train_df = pd.read_csv(config.train_folds_path, index_col=0)
    train_df = train_df.loc[subm_df.index].copy()
    scores = roc_auc_score(train_df[config.classes].values,
                           subm_df[config.classes].values, average=None)
    scores_dict = {cls: scr for cls, scr in zip(config.classes, scores)}
    scores_dict['Overal'] = np.mean(scores)

    with open(VAL_PREDICTION_DIR / 'scores.json', 'w') as outfile:
        json.dump(scores_dict, outfile)


if __name__ == "__main__":
    print("Segm experiment", SEGM_EXPERIMENT)
    print("Device", DEVICE)
    print("Batch size", BATCH_SIZE)

    if SEGM_EXPERIMENT:
        remove_than_make_dir(SEGM_PREDICTION_DIR)
        segmentation_pred()

    if CLS_EXPERIMENT:
        remove_than_make_dir(VAL_PREDICTION_DIR)
        pred_dict = classification_pred()
        make_submission(pred_dict)
