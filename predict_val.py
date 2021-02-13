import argparse
import numpy as np

from src.predictor import Predictor
from src.datasets import get_folds_data
from src.utils import (
    get_best_model_path,
    remove_than_make_dir,
    save_and_score_val_subm
)

from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
args = parser.parse_args()

EXPERIMENT = args.experiment
VAL_PREDICTION_DIR = config.predictions_dir / EXPERIMENT / 'val'
BATCH_SIZE = 4
DEVICE = 'cuda'


def classification_pred():
    print(f"Start predict: {EXPERIMENT}")

    pred_dict = dict()
    for fold in config.folds:
        print("Predict fold", fold)
        model_path = get_best_model_path(
            config.experiments_dir / EXPERIMENT / f'fold_{fold}'
        )
        print("Model path", model_path)

        predictor = Predictor(model_path, BATCH_SIZE,
                              device=DEVICE, num_workers=8, tta=False)
        folds_data = get_folds_data()
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

    save_and_score_val_subm(pred, study_ids, VAL_PREDICTION_DIR)


if __name__ == "__main__":
    print("Device", DEVICE)
    print("Batch size", BATCH_SIZE)

    remove_than_make_dir(VAL_PREDICTION_DIR)
    pred_dict = classification_pred()
    make_submission(pred_dict)
