import argparse
import numpy as np
import pandas as pd

from src.predictor import Predictor
from src.datasets import get_chest_xrays_data
from src.utils import get_best_model_path, remove_than_make_dir

from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
args = parser.parse_args()

EXPERIMENT = args.experiment
BATCH_SIZE = 4
DEVICE = 'cuda'
NUM_WORKERS = 8
CHEST_XRAYS_PREDICTION_DIR = config.predictions_dir / EXPERIMENT / 'chest_xrays'


def classification_pred():
    print(f"Start predict: {EXPERIMENT}")

    pred_lst = []
    for fold in config.folds:
        print("Predict fold", fold)
        model_path = get_best_model_path(
            config.experiments_dir / EXPERIMENT / f'fold_{fold}'
        )
        print("Model path", model_path)

        predictor = Predictor(model_path, BATCH_SIZE,
                              device=DEVICE, tta=False,
                              num_workers=NUM_WORKERS)
        test_data = get_chest_xrays_data()

        fold_pred = predictor.predict(test_data)
        if not config.kernel_mode:
            study_ids = [s['StudyInstanceUID'] for s in test_data]
            fold_pred_dir = CHEST_XRAYS_PREDICTION_DIR / f'fold_{fold}'
            fold_pred_dir.mkdir(exist_ok=True, parents=True)
            np.savez(
                fold_pred_dir / 'preds.npz',
                preds=fold_pred,
                study_ids=study_ids,
            )

        pred_lst.append(fold_pred)

    pred = np.mean(pred_lst, axis=0)
    return pred


def make_submission(pred):
    test_data = get_chest_xrays_data()
    study_ids = [s['StudyInstanceUID'] for s in test_data]
    subm_df = pd.DataFrame(index=study_ids, columns=config.classes)
    subm_df.index.name = 'StudyInstanceUID'
    subm_df.values[:] = pred
    if config.kernel_mode:
        subm_df.to_csv('submission.csv')
    else:
        subm_df.to_csv(CHEST_XRAYS_PREDICTION_DIR / 'submission.csv')


if __name__ == "__main__":
    print("Device", DEVICE)
    print("Batch size", BATCH_SIZE)

    remove_than_make_dir(CHEST_XRAYS_PREDICTION_DIR)
    pred = classification_pred()
    make_submission(pred)
