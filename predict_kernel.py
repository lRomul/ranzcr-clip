import numpy as np
import pandas as pd

from src.predictor import Predictor
from src.datasets import get_test_data
from src.transforms import get_transforms
from src.utils import get_best_model_path

from src import config


EXPERIMENT = "train_001"
IMAGE_SIZE = 512
BATCH_SIZE = 8
TTA = False
DEVICE = 'cuda'
FOLDS = config.folds


def experiment_pred(experiment_dir, test_data):
    print(f"Start predict: {experiment_dir}")
    image_transforms = get_transforms(False, IMAGE_SIZE)

    pred_lst = []
    for fold in FOLDS:
        print("Predict fold", fold)
        fold_dir = experiment_dir / f'fold_{fold}'
        model_path = get_best_model_path(fold_dir)
        print("Model path", model_path)
        predictor = Predictor(model_path, BATCH_SIZE,
                              image_transforms,
                              DEVICE, TTA,
                              num_workers=2)

        pred = predictor.predict(test_data)
        pred_lst.append(pred)

    pred = np.mean(pred_lst, axis=0)
    return pred


def make_submission(pred, test_data):
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
    print("Experiment", EXPERIMENT)
    print("Device", DEVICE)
    print("Image size", IMAGE_SIZE)
    print("Batch size", BATCH_SIZE)

    test_data = get_test_data()
    experiment_dir = config.experiments_dir / EXPERIMENT
    exp_pred = experiment_pred(experiment_dir, test_data)
    make_submission(exp_pred, test_data)
