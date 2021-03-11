import argparse
import numpy as np
import pandas as pd

from src.predictor import Predictor
from src.datasets import get_test_data
from src.utils import (
    get_best_model_path,
    remove_than_make_dir,
    load_and_blend_preds
)
from src.stacking.datasets import get_stacking_test_data
from src.stacking.predictor import StackPredictor

from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--stack_experiment', default='', type=str)
parser.add_argument('--folds', default='', type=str)
parser.add_argument('--tta', action='store_true')
args = parser.parse_args()

BATCH_SIZE = 4
TTA = args.tta
DEVICE = 'cuda'
NUM_WORKERS = 2 if config.kernel_mode else 8
if args.folds:
    FOLDS = [int(fold) for fold in args.folds.split(',')]
else:
    FOLDS = config.folds


def classification_pred(test_data, cls_experiment):
    test_prediction_dir = config.predictions_dir / cls_experiment / 'test'
    remove_than_make_dir(test_prediction_dir)
    print(f"Start predict: {cls_experiment}")
    study_ids = [s['StudyInstanceUID'] for s in test_data]

    pred_lst = []
    for fold in FOLDS:
        print("Predict fold", fold)
        model_path = get_best_model_path(
            config.experiments_dir / cls_experiment / f'fold_{fold}'
        )
        print("Model path", model_path)

        predictor = Predictor(model_path, BATCH_SIZE,
                              device=DEVICE, tta=TTA,
                              num_workers=NUM_WORKERS)

        fold_pred = predictor.predict(test_data)
        if not config.kernel_mode:
            fold_pred_dir = test_prediction_dir / f'fold_{fold}'
            fold_pred_dir.mkdir(exist_ok=True, parents=True)
            np.savez(
                fold_pred_dir / 'preds.npz',
                preds=fold_pred,
                study_ids=study_ids,
            )

        pred_lst.append(fold_pred)

    np.savez(
        test_prediction_dir / 'preds.npz',
        preds=np.mean(pred_lst, axis=0),
        study_ids=study_ids,
    )


def stacking_pred(stack_test_data, stacking_experiment):
    test_prediction_dir = config.predictions_dir / stacking_experiment / 'test'
    remove_than_make_dir(test_prediction_dir)
    print(f"Start stack predict: {stacking_experiment}")
    study_ids = [s['StudyInstanceUID'] for s in stack_test_data]

    pred_lst = []
    for fold in FOLDS:
        print("Predict fold", fold)
        model_path = get_best_model_path(
            config.experiments_dir / stacking_experiment / f'fold_{fold}'
        )
        print("Stack model path", model_path)
        predictor = StackPredictor(model_path, BATCH_SIZE, device=DEVICE)
        assert predictor.model.params['experiments'] == args.experiment

        fold_pred = predictor.predict(stack_test_data)
        if not config.kernel_mode:
            fold_pred_dir = test_prediction_dir / f'fold_{fold}'
            fold_pred_dir.mkdir(exist_ok=True, parents=True)
            np.savez(
                fold_pred_dir / 'preds.npz',
                preds=fold_pred,
                study_ids=study_ids,
            )

        pred_lst.append(fold_pred)

    np.savez(
        test_prediction_dir / 'preds.npz',
        preds=np.mean(pred_lst, axis=0),
        study_ids=study_ids,
    )


def make_submission(experiments):
    pred_paths = [config.predictions_dir / exp / 'test' / 'preds.npz'
                  for exp in experiments]
    print("Make submission from predictions", pred_paths)
    blend_preds, study_ids = load_and_blend_preds(pred_paths)

    subm_df = pd.DataFrame(index=study_ids, columns=config.classes)
    subm_df.index.name = 'StudyInstanceUID'
    subm_df.values[:] = blend_preds
    if config.kernel_mode:
        subm_df.to_csv('submission.csv')
    else:
        test_prediction_dir = config.predictions_dir / ','.join(experiments) / 'test'
        if not test_prediction_dir.exists():
            test_prediction_dir.mkdir(parents=True, exist_ok=True)
        subm_df.to_csv(test_prediction_dir / "submission.csv")


if __name__ == "__main__":
    experiments = sorted(args.experiment.split(','))
    stack_experiments = sorted(args.stack_experiment.split(','))
    assert experiments
    print("Device", DEVICE)
    print("Batch size", BATCH_SIZE)
    print("TTA", TTA)
    print("Experiments", experiments)
    print("Stack experiments", stack_experiments)
    print("Folds", FOLDS)

    test_data = get_test_data()
    for experiment in experiments:
        classification_pred(test_data, experiment)
    if stack_experiments:
        stack_test_data = get_stacking_test_data(experiments)
        for stack_experiment in stack_experiments:
            stacking_pred(stack_test_data, stack_experiment)
    make_submission(experiments + stack_experiments)
