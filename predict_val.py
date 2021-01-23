import cv2
import shutil
import argparse
import numpy as np

from src.segm.predictor import SegmPredictor
from src.datasets import get_folds_data
from src.utils import get_best_model_path

from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--segm', required=True, type=str)
args = parser.parse_args()

SEGM_EXPERIMENT = args.segm
BATCH_SIZE = 8
DEVICE = 'cuda'


def segmentation_pred():
    segm_experiment_dir = config.segm_experiments_dir / SEGM_EXPERIMENT
    segm_prediction_dir = config.segm_predictions_dir / SEGM_EXPERIMENT
    if segm_prediction_dir.exists():
        shutil.rmtree(segm_prediction_dir)
    segm_prediction_dir.mkdir(parents=True, exist_ok=True)
    print(f"Segm experiment dir: {segm_experiment_dir}")
    print(f"Segm prediction dir: {segm_prediction_dir}")

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
            mask_path = str(segm_prediction_dir / (study_id + '.jpg'))
            cv2.imwrite(mask_path, mask)


if __name__ == "__main__":
    print("Segm experiment", SEGM_EXPERIMENT)
    print("Device", DEVICE)
    print("Batch size", BATCH_SIZE)

    segmentation_pred()
