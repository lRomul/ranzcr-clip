import argparse
import numpy as np

from src.utils import (
    remove_than_make_dir,
    load_and_blend_preds,
    save_and_score_val_subm
)

from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
args = parser.parse_args()


def prepare_val(experiments):
    pred_paths = [config.predictions_dir / e / 'val' / 'preds.npz'
                  for e in experiments]
    print(f'Blend: {pred_paths}')
    blend_preds, study_ids = load_and_blend_preds(pred_paths)

    val_pred_dir = config.predictions_dir / ','.join(experiments) / 'val'
    remove_than_make_dir(val_pred_dir)
    print(f'Save to {val_pred_dir}')
    np.savez(
        val_pred_dir / 'preds.npz',
        preds=blend_preds,
        study_ids=study_ids,
    )

    save_and_score_val_subm(blend_preds, study_ids, val_pred_dir)


def prepare_test(experiments):
    for fold in config.folds:
        pred_paths = [config.predictions_dir / e / 'test' / f'fold_{fold}' / 'preds.npz'
                      for e in experiments]
        print(f'Blend: {pred_paths}')
        blend_preds, study_ids = load_and_blend_preds(pred_paths)

        test_pred_dir = config.predictions_dir / ','.join(experiments) / 'test' / f'fold_{fold}'
        remove_than_make_dir(test_pred_dir)
        print(f'Save to {test_pred_dir}')
        np.savez(
            test_pred_dir / 'preds.npz',
            preds=blend_preds,
            study_ids=study_ids,
        )


if __name__ == "__main__":
    experiments = sorted(args.experiment.split(','))
    assert experiments
    print("Experiments", experiments)

    prepare_val(experiments)
    prepare_test(experiments)
