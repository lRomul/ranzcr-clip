import re
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from src import config


def image_show(image, title='', figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.margins(0, 0)
    plt.show()


def get_best_model_path(dir_path, return_score=False):
    dir_path = Path(dir_path)
    model_scores = []
    for model_path in dir_path.glob('*.pth'):
        score = re.search(r'-(\d+(?:\.\d+)?).pth', str(model_path))
        if score is not None:
            score = float(score.group(0)[1:-4])
            model_scores.append((model_path, score))

    if not model_scores:
        if return_score:
            return None, -np.inf
        else:
            return None

    model_score = sorted(model_scores, key=lambda x: x[1])
    best_model_path = model_score[-1][0]
    if return_score:
        best_score = model_score[-1][1]
        return best_model_path, best_score
    else:
        return best_model_path


def remove_than_make_dir(dir_path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def load_preds(pred_paths):
    pred_lst = []
    study_ids = None
    for pred_path in pred_paths:
        pred_npz = np.load(pred_path)
        preds = pred_npz['preds']
        if study_ids is not None:
            assert np.all(study_ids == pred_npz['study_ids'])
        study_ids = pred_npz['study_ids']
        pred_lst.append(preds)
    return pred_lst, study_ids


def load_and_blend_preds(pred_paths):
    pred_lst, study_ids = load_preds(pred_paths)
    blend_preds = np.mean(pred_lst, axis=0)
    return blend_preds, study_ids


def load_and_concat_preds(pred_paths):
    pred_lst, study_ids = load_preds(pred_paths)
    concat_preds = np.concatenate(pred_lst, axis=1)
    return concat_preds, study_ids


def save_and_score_val_subm(pred, study_ids, dir_path):
    subm_df = pd.DataFrame(index=study_ids, columns=config.classes)
    subm_df.index.name = 'StudyInstanceUID'
    subm_df.values[:] = pred
    subm_df.to_csv(dir_path / 'submission.csv')

    train_df = pd.read_csv(config.train_folds_path, index_col=0)
    train_df = train_df.loc[subm_df.index].copy()
    scores = roc_auc_score(train_df[config.classes].values,
                           subm_df[config.classes].values, average=None)
    scores_dict = {cls: scr for cls, scr in zip(config.classes, scores)}
    scores_dict['Overal'] = np.mean(scores)

    with open(dir_path / 'scores.json', 'w') as outfile:
        json.dump(scores_dict, outfile)
