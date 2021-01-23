import re
from pathlib import Path
import matplotlib.pyplot as plt


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
        return None

    model_score = sorted(model_scores, key=lambda x: x[1])
    best_model_path = model_score[-1][0]
    if return_score:
        best_score = model_score[-1][1]
        return best_model_path, best_score
    else:
        return best_model_path
