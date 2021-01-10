import cv2
import shutil

from src.datasets import get_folds_data, draw_visualization
from src import config


if __name__ == "__main__":
    if config.train_visualizations_dir.exists():
        shutil.rmtree(config.train_visualizations_dir)

    config.train_visualizations_dir.mkdir(parents=True, exist_ok=True)

    folds_data = get_folds_data()
    for sample in folds_data:
        image = draw_visualization(sample)
        image_name = sample['StudyInstanceUID'] + '.jpg'
        cv2.imwrite(str(config.train_visualizations_dir / image_name), image)
