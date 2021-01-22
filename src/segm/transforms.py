import cv2

import albumentations as alb
import albumentations.pytorch

from src.transforms import (
    Albumentations
)

cv2.setNumThreads(0)


def get_transforms(train: bool, size: int,
                   interpolation=cv2.INTER_CUBIC,
                   border_mode=cv2.BORDER_CONSTANT):
    if train:
        transforms = Albumentations([
            alb.RandomResizedCrop(size, size, scale=(0.8, 1.0),
                                  interpolation=interpolation),
            alb.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2),
                                         contrast_limit=(-0.2, 0.2)),
            alb.ShiftScaleRotate(p=0.2, shift_limit=0.0625,
                                 scale_limit=0.2, rotate_limit=20,
                                 interpolation=interpolation,
                                 border_mode=border_mode),
            alb.Normalize(mean=[0.485], std=[0.229]),
            alb.pytorch.ToTensorV2(transpose_mask=True)
        ])
    else:
        transforms = Albumentations([
            alb.Resize(size, size, interpolation=interpolation),
            alb.Normalize(mean=[0.485], std=[0.229]),
            alb.pytorch.ToTensorV2(transpose_mask=True)
        ])
    return transforms
