import cv2
import random
import numpy as np

import albumentations as alb
import albumentations.pytorch

cv2.setNumThreads(0)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, trg=None):
        if trg is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, trg = t(image, trg)
            return image, trg


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if trg is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, trg = self.transform(image, trg)
            return image, trg


class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, trg=None):
        transform = np.random.choice(self.transforms, p=self.p)
        if trg is None:
            image = transform(image)
            return image
        else:
            image, trg = transform(image, trg)
            return image, trg


class Albumentations:
    def __init__(self, transforms, p=1.0):
        self.albumentations = alb.Compose(transforms, p=p)

    def __call__(self, image):
        augmented = self.albumentations(image=image)
        image = augmented["image"]
        return image


def get_transforms(train: bool, size: int,
                   interpolation=cv2.INTER_CUBIC,
                   border_mode=cv2.BORDER_CONSTANT):
    if train:
        transforms = Albumentations([
            alb.RandomResizedCrop(size, size, interpolation=interpolation,
                                  scale=(0.9, 1), p=1),
            alb.HorizontalFlip(p=0.5),
            alb.ShiftScaleRotate(rotate_limit=20, scale_limit=0, border_mode=border_mode,
                                 interpolation=interpolation, p=0.5),
            alb.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),
                                         contrast_limit=(-0.2, 0.2), p=0.7),
            alb.CLAHE(clip_limit=(1, 4), p=0.5),
            alb.OneOf([
                alb.OpticalDistortion(distort_limit=0.5, border_mode=border_mode,
                                      interpolation=interpolation),
                alb.GridDistortion(num_steps=5, distort_limit=0.5,
                                   border_mode=border_mode, interpolation=interpolation),
                alb.ElasticTransform(alpha=2, border_mode=border_mode,
                                     interpolation=interpolation),
            ], p=0.2),
            alb.OneOf([
                alb.GaussNoise(var_limit=(10.0, 50.0)),
                alb.GaussianBlur(),
                alb.MotionBlur(),
                alb.MedianBlur(),
            ], p=0.2),
            alb.OneOf([
                alb.JpegCompression(),
                alb.Downscale(scale_min=0.25, scale_max=0.5,
                              interpolation=interpolation),
            ], p=0.2),
            alb.IAAPiecewiseAffine(p=0.2),
            alb.IAASharpen(p=0.2),
            alb.Cutout(max_h_size=int(size * 0.1), max_w_size=int(size * 0.1),
                       num_holes=5, p=0.5),
            alb.Normalize(mean=[0.485], std=[0.229]),
            alb.pytorch.ToTensorV2()
        ])
    else:
        transforms = Albumentations([
            alb.Resize(size, size, interpolation=interpolation),
            alb.Normalize(mean=[0.485], std=[0.229]),
            alb.pytorch.ToTensorV2()
        ])
    return transforms
