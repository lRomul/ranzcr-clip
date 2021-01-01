import cv2
import random
import numpy as np

import torch
import albumentations as alb


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


class ImageToTensor:
    def __call__(self, image):
        image = np.stack([image, image, image], axis=0)
        image = image.astype(np.float32) / 256
        image = torch.from_numpy(image)
        return image


class Albumentations:
    def __init__(self, transforms, p=1.0):
        self.albumentations = alb.Compose(transforms, p=p)

    def __call__(self, image):
        augmented = self.albumentations(image=image)
        image = augmented["image"]
        return image


def get_transforms(train: bool, size: int):
    if train:
        transforms = Compose([
            Albumentations([
                alb.RandomResizedCrop(size, size),
                alb.Normalize(mean=[0.485], std=[0.229])
            ]),
            ImageToTensor()
        ])
    else:
        transforms = Compose([
            Albumentations([
                alb.Resize(size, size),
                alb.Normalize(mean=[0.485], std=[0.229])
            ]),
            ImageToTensor()
        ])
    return transforms
