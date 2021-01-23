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

    def __call__(self, image, trg=None):
        if trg is None:
            if isinstance(image, (tuple, list)):
                img_lst = []
                for img in image:
                    augmented = self.albumentations(image=img)
                    img_lst.append(augmented["image"])
                return img_lst
            else:
                augmented = self.albumentations(image=image)
                image = augmented["image"]
                return image
        else:
            augmented = self.albumentations(image=image, mask=trg)
            image = augmented["image"]
            trg = augmented["mask"]
            return image, trg


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
            alb.CoarseDropout(p=0.2),
            alb.Cutout(p=0.2, max_h_size=16, max_w_size=16,
                       fill_value=0., num_holes=16),
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
