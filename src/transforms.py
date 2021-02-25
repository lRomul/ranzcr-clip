import cv2

import albumentations as alb
import albumentations.pytorch

cv2.setNumThreads(0)


class Albumentations:
    def __init__(self, transforms, p=1.0):
        self.albumentations = alb.Compose(transforms, p=p)

    def __call__(self, image, mask=None):
        if mask is None:
            augmented = self.albumentations(image=image)
            image = augmented["image"]
            return image
        else:
            augmented = self.albumentations(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            return image, mask


def get_transforms(train: bool, size: int, n_channels: int,
                   interpolation=cv2.INTER_CUBIC):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if train:
        transforms = Albumentations([
            alb.Resize(size, size, interpolation=interpolation),
            alb.HorizontalFlip(p=0.5),
            alb.Normalize(mean=mean[:n_channels], std=[std[:n_channels]]),
            alb.pytorch.ToTensorV2()
        ])
    else:
        transforms = Albumentations([
            alb.Resize(size, size, interpolation=interpolation),
            alb.Normalize(mean=mean[:n_channels], std=[std[:n_channels]]),
            alb.pytorch.ToTensorV2()
        ])
    transforms.n_channels = n_channels
    return transforms
