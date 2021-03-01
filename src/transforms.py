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
                   interpolation=cv2.INTER_CUBIC,
                   border_mode=cv2.BORDER_CONSTANT):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if train:
        transforms = Albumentations([
            alb.RandomResizedCrop(size, size, scale=(0.8, 1.0),
                                  interpolation=interpolation),
            alb.HorizontalFlip(p=0.5),
            alb.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2),
                                         contrast_limit=(-0.2, 0.2)),
            alb.ShiftScaleRotate(p=0.2, shift_limit=0.0625,
                                 scale_limit=0.2, rotate_limit=20,
                                 interpolation=interpolation,
                                 border_mode=border_mode),
            alb.CoarseDropout(p=0.3, max_holes=16,
                              max_height=16, max_width=16),
            alb.Cutout(p=0.3, max_h_size=32, max_w_size=32,
                       fill_value=0., num_holes=16),
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
