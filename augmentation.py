import albumentations as albu
import numpy as np
from albumentations import (
    Blur, HorizontalFlip, IAAEmboss, OneOf,
    RandomBrightnessContrast, RandomCrop, RandomGamma, RandomRotate90,
    ShiftScaleRotate, Transpose, VerticalFlip, ElasticTransform,
    GridDistortion, OpticalDistortion)


def aug_with_crop(image_size = 256, crop_prob = 1):
    # Monkey-patch lol
    albu.augmentations.functional.MAX_VALUES_BY_DTYPE[np.dtype('float64')] = 1.0
    return albu.Compose([
        RandomCrop(width = image_size, height = image_size, p=crop_prob),
        HorizontalFlip(p=0.3),
        VerticalFlip(p=0.3),
        RandomRotate90(p=0.3),
        Transpose(p=0.3),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.1),
        RandomBrightnessContrast(p=0.3),
        RandomGamma(p=0.1),
        IAAEmboss(p=0.1),
        Blur(p=0.001, blur_limit = 3),
        OneOf([
            ElasticTransform(p=0.1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.1),
            OpticalDistortion(p=0.05, distort_limit=2, shift_limit=0.5)
        ], p=0.5)
    ], p = 1)
