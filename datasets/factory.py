import albumentations as alb
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


def create_data_transforms(opt):
    data_transform = {
        'base':
            alb.Compose([
                alb.Flip(),
                alb.RandomRotate90(p=0.5),
                alb.RandomResizedCrop(opt['datasets']['image_size'], opt['datasets']['image_size'], scale=(0.3, 1.0), ratio=(1.0, 1.0), p=0.4), #
                alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5), #  
                alb.CenterCrop(opt['datasets']['image_size'], opt['datasets']['image_size'])
            ]),
        'train':
            alb.Compose([
                alb.ToGray(p=0.1),
                alb.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
                alb.GaussianBlur(blur_limit=3, sigma_limit=0, p=0.05),
                alb.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.1),
                alb.OneOf([
                    alb.RandomBrightnessContrast(),
                    alb.FancyPCA(),
                    alb.HueSaturationValue(),
                ]),
                alb.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]),
        'val':
            alb.Compose([
                alb.Resize(opt['datasets']['image_size'], opt['datasets']['image_size'], interpolation=1),
                alb.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]),
        'test':
            alb.Compose([
                alb.Resize(opt['datasets']['image_size'], opt['datasets']['image_size'], interpolation=1),
                alb.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]),
        'mask':
            alb.Compose([
                # alb.CenterCrop(opt['datasets']['image_size'], opt['datasets']['image_size']),
                alb.Resize(opt['datasets']['image_size'], opt['datasets']['image_size'], interpolation=1),
                # no normalize
                ToTensorV2(),
            ])
    }

    return data_transform




