"""
Dataset for training calibration network
"""


from pathlib import Path
import numpy as np
import logging
import torch
import json
from .base_dataset import BaseDataset
from ...settings import DATA_PATH, DATASETS_PATH
import albumentations as A
from .view import read_image, numpy_image_to_torch, resize_image
import cv2

logger = logging.getLogger(__name__)


class CalibrationDataset(BaseDataset):
    default_conf = {
        'data_dir': 'minidata/',
        'train_split': 'train.txt',
        'val_split': 'val.txt',
        'test_split': 'test.txt',
        'grayscale': False,
        'resize_method': 'simple',
        'seed': 0,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _Dataset(self.conf, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.root = Path(DATA_PATH).parent
        self.conf = conf
        self.items = []
        self.split = split

        with open(Path(DATASETS_PATH, self.conf[f'{split}_split']), 'r') as f:
            self.items = f.read().splitlines()
        self.transforms = []
        logger.info(f'Augment is set to {conf.augment}')
        if conf.augment:
            self.transforms = [
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(var_limit=(5.0, 112.0), mean=0,
                             per_channel=True, p=0.75),
                A.Downscale(scale_min=0.5, scale_max=0.95,
                            interpolation=dict(downscale=cv2.INTER_AREA, upscale=cv2.INTER_LINEAR), p=0.5),
                A.Downscale(scale_min=0.5, scale_max=0.95,
                            interpolation=cv2.INTER_LINEAR, p=0.5),
                A.JpegCompression(quality_lower=20,
                                  quality_upper=85, p=1, always_apply=True),
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.2, p=0.4),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                A.ToGray(always_apply=False, p=0.2),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, p=0.25),
                A.MotionBlur(blur_limit=5, allow_shifted=True, p=0.25),
                A.MultiplicativeNoise(
                    multiplier=[0.85, 1.15], elementwise=True, p=0.5)
            ]
            logger.info(
                f"Augmentations used during train and val are: {self.transforms}. These transformations are applied in random order.")
        logger.info(
            f'{len(self.transforms)} transformations are being applied.')

    def __getitem__(self, idx):
        im = read_image(Path(DATA_PATH, self.conf.data_dir,
                        self.items[idx] + '.jpg'), grayscale=self.conf.grayscale)
        im = resize_image(im, resize_method=self.conf.resize_method)
        if self.split == 'train' and len(self.transforms) > 0:
            np.random.shuffle(self.transforms)
            transform = A.Compose([*self.transforms])
            im = transform(image=im)["image"]

        with open(Path(DATA_PATH, self.conf.data_dir, self.items[idx] + '.json')) as f:
            labels = json.load(f)
        im = numpy_image_to_torch(im)
        data = {
            'image': im,
            'path': str(Path(DATA_PATH, self.conf.data_dir, self.items[idx] + '.jpg')),
            **labels
        }
        return data

    def __len__(self):
        return len(self.items)
