from pathlib import Path
import numpy as np
import logging
import torch
from .base_dataset import BaseDataset
from .view import read_image, numpy_image_to_torch, resize_image
import os
import cv2

logger = logging.getLogger(__name__)


class ExternalDataset(BaseDataset):
    default_conf = {
        'data_dir': '',
        'train_split': '',
        'val_split': '',
        'test_split': '',
        'grayscale': False,
        'seed': 0,
        'resize_method': 'simple',
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _Dataset(self.conf)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf):
        self.conf = conf
        self.items = []
        for path in os.listdir(conf.data_dir):
            path = conf.data_dir + path
            self.items.append({'path': path})

    def __getitem__(self, idx):

        item = self.items[idx]
        path = item['path']
        im = read_image(Path(path), grayscale=False)
        im = numpy_image_to_torch(resize_image(im, resize_method=self.conf.resize_method))
    
        data = {
            'image': im,
            **item,
        }
        return data

    def __len__(self):
        return len(self.items)
