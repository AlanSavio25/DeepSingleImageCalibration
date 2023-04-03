from pathlib import Path
import numpy as np
import logging
import torch
from .base_dataset import BaseDataset
from .view import read_image, numpy_image_to_torch
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
        original_height, original_width = im.shape[:2]

        if original_width >= original_height:
            aspect_ratio = original_width / original_height
            downsampled_width = int(224 * aspect_ratio)
            downsampled_image = cv2.resize(
                im, (downsampled_width, 224), interpolation=cv2.INTER_AREA)

            down_height, down_width = downsampled_image.shape[:2]
            assert down_height <= down_width
            start_x = (down_width - down_height) // 2
            start_y = 0
            cropped_image_numpy = np.array(
                downsampled_image[start_y:start_y+down_height, start_x:start_x+down_height])
            cropped_image_torch = numpy_image_to_torch(cropped_image_numpy)

        elif original_height > original_width:
            aspect_ratio = original_height / original_width
            downsampled_height = int(224 * aspect_ratio)
            downsampled_image = cv2.resize(
                im, (224, downsampled_height), interpolation=cv2.INTER_AREA)

            down_height, down_width = downsampled_image.shape[:2]
            assert down_width <= down_height
            start_x = 0
            start_y = (down_height - down_width) // 2
            cropped_image_numpy = np.array(
                downsampled_image[start_y:start_y+down_width, start_x:start_x+down_width])
            cropped_image_torch = numpy_image_to_torch(cropped_image_numpy)

        assert cropped_image_torch.shape[1] == cropped_image_torch.shape[2] == 224

        data = {
            'image': cropped_image_torch,
            **item,
        }
        return data

    def __len__(self):
        return len(self.items)
