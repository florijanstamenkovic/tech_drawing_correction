import logging
import os
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from tech_drawing_correction import augment


SIDE_LENGTH = 800
LOGGER = logging.getLogger(__name__)


def _load(path):
    img = Image.open(path)
    img = img.convert("L")
    img_np = np.array(img, dtype=np.float32) / 255
    return img_np


def _save(img_np, path):
    Image.fromarray(img_np * 255).convert("RGB").save(path)


class _Dataset(Dataset):

    def __init__(self, y_dir, x_dir, limit):
        self._x = []
        self._y = []

        for file_name in os.listdir(y_dir)[:limit]:
            y = _load(os.path.join(y_dir, file_name))
            self._y.append(y)

            if x_dir is not None:
                os.makedirs(x_dir, exist_ok=True)
                x_path = os.path.join(x_dir, file_name)
                if os.path.exists(x_path):
                    x = _load(x_path)
                else:
                    x = augment.random_augment(y)
                    _save(x, x_path)
                self._x.append(x)


    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        y = self._y[idx]
        h, w = y.shape

        crop_h = random.randint(0, h - SIDE_LENGTH - 1)
        crop_w = random.randint(0, w - SIDE_LENGTH - 1)

        y_crop = y[crop_h:crop_h + SIDE_LENGTH, crop_w:crop_w + SIDE_LENGTH]

        if len(self._x) > idx:
            x = self._x[idx]
        else:
            x = augment.random_augment(y)

        x_crop = x[crop_h:crop_h + SIDE_LENGTH, crop_w:crop_w + SIDE_LENGTH]

        return (torch.FloatTensor(np.expand_dims(x_crop, axis=0)),
                torch.FloatTensor(np.expand_dims(y_crop, axis=0)))


class TrainDataset(_Dataset):
    def __init__(self, limit):
        LOGGER.info("Loading train data...")
        super().__init__("./data/train/png/", None, limit)
        LOGGER.info("Train dataset loaded, %d images" % len(self._y))


class TestDataset(_Dataset):
    def __init__(self, limit):
        LOGGER.info("Loading test data...")
        super().__init__("./data/test/png/", "./data/test/png_aug/", limit)
        LOGGER.info("Test dataset loaded, %d images" % len(self._y))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    datasets = {"train": TrainDataset(), "test": TestDataset()}
    os.makedirs("output", exist_ok=True)

    def save_tensor(img, path):
        img = img.cpu()
        if img.size()[0] == 1:
            # convert (1, H, W) to (H, W)
            img = img.squeeze(0)
        else:
            # (C, H, W) to (H, W, C)
            img = img.permute(1, 2, 0)
        img = (img.numpy() * 255).astype(np.uint8)
        Image.fromarray(img).convert("RGB").save(path)

    def save(dataset_name):
        dataset = datasets[dataset_name]
        idx = random.randint(0, len(dataset) - 1)
        for i in "abc":
            x, y = dataset[idx]
            save_tensor(x, "output/%s_%d_x_%s.png" % (dataset_name, idx, i))
            save_tensor(y, "output/%s_%d_y_%s.png" % (dataset_name, idx, i))

    for _ in range(3):
        for dataset_name in ("train", "test"):
            save(dataset_name)
