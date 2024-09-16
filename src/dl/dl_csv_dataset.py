# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 7/14/2023 4:29 PM
# @Last Modified by: zhenwan
# @Last Modified time: 7/14/2023  4:29 PM
# @file_name: csv_dataset.
# @IDE: PyCharm
# @copyright: zhenwan
# code_first.src.dl.components.dl_csv_dataset.py
import numbers
import pandas as pd
import torch
from torch.utils.data import Dataset
from os.path import join
from torch.utils.data import random_split


class DatasetType:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TEST_KNOWN = "test_known"
    TEST_UNKNOWN = "test_unknown"


class CSVDataset(Dataset):
    def __init__(self, root: str, data_name: str, name: str, num_classes: int, method: str, threshold: float,
                 augmenter: str, transform: str):
        self.root = root
        self.data_name = data_name
        self.name = name
        self.num_classes = num_classes
        self.method = method
        self.threshold = threshold
        self.augmenter = augmenter
        self.transform = transform

        # self.data_file = self.get_csvfile()
        self.csv_path = join(self.root, self.data_name, 'processed', self.name, self.get_csvfile())

        try:
            self.csv_df = pd.read_csv(self.csv_path)
            self.X = self.csv_df.iloc[:, :-1].values
            self.y = self.csv_df.iloc[:, -1].values
        except FileNotFoundError as e:
            # Handle the error gracefully (e.g., log an error message)
            raise e

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx])

    def get_features_nums(self):
        return self.csv_df.shape[1] - 1

    def get_data(self):
        return self.X, self.y

    # def get_data_file(self):
    #     return self.data_file

    # def get_labels(self):
    #     return get_labels(self.num_classes)

    def get_csvfile(self) -> str:
        if self.name == DatasetType.TRAIN:
            prefix = "train_un"
        elif self.name == DatasetType.VAL:
            prefix = "val_un"
        else:
            prefix = "test"

        if isinstance(self.num_classes, numbers.Number):
            prefix += f"_{self.num_classes}_num"
            if self.method:
                prefix += f"_{self.method}_{self.threshold}"
            if self.augmenter:
                prefix += f"_{self.augmenter.lower()}"
            if self.transform:
                prefix += f"_{self.transform.upper()}"

        return prefix + ".csv"

    def get_csv_filename(self) -> str:
        if self.name == DatasetType.TRAIN:
            prefix = "train_un"
        elif self.name == DatasetType.VAL:
            prefix = "val_un"
        else:
            prefix = "test"

        if isinstance(self.num_classes, numbers.Number):
            prefix += f"_{self.num_classes}_num"
            if self.method:
                prefix += f"_{self.method}_{self.threshold}"
            if self.augmenter:
                prefix += f"_{self.augmenter.lower()}"
            if self.transform:
                prefix += f"_{self.transform.upper()}"

        return prefix


def split_dataset(dataset, train_ratio=0.7):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [train_size, val_size], generator=generator)
