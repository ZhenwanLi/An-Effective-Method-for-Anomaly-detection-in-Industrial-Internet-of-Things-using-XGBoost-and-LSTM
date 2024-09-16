# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 7/14/2023 4:29 PM
# @Last Modified by: zhenwan
# @Last Modified time: 7/14/2023  4:29 PM
# @file_name: csv_dataset.
# @IDE: PyCharm
# @copyright: zhenwan
import math
import numbers

import pandas as pd
import torch
from torch.utils.data import Dataset

from nsl_kdd_label_types import get_labels


class CSVDataset(Dataset):
    def __init__(self, root, data_name, name, num_classes, method, threshold, augmenter, transform):
        self.root = root
        self.data_name = data_name
        self.name = name
        self.num_classes = num_classes
        self.method = method
        self.threshold = threshold
        self.augmenter = augmenter
        self.transform = transform
        if self.name == "train":
            self.data_file = self.get_train_csvfile()
            self.csv_path = f'{self.root}/{self.data_name}/processed/{self.name}/{self.data_file}'
        if self.name == "val":
            self.data_file = self.get_val_csvfile()
            self.csv_path = f'{self.root}/{self.data_name}/processed/{self.name}/{self.data_file}'
        if self.name in ["test", "test_known", "test_unknown"]:
            self.data_file = self.get_test_csvfile()
            self.csv_path = f'{self.root}/{self.data_name}/processed/test/{self.data_file}'

        self.csv_df = pd.read_csv(self.csv_path)
        # 从DataFrame中提取需要的列作为feature和label
        self.X = self.csv_df.iloc[:, :-1].values
        self.y = self.csv_df.iloc[:, -1].values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx])

    def get_features_nums(self):
        return self.csv_df.shape[1]-1

    def get_data(self):
        return self.X, self.y

    def get_data_file(self):
        return self.data_file

    def split_data(self):
        train_sets = math.floor(len(self.y)*0.3)
        val_sets = len(self.y) - train_sets
        print(f"train_val_split: [{train_sets}, {val_sets}]")

    def get_labels(self):
        return get_labels(self.num_classes)

    def get_train_csvfile(self) -> str:
        filename = f"train_un"
        # if num_classes is not None:
        if isinstance(self.num_classes, numbers.Number):
            filename += f"_{self.num_classes}_num"
            if self.method:
                filename += f"_{self.method}_{self.threshold}"
            if self.augmenter:
                filename += f"_{self.augmenter.lower()}"
            if self.transform:
                filename += f"_{self.transform.upper()}"
        return filename + ".csv"

    def get_val_csvfile(self) -> str:
        filename = f"val_un"
        # if num_classes is not None:
        if isinstance(self.num_classes, numbers.Number):
            filename += f"_{self.num_classes}_num"
            if self.method:
                filename += f"_{self.method}_{self.threshold}"
            if self.augmenter:
                filename += f"_{self.augmenter.lower()}"
            if self.transform:
                filename += f"_{self.transform.upper()}"
        return filename + ".csv"

    def get_test_csvfile(self) -> str:
        filename = f"test"
        # if num_classes is not None:
        if isinstance(self.num_classes, numbers.Number):
            filename += f"_{self.num_classes}_num"
            if self.method:
                filename += f"_{self.method}_{self.threshold}"
            if self.augmenter:
                filename += f"_{self.augmenter.lower()}"
            if self.transform:
                filename += f"_{self.transform.upper()}"
        return filename + ".csv"


if __name__ == '__main__':
    for root in [f'../../../data']:

        print(root)
        for data_name in ['UNSW_NB15_10']:
            for name in ['train', 'val', 'test']:
                for num_classes in [2, 10]:
                    for method in ["xgboost"]:
                        # for threshold in [0, 0.005, 0.01, 0.05]:
                        for threshold in [0.002]:
                            for augmenter in ['', "smote"]:
                                for transform in ['']:
                                    nsl_kdd = CSVDataset(root, data_name, name, num_classes, method, threshold, augmenter, transform)
                                    print(nsl_kdd)
                                    print(nsl_kdd.get_data_file())
                                    print(nsl_kdd.get_features_nums())
                                    # print(nsl_kdd.X.shape)
                                    # print(nsl_kdd.y.shape)
