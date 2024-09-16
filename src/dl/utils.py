# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 9/21/2023 2:41 PM
# @Last Modified by: zhenwan
# @Last Modified time: 9/21/2023  2:41 PM
# @file_name: utils.
# @IDE: PyCharm
# @copyright: zhenwan
# utils.py
import os

def create_dataset_folder(dataset_name, istrain):
    folder_name = f"experiment_results/{dataset_name}/{istrain}"
    folder_train = f"experiment_results/{dataset_name}/train"
    os.makedirs(folder_name, exist_ok=True)  # 创建文件夹，如果文件夹不存在则创建
    return folder_name, folder_train


class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
