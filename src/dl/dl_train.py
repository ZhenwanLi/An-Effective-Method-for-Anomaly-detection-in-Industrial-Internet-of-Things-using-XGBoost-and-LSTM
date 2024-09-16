# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 9/21/2023 2:37 PM
# @Last Modified by: zhenwan
# @Last Modified time: 9/21/2023  2:37 PM
# @file_name: train.
# @IDE: PyCharm
# @copyright: zhenwan
# dl_train.py
import os
import time
import hydra
import joblib
import json

import numpy as np
import psutil
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from dl_csv_dataset import CSVDataset, split_dataset
from utils import EarlyStopping

MB_TO_BYTES = 1024 * 1024
GB_TO_BYTES = MB_TO_BYTES * 1024


class TrainDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        print(self.cfg)
        self.name = "train"
        self.dataset = self._initialize_dataset()
        self.csv_file_name = self.dataset.get_csv_filename()
        self.train_data, self.val_data = split_dataset(self.dataset, train_ratio=self.cfg.train_ratio)

    def _initialize_dataset(self):
        return CSVDataset(
            root=self.cfg.data_dir,
            data_name=self.cfg.data_name,
            name=self.name,
            num_classes=self.cfg.num_classes,
            method=self.cfg.method,
            threshold=self.cfg.threshold,
            augmenter=self.cfg.augmenter,
            transform=self.cfg.transform
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )


class Trainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        print(self.cfg)
        self.device = device
        print(self.device)
        self.train_data = TrainDataModule(cfg.data)
        self.model_name = self.cfg.model_name
        self.model = self._initialize_model()
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        self.criterion = self._initialize_criterion()
        self.train_directory = self._get_train_directory()
        print(self.train_directory)
        self.best_val_accuracy = 0.0

    def _get_train_directory(self):
        directory = self.cfg.paths.train_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def _initialize_model(self):
        return hydra.utils.instantiate(self.cfg.model.net)

    def _initialize_optimizer(self):
        optimizer_class = hydra.utils.instantiate(self.cfg.model.optimizer, _recursive_=False)
        return optimizer_class(self.model.parameters())

    def _initialize_scheduler(self):
        return hydra.utils.instantiate(self.cfg.model.scheduler)(self.optimizer)

    def _initialize_criterion(self):
        return hydra.utils.instantiate(self.cfg.model.criterion)

    def train(self):
        if self._all_required_files_exist():
            return
        else:
            self.train_model()

    def _all_required_files_exist(self):
        model_path = self._get_file_path(f"{self.model_name}.pkl")
        records_path = self._get_file_path('training_records.json')
        infos_path = self._get_file_path('training_infos.json')
        return os.path.exists(model_path) and os.path.exists(records_path) and os.path.exists(infos_path)

    def train_model(self):
        start_time = time.time()

        early_stopping = EarlyStopping(patience=20, verbose=True)

        all_training_records = self._iterate_through_epochs(early_stopping)

        self._save_training_info(start_time)
        self._save_training_records(all_training_records)

    def _iterate_through_epochs(self, early_stopping):
        all_training_records = []

        for epoch in range(self.cfg.trainer.max_epochs):

            epoch_train_loss, epoch_train_accuracy = self._train_epoch(self.train_data.train_dataloader())
            epoch_val_loss, epoch_val_accuracy, epoch_val_y_true, epoch_val_y_pred = self._validate_epoch(self.train_data.val_dataloader())

            self._check_and_update_best_val_accuracy(epoch_val_y_true, epoch_val_y_pred)

            all_training_records.append(
                self._create_training_record(epoch, epoch_train_loss, epoch_train_accuracy, epoch_val_loss,
                                             epoch_val_accuracy))

            # Update scheduler
            self.scheduler.step(epoch_val_loss)

            # Early stopping
            if early_stopping(epoch_val_loss):
                print("Early stopping")
                break

        return all_training_records

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_correct += self._get_correct_predictions_count(outputs, labels)
            total_samples += labels.size(0)

        average_loss = total_loss / len(train_loader)
        accuracy = 100 * total_correct / total_samples
        return average_loss, accuracy

    def _validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        y_true = []  # To store true labels
        y_pred = []  # To store predicted labels

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                total_correct += self._get_correct_predictions_count(outputs, labels)
                total_samples += labels.size(0)

                # Append true labels and predicted labels
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        average_loss = total_loss / len(val_loader)
        accuracy = 100 * total_correct / total_samples

        return average_loss, accuracy, y_true, y_pred

    def _calculate_balanced_accuracy(self, y_true, y_pred):
        # 使用混淆矩阵计算平衡准确率
        conf_matrix = confusion_matrix(y_true, y_pred)
        balanced_accuracy = 0.5 * (conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) +
                                   conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]))
        return balanced_accuracy
    def _check_and_update_best_val_accuracy(self,  val_y_true, val_y_pred):
        balanced_accuracy = self._calculate_balanced_accuracy(val_y_true, val_y_pred)
        if balanced_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = balanced_accuracy
            self._save_model()

    def _create_training_record(self, epoch, train_loss, train_accuracy, val_loss, val_accuracy):
        return {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }

    def _get_correct_predictions_count(self, outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        correct_count = (predicted == labels).sum().item()
        return correct_count

    def _save_training_records(self, records):
        self._save_to_json_file('training_records.json', records)

    def _save_training_info(self, start_time):
        end_time = time.time()
        training_time = end_time - start_time

        process = psutil.Process(os.getpid())
        cpu_memory_usage_GB = process.memory_info().rss / GB_TO_BYTES
        # memory_usage_MB = (torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0)
        # gpu_memory_usage_GB = (torch.cuda.memory_allocated(
        #     self.device) if torch.cuda.is_available() else 0) / GB_TO_BYTES

        # Model size
        model_path = self._get_file_path(f"{self.model_name}.pkl")
        joblib.dump(self.model, model_path)
        model_size_mb = os.path.getsize(model_path) / MB_TO_BYTES

        # Save the data
        training_infos = {
            "training time (s)": training_time,
            "model size (MB)": model_size_mb,
            "CPU memory usage (GB)": cpu_memory_usage_GB,
            # "GPU memory usage (GB)": gpu_memory_usage_GB,
        }

        with open(self._get_file_path('training_infos.json'), 'w') as file:
            json.dump(training_infos, file)

    def _save_to_json_file(self, file_name, data):
        file_path = self._get_file_path(file_name)
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, default=lambda obj: int(obj) if isinstance(obj, np.integer) else TypeError)

    def _get_file_path(self, filename):
        return os.path.join(self.train_directory, filename)

    def _save_model(self):
        model_path = self._get_file_path(f"{self.model_name}.pkl")
        joblib.dump(self.model, model_path)

