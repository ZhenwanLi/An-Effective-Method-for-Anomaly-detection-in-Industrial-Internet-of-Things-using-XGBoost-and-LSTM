# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 9/21/2023 2:38 PM
# @Last Modified by: zhenwan
# @Last Modified time: 9/21/2023  2:38 PM
# @file_name: eval.
# @IDE: PyCharm
# @copyright: zhenwan
# dl_test.py
import io
import json
import os
import time

import hydra
import joblib
import numpy as np
import psutil
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from dl_csv_dataset import CSVDataset
from torch.utils.data import DataLoader

# Constants
MB_TO_BYTES = 1024 * 1024
GB_TO_BYTES = MB_TO_BYTES * 1024


class TestDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.test_data = self._initialize_dataset()
        self.csv_file_name = self.test_data.get_csv_filename()

    def _initialize_dataset(self):
        return CSVDataset(
            root=self.cfg.data_dir,
            data_name=self.cfg.data_name,
            name="test",
            num_classes=self.cfg.num_classes,
            method=self.cfg.method,
            threshold=self.cfg.threshold,
            augmenter=self.cfg.augmenter,
            transform=self.cfg.transform
        )

    def __str__(self):
        return f"TestDataModule with {len(self.test_data)} samples."

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )


class Evaluator:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        print(self.device)
        self.test_data = TestDataModule(cfg.data)
        self.model_name = self.cfg.model_name
        self.criterion = hydra.utils.instantiate(self.cfg.model.criterion)
        self.model = self._load_model()
        self.test_directory = self._get_test_directory()

    def _load_model(self):
        try:
            model = joblib.load(self._get_model_path())
            model.eval()
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model from {self._get_model_path()}. Reason: {e}")

    def eval(self):
        """Evaluate the model and save results."""
        if self._all_required_files_exist():
            return
        else:
            self.eval_model()

    def eval_tsne(self):
        if self._all_required_tsne_files_exist():
            return
        elif self.model_name == 'MIX_LSTM':
            labels = self._get_tsne_y_data()
            self._save_tsne_y_data(labels)

            layer_names = [name for name, _ in self.model.named_children()]
            for layer_name in layer_names:
                outputs = self._get_tsne_X_data(layer_name)
                # self._save_tsne_X_data(layer_name, outputs)
                self._save_reduced_data(layer_name=layer_name, activations=outputs)
        else:
            return

    def _get_tsne_X_data(self, layer_name):
        test_loader = self.test_data.test_dataloader()
        activations_list = []

        with torch.no_grad():
            for X_test, y_test in test_loader:
                activations = self._get_intermediate_outputs(layer_name=layer_name, input_data=X_test)
                batch_value = X_test.size()[0]
                reshaped_activations = self.reshape_tensor(activations, batch_value)
                activations_list.append(reshaped_activations)

        all_activations = torch.cat(activations_list, dim=0)

        return all_activations

    def _get_tsne_y_data(self):
        test_loader = self.test_data.test_dataloader()
        labels_list = []

        with torch.no_grad():
            for X_test, y_test in test_loader:
                labels_list.append(y_test)

        all_labels = torch.cat(labels_list, dim=0)

        return all_labels

    def reshape_tensor(self, tensor, target_dim_value):
        # 检查Tensor的shape中是否存在目标维度值
        if target_dim_value in tensor.shape:
            # 获取特定值的位置
            target_dim_idx = tensor.shape.index(target_dim_value)

            # 计算第二个维度的值
            other_dims_product = 1
            for idx, dim_value in enumerate(tensor.shape):
                if idx != target_dim_idx:
                    other_dims_product *= dim_value

            # 根据新的维度重塑Tensor
            reshaped_tensor = tensor.view(target_dim_value, other_dims_product)

            return reshaped_tensor
        else:
            # 如果Tensor的shape中没有目标维度值，则返回原始Tensor
            return tensor

    def _get_intermediate_outputs(self, layer_name, input_data):
        activations = None

        def hook(_, __, output):
            nonlocal activations
            activations = output

        hook_ref = getattr(self.model, layer_name).register_forward_hook(hook)
        self.model(input_data)
        hook_ref.remove()

        # 如果activations是一个tuple
        if isinstance(activations, tuple):
            # 假设我们需要第一个tensor（这只是一个起点，您可能需要根据模型的特定情况进行调整）
            activations = activations[0]

        return activations

    def eval_model(self):
        start_time = time.time()
        y_test, y_pred, y_pred_pro = self._get_predictions()
        self._save_testing_info(time.time() - start_time)
        self._save_testing_records(y_test, y_pred, y_pred_pro)
        self._save_metrics(y_test, y_pred, y_pred_pro)

    def _all_required_files_exist(self):
        records_path = self._get_file_path('testing_records.json')
        infos_path = self._get_file_path('testing_infos.json')
        required_files = [records_path, infos_path] + [
            self._get_file_path(f'{prefix}_metrics.json')
            for prefix in ['basic', 'advanced', 'derived']
        ]
        return all(os.path.exists(file) for file in required_files)

    def _all_required_tsne_files_exist(self):
        required_files = [
            os.path.join(self._get_tsne_directory(), f'{prefix}.npy')
            for prefix in ['fc', 'label', 'lstm_layer1', 'lstm_layer2', 'lstm_layer3']
        ]
        return all(os.path.exists(file) for file in required_files)

    def _get_predictions(self):
        self.model.eval()
        test_loader = self.test_data.test_dataloader()
        all_y_pred, all_y_pred_pro, all_y_test = [], [], []

        for X_test, y_test in test_loader:
            with torch.no_grad():
                test_outputs = self.model(X_test)
                print(test_outputs)
                test_predicted_classes = torch.argmax(test_outputs, dim=1)


                all_y_pred.append(test_predicted_classes.numpy())
                all_y_pred_pro.append(test_outputs.numpy())
                all_y_test.append(y_test.numpy())

        return (
            np.concatenate(all_y_test, axis=0),
            np.concatenate(all_y_pred, axis=0),
            np.concatenate(all_y_pred_pro, axis=0)
        )

    def _get_file_path(self, filename):
        return os.path.join(self.test_directory, filename)

    def _save_to_json_file(self, file_name, data):
        file_path = self._get_file_path(file_name)
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, default=lambda obj: int(obj) if isinstance(obj, np.integer) else TypeError)

    def _save_testing_info(self, testing_time):
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        model_size_MB = buffer.tell() / MB_TO_BYTES

        process = psutil.Process(os.getpid())
        cpu_memory_usage_GB = process.memory_info().rss / GB_TO_BYTES
        # memory_usage_MB = (torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0)
        # memory_usage_GB = (torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0) / GB_TO_BYTES

        testing_info = {
            "testing time (s)": testing_time,
            "model size (MB)": model_size_MB,
            "memory usage (GB)": cpu_memory_usage_GB
        }
        self._save_to_json_file('testing_infos.json', testing_info)

    def _save_testing_records(self, y_test, y_pred, y_pred_pro):
        results = {
            "model_name": self.model_name,
            "target_names": list(self.cfg.data.target_names),
            "test_directory": self.test_directory,
            "y_pred": y_pred.tolist(),
            "y_pred_pro": y_pred_pro.tolist(),
            "y_test": y_test.tolist()
        }
        self._save_to_json_file('testing_records.json', results)

    def _get_basic_metrics(self, y_test, y_pred):
        num_classes = len(np.unique(y_test))
        metrics = {}
        for cls in range(num_classes):
            y_test_bin = np.where(y_test == cls, 1, 0)
            y_pred_bin = np.where(y_pred == cls, 1, 0)
            TP = np.sum((y_test_bin == 1) & (y_pred_bin == 1))
            TN = np.sum((y_test_bin == 0) & (y_pred_bin == 0))
            FP = np.sum((y_test_bin == 0) & (y_pred_bin == 1))
            FN = np.sum((y_test_bin == 1) & (y_pred_bin == 0))
            label = self.cfg.data.target_names[cls]
            metrics[label] = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
        return metrics

    def _calculate_metrics_from_file(self, file_path):
        # Load the basic metrics from the provided JSON file
        with open(file_path, 'r') as file:
            metrics = json.load(file)

        print(metrics)
        derived_metrics = {}

        for class_name, class_metrics in metrics.items():
            TP = class_metrics['TP']
            TN = class_metrics['TN']
            FP = class_metrics['FP']
            FN = class_metrics['FN']

            try:
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                specificity = TN / (TN + FP)
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                npv = TN / (TN + FN)
                fpr = FP / (FP + TN)
                fdr = FP / (FP + TP)

                derived_metrics[class_name] = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "Specificity": specificity,
                    "F1 Score": f1_score,
                    "NPV": npv,
                    "FPR": fpr,
                    "FDR": fdr
                }
            except ZeroDivisionError:
                raise ValueError(f"Denominator in one of the metric calculations for class '{class_name}' is zero.")

        return derived_metrics

    def _get_advanced_metrics(self, y_test, y_pred, y_pred_pro):
        num_classes = len(np.unique(y_test))
        metrics = {}

        for cls in range(num_classes):
            y_test_bin = np.where(y_test == cls, 1, 0)

            # AUC-ROC
            if len(np.unique(y_test_bin)) > 1:
                roc_auc = roc_auc_score(y_test_bin, y_pred_pro[:, cls])
            else:
                roc_auc = None

            # Matthews correlation coefficient
            mcc = matthews_corrcoef(y_test_bin, np.where(y_pred == cls, 1, 0))

            # False Alarm Rate
            TN, FP, FN, TP = confusion_matrix(y_test_bin, np.where(y_pred == cls, 1, 0)).ravel()
            FAR = FP / (FP + TN) if (FP + TN) > 0 else None

            # Area Under the PR Curve
            if len(np.unique(y_test_bin)) > 1:
                pr_auc = average_precision_score(y_test_bin, y_pred_pro[:, cls])
            else:
                pr_auc = None

            label = self.cfg.data.target_names[cls]
            metrics[label] = {"AUC-ROC": roc_auc, "MCC": mcc, "FAR": FAR, "PR AUC": pr_auc}

        return metrics

    def _save_metrics(self, y_test, y_pred, y_pred_pro):
        basic_metrics = self._get_basic_metrics(y_test, y_pred)
        advanced_metrics = self._get_advanced_metrics(y_test, y_pred, y_pred_pro)

        self._save_to_json_file('basic_metrics.json', basic_metrics)
        self._save_to_json_file('advanced_metrics.json', advanced_metrics)

        basic_metrics_path = self._get_file_path('basic_metrics.json')
        derived_metrics = self._calculate_metrics_from_file(basic_metrics_path)
        self._save_to_json_file('derived_metrics.json', derived_metrics)

    def _get_test_directory(self):
        directory = self.cfg.paths.eval_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def _get_tsne_directory(self):
        directory = self.cfg.paths.tsne_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def _get_model_path(self):
        return os.path.join(self.cfg.paths.train_dir, f"{self.model_name}.pkl")

    def _save_tsne_X_data(self, layer_name, outputs):
        self.tsne_directory = self._get_tsne_directory()
        X_file_path = os.path.join(self.tsne_directory, f'{layer_name}.npy')
        np.save(X_file_path, outputs)

    def _save_tsne_y_data(self, labels):
        self.tsne_directory = self._get_tsne_directory()
        y_file_path = os.path.join(self.tsne_directory, f'label.npy')
        np.save(y_file_path, labels)

    def _save_reduced_data(self, layer_name, activations):
        tsne = TSNE(n_components=2, random_state=0)
        reduced_data = tsne.fit_transform(activations)
        file_path = os.path.join(self.tsne_directory, f'{layer_name}.npy')
        np.save(file_path, reduced_data)
