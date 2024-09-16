# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 5/15/2023 10:25 AM
# @Last Modified by: zhenwan
# @Last Modified time: 5/15/2023  10:25 AM
# @file_name: cnn_lstm_Adam_CBL.
# @IDE: PyCharm
# @copyright: zhenwan
import numbers
from string import Template
import os
from itertools import product

from src.components.data import unsw_nb15_10_label_types, nsl_kdd_label_types, kdd99_10_label_types
from src.components.data.nsl_kdd import NSL_KDD
from src.components.data.unsw_nb15_10 import UNSW_NB15_10
from src.dl.dl_csv_dataset import CSVDataset


def get_file_name(data_name, num_classes, method, threshold, input_size, augmenter, transform, model, optimizer,
                  criterion, max_epochs, ext):
    if augmenter and transform:
        file_name = f"{data_name}_{num_classes}_{method}_{threshold}_{input_size}_{transform}_{augmenter}_{model}_{optimizer}_{criterion}_{max_epochs}{ext}"
        return file_name
    elif augmenter:
        file_name = f"{data_name}_{num_classes}_{method}_{threshold}_{input_size}_{augmenter}_{model}_{optimizer}_{criterion}_{max_epochs}{ext}"
        return file_name
    elif transform:
        file_name = f"{data_name}_{num_classes}_{method}_{threshold}_{input_size}_{transform}_{model}_{optimizer}_{criterion}_{max_epochs}{ext}"
        return file_name
    else:
        file_name = f"{data_name}_{num_classes}_{method}_{threshold}_{input_size}_{model}_{optimizer}_{criterion}_{max_epochs}{ext}"
        return file_name


def get_train_csvfile(num_classes, method, threshold, augmenter, transform) -> str:
    filename = f"train_un"
    # if num_classes is not None:
    if isinstance(num_classes, numbers.Number):
        filename += f"_{num_classes}_num"
        if method:
            filename += f"_{method}_{threshold}"
        if augmenter:
            filename += f"_{augmenter.lower()}"
        if transform:
            filename += f"_{transform.upper()}"
    return filename + ".csv"


def generate_files(template_name, lab_type, lab_point, data_name, num_classes, method, threshold, input_size, augmenter,
                   transform, model, optimizer,
                   criterion, max_epochs, target_names):
    template = Template(open(template_name).read().replace("$", "$$"))
    os.makedirs(template_name.split(".")[0], exist_ok=True)
    file_name = get_file_name(data_name, num_classes, method, threshold, input_size, augmenter, transform, model,
                              optimizer, criterion, max_epochs, '.yaml')
    print(file_name)
    replacements = {
        "$$lab_type": lab_type,
        "$$lab_point": lab_point,
        "$$data_name": data_name.upper(),
        "$$model_name": model.upper(),
        "$$optimizer_name": optimizer,
        "$$criterion_name": criterion,
        "$$num_classes": num_classes,
        "$$method": method,
        "$$threshold": threshold,
        "$$augmenter": augmenter,
        "$$transform": transform,
        "$$input_size": input_size,
        "$$max_epochs": max_epochs,
        # '$tags$': ["kdd99", "simple_dense_net"],
        '$$name': file_name.rsplit(".", 1)[0],
        '$$target_names': target_names
    }

    with open(os.path.join(template_name.split(".")[0], file_name), "w") as f:
        rendered_template = template.substitute(replacements)
        text = rendered_template
        for var, val in replacements.items():
            text = text.replace(var, str(val))
        f.write(text)


def generate_data_config(data_name, num_class, method, threshold, augmenter, transform):
    if data_name.upper() == "UNSW_NB15_10":
        print(f"load {data_name.upper()}")
        UNSW_NB15_10(root=f'../../../data', download=True, processed=True, num_classes=num_class, method=method,
                     threshold=threshold, augmenter=augmenter, transform=transform)
    if data_name.upper() == "NSL_KDD":
        print(f"load {data_name.upper()}")
        NSL_KDD(root=f'../../../data', download=True, processed=True, num_classes=num_class, method=method,
                threshold=threshold, augmenter=augmenter, transform=transform)
    data = CSVDataset(root=f'../../../data', data_name=data_name.upper(), name="train", num_classes=num_class,
                      method=method, threshold=threshold, augmenter=augmenter, transform=transform)
    return data.get_features_nums()


def execute_configs(configs):
    for config in configs:
        lab_type, lab_point, data_name, num_class, method, threshold, augmenter, transform, model, optimizer, criterion, max_epoch = config
        template_name = f"configs_example.yaml"
        print(template_name)
        input_size = generate_data_config(data_name, num_class, method, threshold, augmenter, transform)
        if data_name == "unsw_nb15_10":
            target_names = unsw_nb15_10_label_types.get_labels(num_class)
        if data_name == "nsl_kdd":
            target_names = nsl_kdd_label_types.get_labels(num_class)

        generate_files(template_name, lab_type, lab_point, data_name, num_class, method, threshold, input_size,
                       augmenter, transform,
                       model, optimizer, criterion, max_epoch, target_names)


def main():
    thresholds_config_values = {
        "lab_types": ['dl'],
        "lab_points": ['thresholds'],
        "data_names": ['nsl_kdd'],
        "num_classes": [2],
        "methods": ['xgboost'],
        "thresholds": [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005],
        "augmenters": [""],
        "transforms": [""],
        "models": ['mix_lstm'],
        "optimizers": ['Adam'],
        "criterions": ['WeightClassBalancedLoss'],
        "max_epochs": ['30']
    }

    models_config_values = {
        "lab_types": ['dl'],
        "lab_points": ['models'],
        "data_names": ['nsl_kdd'],
        "num_classes": [2],
        "methods": ['xgboost'],
        "thresholds": [0.003],
        "augmenters": [""],
        "transforms": [""],
        "models": ['mix_lstm', 'mix_cnn_lstm', 'cnn_lstm', 'cnn', 'lstm', 'mlp'],
        "optimizers": ['Adam'],
        "criterions": ['WeightClassBalancedLoss'],
        "max_epochs": ['30']
    }

    criterions_config_values = {
        "lab_types": ['dl'],
        "lab_points": ['criterions'],
        "data_names": ['nsl_kdd'],
        "num_classes": [2],
        "methods": ['xgboost'],
        "thresholds": [0.003],
        "augmenters": [""],
        "transforms": [""],
        "models": ['mix_lstm'],
        "optimizers": ['Adam'],
        "criterions": ['CrossEntropyLoss', 'ClassBalancedLoss', 'FocalLoss', 'MultiDiceLoss', 'MultiTverskyLoss',
                       'WeightClassBalancedLoss'],
        "max_epochs": ['30']
    }

    test_config_values = {
        "lab_types": ['dl'],
        "lab_points": ['balance_test'],
        "data_names": ['nsl_kdd'],
        "num_classes": [2],
        "methods": ['xgboost'],
        "thresholds": [0.002],
        "augmenters": [""],
        "transforms": [""],
        # "models": ['mix_lstm'],
        "models": ['mix_lstm'],
        "optimizers": ['Adam'],
        "criterions": ['WeightClassBalancedLoss'],
        # "criterions": ['ClassBalancedLoss'],
        # "max_epochs": ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        # "max_epochs": ['10', '15', '20', '25', '30']
        "max_epochs": ['20', '25']
    }

    best_config_values = {
        "lab_types": ['dl'],
        "lab_points": ['best'],
        "data_names": ['unsw_nb15_10'],
        # "data_names": ['nsl_kdd'],
        "num_classes": [2],
        "methods": ['xgboost'],
        "thresholds": [0.002],
        "augmenters": [""],
        "transforms": [""],
        # "models": ['mix_lstm'],
        "models": ['mix_lstm'],
        "optimizers": ['Adam'],
        "criterions": ['WeightClassBalancedLoss'],
        # "criterions": ['ClassBalancedLoss'],
        # "max_epochs": ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        # "max_epochs": ['10', '15', '20', '25', '30']
        "max_epochs": ['10']
    }

    # for config_values in [thresholds_config_values, criterions_config_values, models_config_values]:
    for config_values in [best_config_values]:
        all_configs = product(*config_values.values())
        execute_configs(all_configs)


if __name__ == '__main__':
    main()
