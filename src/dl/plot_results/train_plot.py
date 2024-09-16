# # -*- coding: utf-8 -*-
# # @Author: zhenwan
# # @Time: 10/15/2023 5:35 PM
# # @Last Modified by: zhenwan
# # @Last Modified time: 10/15/2023  5:35 PM
# # @file_name: plot_results.
# # @IDE: PyCharm
# # @copyright: zhenwan
# # train_plot.py
# import os
# import json
#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# def extract_data_from_folder(data_dir):
#     """
#     Extract training loss data from the specified directory.
#     :param data_dir: path to the directory containing thresholds data
#     :return: DataFrame containing train_loss for each threshold
#     """
#     thresholds = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
#
#     data = {}
#
#     for threshold in thresholds:
#         with open(os.path.join(data_dir, threshold, 'train', 'training_records.json'), 'r') as f:
#             records = json.load(f)
#             data[threshold] = [record['train_loss'] for record in records]
#
#     return pd.DataFrame(data)
#
#
# def extract_val_data_from_folder(data_dir):
#     """
#     Extract validation loss data from the specified directory.
#     :param data_dir: path to the directory containing thresholds data
#     :return: DataFrame containing val_loss for each threshold
#     """
#     thresholds = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
#
#     data = {}
#
#     for threshold in thresholds:
#         with open(os.path.join(data_dir, threshold, 'train', 'training_records.json'), 'r') as f:
#             records = json.load(f)
#             data[threshold] = [record['val_loss'] for record in records]
#
#     return pd.DataFrame(data)
#
#
# def extract_train_acc_from_folder(data_dir):
#     """
#     Extract training accuracy data from the specified directory.
#     :param data_dir: path to the directory containing thresholds data
#     :return: DataFrame containing train_accuracy for each threshold
#     """
#     thresholds = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
#
#     data = {}
#
#     for threshold in thresholds:
#         with open(os.path.join(data_dir, threshold, 'train', 'training_records.json'), 'r') as f:
#             records = json.load(f)
#             data[threshold] = [record['train_accuracy'] for record in records]
#
#     return pd.DataFrame(data)
#
#
# def extract_val_acc_from_folder(data_dir):
#     """
#     Extract validation accuracy data from the specified directory.
#     :param data_dir: path to the directory containing thresholds data
#     :return: DataFrame containing val_accuracy for each threshold
#     """
#     folder = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
#
#     data = {}
#
#     for threshold in folder:
#         with open(os.path.join(data_dir, threshold, 'train', 'training_records.json'), 'r') as f:
#             records = json.load(f)
#             data[threshold] = [record['val_accuracy'] for record in records]
#
#     return pd.DataFrame(data)
#
#
# def extract_accuracy_data_from_folder(data_dir, key):
#     """
#     Extract accuracy data (train or validation) from the specified directory.
#     :param data_dir: path to the directory containing folder data
#     :param key: the key in the json file for the accuracy ('train_accuracy' or 'val_accuracy')
#     :return: DataFrame containing accuracy for each threshold
#     """
#     folder = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
#
#     data = {}
#
#     for threshold in folder:
#         with open(os.path.join(data_dir, threshold, 'train', 'training_records.json'), 'r') as f:
#             records = json.load(f)
#             data[threshold] = [record[key] for record in records]
#
#     return pd.DataFrame(data)
#
#
# def plot_accuracy(df, save_path=None):
#     """
#     Plot accuracy from the provided DataFrame.
#     :param df: DataFrame containing accuracy for each threshold
#     :param save_path: path to save the plot, if None, just shows the plot
#     """
#     plt.figure(figsize=(6, 6 * 0.618))
#     for col in df.columns:
#         sns.lineplot(data=df[col], label=f"Threshold: {col.split('_')[4]}, Features: {col.split('_')[5]}")
#
#     plt.title(f'Accuracy for Different {folder}')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(title='Threshold', loc='lower right')
#
#     # Adjust the y-axis limit
#     min_value = df.min().min()
#     plt.ylim(min_value - 4, 95)  # adjust the y-axis limit based on the smallest value
#
#     if save_path:
#         plt.savefig(save_path, dpi=500)
#     else:
#         plt.show()
#
#
# def plot_val_loss(df, save_path=None):
#     """
#     Plot validation loss from the provided DataFrame.
#     :param df: DataFrame containing val_loss for each threshold
#     :param save_path: path to save the plot, if None, just shows the plot
#     """
#     # plt.figure(figsize=(10, 6))
#     plt.figure(figsize=(6, 6 * 0.618))
#     for col in df.columns:
#         sns.lineplot(data=df[col], label=f"Threshold: {col.split('_')[4]}, Features: {col.split('_')[5]}")
#     plt.title(f'Validation Loss for Different {folder}')
#     plt.xlabel('Epochs')
#     plt.ylabel('Validation Loss')
#     plt.legend(title='Threshold', loc='upper right')
#
#     if save_path:
#         plt.savefig(save_path, dpi=500)
#     else:
#         plt.show()
#
#
# def plot_training_loss(df, save_path=None):
#     """
#     Plot training loss from the provided DataFrame.
#     :param df: DataFrame containing train_loss for each threshold
#     :param save_path: path to save the plot, if None, just shows the plot
#     """
#     # plt.figure(figsize=(10, 6))
#     plt.figure(figsize=(6, 6 * 0.618))
#     for col in df.columns:
#         sns.lineplot(data=df[col], label=f"Threshold: {col.split('_')[4]}, Features: {col.split('_')[5]}")
#     plt.title(f'Training Loss for Different {folder}')
#     plt.xlabel('Epochs')
#     plt.ylabel('Train Loss')
#     plt.legend(title='Threshold', loc='upper right')
#
#     if save_path:
#         plt.savefig(save_path, dpi=500)
#     else:
#         plt.show()
#
#
# def ensure_directory_exists(directory_path):
#     """
#     Ensure that the specified directory exists. If not, create it.
#     :param directory_path: path of the directory
#     """
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path)
#
#
# # Using the new functions in the main section
# if __name__ == "__main__":
#     for folder in ['thresholds', 'models', 'criterions']:
#         DATA_DIR = f'C:\\Users\\zhenwan\\Desktop\\graduate\\paperonenew\\new\\results\\dl\\UNSW_NB15_10\\{folder}'
#         OUTPUT_DIR = f'C:\\Users\\zhenwan\\Desktop\\graduate\\paperonenew\\new\\result_plots\\dl\\UNSW_NB15_10\\{folder}\\train'
#         # OUTPUT_DIR = '/src/dl/result_plots/dl/UNSW_NB15_10/thresholds\\train'
#
#         ensure_directory_exists(OUTPUT_DIR)
#
#         # Train Loss Plot
#         train_df = extract_data_from_folder(DATA_DIR)
#         plot_training_loss(train_df, save_path=os.path.join(OUTPUT_DIR, f"train_loss_{folder}.png"))
#
#         # Validation Loss Plot
#         val_df = extract_val_data_from_folder(DATA_DIR)
#         plot_val_loss(val_df, save_path=os.path.join(OUTPUT_DIR, f"val_loss_{folder}.png"))
#
#         # Train Accuracy Plot
#         train_acc_df = extract_accuracy_data_from_folder(DATA_DIR, 'train_accuracy')
#         plot_accuracy(train_acc_df, save_path=os.path.join(OUTPUT_DIR, f"train_accuracy_{folder}.png"))
#
#         # Validation Accuracy Plot
#         val_acc_df = extract_accuracy_data_from_folder(DATA_DIR, 'val_accuracy')
#         plot_accuracy(val_acc_df, save_path=os.path.join(OUTPUT_DIR, f"val_accuracy_{folder}.png"))

# import os
# import json
# import re
#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# def extract_data_from_folder(data_dir, key):
#     """
#     Extract data (train_loss, val_loss, train_accuracy, val_accuracy) from the specified directory based on key.
#     :param data_dir: path to the directory containing thresholds data
#     :param key: the key in the json file for the desired metric
#     :return: DataFrame containing data for each threshold
#     """
#     thresholds = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
#
#     data = {}
#     for threshold in thresholds:
#         with open(os.path.join(data_dir, threshold, 'train', 'training_records.json'), 'r') as f:
#             records = json.load(f)
#             data[threshold] = [record[key] for record in records]
#
#     return pd.DataFrame(data)
#
#
# def plot_metrics(df, folder, y_label, title, save_path=None):
#     """
#     Plot specified metric (loss/accuracy) from the provided DataFrame.
#     :param df: DataFrame containing metric data for each threshold
#     :param y_label: Y-axis label (e.g., "Train Loss", "Accuracy")
#     :param title: Title of the plot
#     :param save_path: path to save the plot, if None, just shows the plot
#     """
#     global label
#     plt.figure(figsize=(6, 6 * 0.618))
#     for col in df.columns:
#         pattern = r"(\w+)_NB15_(\d+)_(\d+)_(\d+\.\d+)_(\d+)_([A-Z_]+)_(\w+)_(\w+)_(\d+)"
#
#         match = re.match(pattern, col)
#
#         if match:
#             dataset_name = match.group(1) + "_NB15" + match.group(2)
#             num_classes = match.group(3)
#             thresholds = match.group(4)
#             features_num = match.group(5)
#             model_name = match.group(6)
#             optimizer_name = match.group(8)
#             criterions_name = match.group(8)
#             mix_epoch = match.group(9)
#
#         # suffix = "acc" if "Accuracy" in y_label else "loss"
#         if folder == "thresholds":
#             label = f"Threshold: {thresholds}, Features: {features_num}"
#         if folder == "models":
#             label = f"Models: {model_name}, Features: {features_num}"
#         if folder == "criterions":
#             label = f"Criterions: {criterions_name}, Features: {features_num}"
#
#         sns.lineplot(data=df[col], label=label)
#     plt.title(title)
#     plt.xlabel('Epochs')
#     plt.ylabel(y_label)
#     plt.legend(title=f'{folder}', loc='upper right' if "Loss" in y_label else 'lower right')
#
#     min_value = df.min().min()
#     max_value = df.max().max()
#     mix_value = max_value - min_value
#     if "Accuracy" in y_label:
#         plt.ylim(min_value - 0.8 * mix_value, max_value + 0.2 * mix_value)
#
#     if "Loss" in y_label:
#         plt.ylim(min_value - 0.2 * mix_value, max_value + 0.8 * mix_value)
#
#     if save_path:
#         plt.savefig(save_path, dpi=500)
#     else:
#         plt.show()
#
#
# def ensure_directory_exists(directory_path):
#     """
#     Ensure that the specified directory exists. If not, create it.
#     :param directory_path: path of the directory
#     """
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path)
#
#
# if __name__ == "__main__":
#     BASE_DIR = 'C:\\Users\\zhenwan\\Desktop\\graduate\\paperonenew\\new'
#
#     for folder in ['thresholds', 'models', 'criterions']:
#         DATA_DIR = os.path.join(BASE_DIR, 'results', 'dl', 'UNSW_NB15_10', folder)
#         OUTPUT_DIR = os.path.join(BASE_DIR, 'result_plots', 'dl', 'UNSW_NB15_10', folder, 'train')
#
#         ensure_directory_exists(OUTPUT_DIR)
#
#         metrics = [
#             ('train_loss', 'Train Loss'),
#             ('val_loss', 'Validation Loss'),
#             ('train_accuracy', 'Accuracy'),
#             ('val_accuracy', 'Accuracy')
#         ]
#
#         for key, label in metrics:
#             df = extract_data_from_folder(DATA_DIR, key)
#             plot_title = f'{label} for Different {folder.title()} on {key}'
#             save_path = os.path.join(OUTPUT_DIR, f"{key}_{folder}.png")
#             plot_metrics(df, folder, label, plot_title, save_path)

# trian_plot.py
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from src.dl.plot_results.ext import extract_info_from_column


def extract_data_from_folder(data_dir, key):
    data_dir = Path(data_dir)
    thresholds = [name for name in data_dir.iterdir() if name.is_dir()]

    data = {}
    for threshold in thresholds:
        with open(threshold / 'train' / 'training_records.json', 'r') as f:
            records = json.load(f)
            data[str(threshold.name)] = [record[key] for record in records]

    return pd.DataFrame(data)





def plot_metrics(dataset, df, folder, y_label, title, save_path=None):
    global label, info
    plt.figure(figsize=(6, 6 * 0.618))
    line_styles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
    palette = sns.color_palette("husl", len(df.columns))  # Choose a different palette if needed

    for style_index, col in enumerate(df.columns):
        info = extract_info_from_column(dataset, col)

        if folder == "thresholds":
            label = f"Threshold: {info['thresholds']}, Features: {info['features_num']}"
        elif folder == "models":
            label = f"Models: {info['model_name']}, Features: {info['features_num']}"
        elif folder == "criterions":
            label = f"Criterions: {info['criterions_name']}, Features: {info['features_num']}"

        # sns.lineplot(data=df[col], label=label)
        sns.lineplot(data=df[col], label=label, linestyle=line_styles[style_index], palette=[palette[style_index]])

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend(loc='upper right' if "Loss" in y_label else 'lower right')
    # plt.legend(title=f'{folder}', loc='upper right' if "Loss" in y_label else 'lower right')

    min_value, max_value = df.values.min(), df.values.max()
    gap = max_value - min_value
    plt.ylim(min_value - 0.2 * gap if "Loss" in y_label else min_value - 0.8 * gap,
             max_value + 0.8 * gap if "Loss" in y_label else max_value + 0.2 * gap)
    # plt.xlim()
    # print(int(info['mix_epoch']))
    bit = int(int(info['mix_epoch'])/14+1)
    print(range(int(info['mix_epoch']))[::bit])
    plt.xticks(range(int(info['mix_epoch']))[::bit], labels=range(1, len(range(int(info['mix_epoch'])+1)))[::bit])
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    BASE_DIR = Path('/home/tyxk/Desktop/ZhenWan/new')
    for dataset in ['UNSW_NB15_10', 'NSL_KDD']:
        for folder in ['thresholds', 'models', 'criterions']:
            DATA_DIR = BASE_DIR / 'results' / 'dl' / dataset / folder
            OUTPUT_DIR = BASE_DIR / 'result_plots' / 'dl' / dataset / folder / 'train'

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            metrics = [
                ('train_loss', 'Train Loss'),
                ('val_loss', 'Validation Loss'),
                ('train_accuracy', 'Accuracy'),
                ('val_accuracy', 'Accuracy')
            ]

            for key, label in metrics:
                df = extract_data_from_folder(DATA_DIR, key)
                plot_title = f'{label} for Different {folder.title()} on {key}'
                save_path = OUTPUT_DIR / f"{key}_{folder}.png"
                plot_metrics(dataset, df, folder, label, plot_title, save_path)
