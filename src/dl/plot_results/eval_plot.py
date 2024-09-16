# # -*- coding: utf-8 -*-
# # @Author: zhenwan
# # @Time: 10/15/2023 6:22 PM
# # @Last Modified by: zhenwan
# # @Last Modified time: 10/15/2023  6:22 PM
# # @file_name: test_plot.
# # @IDE: PyCharm
# # @copyright: zhenwan
# # eval_plot.py
# import os
# import json
# import re
# from pathlib import Path
#
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.metrics import precision_recall_curve, average_precision_score
#
# from sklearn.metrics import roc_curve, roc_auc_score
#
# def extract_info_from_column(column):
#     pattern = r"(\w+)_NB15_(\d+)_(\d+)_(\d+\.\d+)_(\d+)_([A-Z_]+)_(\w+)_(\w+)_(\d+)"
#     match = re.match(pattern, column)
#
#     if match:
#         return {
#             'thresholds': match.group(4),
#             'features_num': match.group(5)
#         }
#     return {'thresholds': '', 'features_num': ''}
#
#
# def extract_data_from_folder(data_dir, key):
#     data_dir = Path(data_dir)
#     thresholds = [name for name in data_dir.iterdir() if name.is_dir()]
#
#     data = {}
#     for threshold in thresholds:
#         with open(threshold / 'train' / 'training_records.json', 'r') as f:
#             records = json.load(f)
#             data[str(threshold.name)] = [record[key] for record in records]
#
#     return pd.DataFrame(data)
#
# def extract_roc_data_from_thresholds(data_dir):
#     thresholds = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
#     auc_scores = {}
#     roc_data = {}
#
#     for threshold in thresholds:
#         with open(os.path.join(data_dir, threshold, 'test', 'testing_records.json'), 'r') as f:
#             records = json.load(f)
#             y_test = records['y_test']
#             y_scores = [proba[1] for proba in records['y_pred_pro']]  # Assuming binary classification
#
#             fpr, tpr, _ = roc_curve(y_test, y_scores)
#             roc_data[threshold] = {'fpr': fpr, 'tpr': tpr}
#
#             auc = roc_auc_score(y_test, y_scores)
#             auc_scores[threshold] = auc
#
#     return roc_data, auc_scores
#
# def plot_roc_curve(data, auc_scores, save_path=None):
#     plt.figure(figsize=(6, 6*0.618))
#     for threshold, roc_values in data.items():
#         info = extract_info_from_column(threshold)
#         fpr = roc_values['fpr']
#         tpr = roc_values['tpr']
#         label = f"Threshold: {info['thresholds']}, Features: {info['features_num']}, AUC: {auc_scores[threshold]:.3f}"
#         plt.plot(fpr, tpr, label=label)
#
#     plt.title('ROC Curve for Different Thresholds')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend(loc='lower right')
#     plt.grid(True)
#
#     if save_path:
#         plt.savefig(save_path, dpi=500)
#     else:
#         plt.show()
#
#
# def extract_pr_data_from_thresholds(data_dir):
#     """
#     Extract precision and recall data for PR curve from the specified directory.
#     :param data_dir: path to the directory containing thresholds data
#     :return: Dictionary containing precision and recall for each threshold
#     """
#     thresholds = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
#     ap_scores = {}
#     pr_data = {}
#
#     for threshold in thresholds:
#         with open(os.path.join(data_dir, threshold, 'test', 'testing_records.json'), 'r') as f:
#             records = json.load(f)
#             y_test = records['y_test']
#             y_scores = [proba[1] for proba in
#                         records['y_pred_pro']]  # Assuming binary classification, taking score of class 1
#             precision, recall, _ = precision_recall_curve(y_test, y_scores)
#             pr_data[threshold] = {'precision': precision, 'recall': recall}
#
#             ap = average_precision_score(y_test, y_scores)
#             ap_scores[threshold] = ap
#
#     return pr_data, ap_scores
#
#
# def plot_pr_curve(pr_data, ap_scores, save_path=None):
#     plt.figure(figsize=(6, 6*0.618))
#     for threshold, pr_values in pr_data.items():
#         info = extract_info_from_column(threshold)
#         precision = pr_values['precision']
#         recall = pr_values['recall']
#         label = f"Threshold: {info['thresholds']}, Features: {info['features_num']}, AP: {ap_scores[threshold]:.3f}"
#         plt.plot(recall, precision, label=label)
#
#     plt.title('Precision-Recall Curve for Different Thresholds')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.legend(loc='lower left')
#
#     if save_path:
#         plt.savefig(save_path, dpi=500)
#     else:
#         plt.show()
#
#
# if __name__ == "__main__":
#     BASE_DIR = Path('C:\\Users\\zhenwan\\Desktop\\graduate\\paperonenew\\new')
#
#     for folder in ['thresholds', 'models', 'criterions']:
#         DATA_DIR = BASE_DIR / 'results' / 'dl' / 'UNSW_NB15_10' / folder
#         OUTPUT_DIR = BASE_DIR / 'result_plots' / 'dl' / 'UNSW_NB15_10' / folder / 'train'
#
#         OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#
#         metrics = {
#             'pr_curve': {
#                 'extractor': extract_pr_data_from_thresholds,
#                 'plotter': plot_pr_curve,
#                 'filename': "pr_curve_thresholds.png",
#                 'title': 'Precision-Recall Curve'
#             },
#             'roc_curve': {
#                 'extractor': extract_roc_data_from_thresholds,
#                 'plotter': plot_roc_curve,
#                 'filename': "roc_curve_thresholds.png",
#                 'title': 'ROC Curve'
#             }
#         }
#
#         for key, meta in metrics.items():
#             data, scores = meta['extractor'](DATA_DIR)
#             output_path = OUTPUT_DIR / meta['filename']
#             meta['plotter'](data, scores, save_path=output_path)

# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 10/15/2023 6:22 PM
# @Last Modified by: zhenwan
# @Last Modified time: 10/15/2023  6:22 PM
# @file_name: test_plot.
# @IDE: PyCharm
# @copyright: zhenwan
#
# import os
# import json
# import re
# from pathlib import Path
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
#
# def extract_info_from_column(column):
#     pattern = r"(\w+)_NB15_(\d+)_(\d+)_(\d+\.\d+)_(\d+)_([A-Z_]+)_(\w+)_(\w+)_(\d+)"
#     match = re.match(pattern, column)
#
#     if match:
#         return {
#             'thresholds': match.group(4),
#             'features_num': match.group(5)
#         }
#     return {'thresholds': '', 'features_num': ''}
#
# def extract_roc_data_from_folder(data_dir):
#     folder = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
#     auc_scores = {}
#     roc_data = {}
#
#     for threshold in folder:
#         with open(os.path.join(data_dir, threshold, 'test', 'testing_records.json'), 'r') as f:
#             records = json.load(f)
#             y_test = records['y_test']
#             y_scores = [proba[1] for proba in records['y_pred_pro']]  # Assuming binary classification
#
#             fpr, tpr, _ = roc_curve(y_test, y_scores)
#             roc_data[threshold] = {'fpr': fpr, 'tpr': tpr}
#
#             auc = roc_auc_score(y_test, y_scores)
#             auc_scores[threshold] = auc
#
#     return roc_data, auc_scores
#
# def plot_roc_curve(data, auc_scores, save_path=None):
#     plt.figure(figsize=(6, 6*0.618))
#     for threshold, roc_values in data.items():
#         info = extract_info_from_column(threshold)
#         fpr = roc_values['fpr']
#         tpr = roc_values['tpr']
#         label = f"Threshold: {info['thresholds']}, Features: {info['features_num']}, AUC: {auc_scores[threshold]:.3f}"
#         plt.plot(fpr, tpr, label=label)
#
#     plt.title(f'ROC Curve for Different {folder}')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend(loc='lower right')
#     plt.grid(True)
#
#     if save_path:
#         plt.savefig(save_path, dpi=500)
#     else:
#         plt.show()
#
# def extract_pr_data_from_folder(data_dir):
#     folder = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
#     ap_scores = {}
#     pr_data = {}
#
#     for threshold in folder:
#         with open(os.path.join(data_dir, threshold, 'test', 'testing_records.json'), 'r') as f:
#             records = json.load(f)
#             y_test = records['y_test']
#             y_scores = [proba[1] for proba in records['y_pred_pro']]  # Assuming binary classification
#
#             precision, recall, _ = precision_recall_curve(y_test, y_scores)
#             pr_data[threshold] = {'precision': precision, 'recall': recall}
#
#             ap = average_precision_score(y_test, y_scores)
#             ap_scores[threshold] = ap
#
#     return pr_data, ap_scores
#
# def plot_pr_curve(pr_data, ap_scores, save_path=None):
#     plt.figure(figsize=(6, 6*0.618))
#     for threshold, pr_values in pr_data.items():
#         info = extract_info_from_column(threshold)
#         precision = pr_values['precision']
#         recall = pr_values['recall']
#         label = f"Threshold: {info['thresholds']}, Features: {info['features_num']}, AP: {ap_scores[threshold]:.3f}"
#         plt.plot(recall, precision, label=label)
#
#     plt.title(f'Precision-Recall Curve for Different {folder}')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.legend(loc='lower left')
#
#     if save_path:
#         plt.savefig(save_path, dpi=500)
#     else:
#         plt.show()
#
# if __name__ == "__main__":
#     BASE_DIR = Path('C:\\Users\\zhenwan\\Desktop\\graduate\\paperonenew\\new')
#
#     for folder in ['thresholds', 'models', 'criterions']:
#         DATA_DIR = BASE_DIR / 'results' / 'dl' / 'UNSW_NB15_10' / folder
#         OUTPUT_DIR = BASE_DIR / 'result_plots' / 'dl' / 'UNSW_NB15_10' / folder / 'test'
#         OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#
#         metrics = {
#             'pr_curve': {
#                 'extractor': extract_pr_data_from_folder,
#                 'plotter': plot_pr_curve,
#                 'filename': f"pr_curve_{folder}.png",
#                 'title': f'Precision-Recall Curve for Different {folder.title()} on test'
#             },
#             'roc_curve': {
#                 'extractor': extract_roc_data_from_folder,
#                 'plotter': plot_roc_curve,
#                 'filename': f"roc_curve_{folder}.png",
#                 'title': f'ROC Curve for Different {folder.title()} on test'
#             }
#         }
#
#         for key, meta in metrics.items():
#             data, scores = meta['extractor'](DATA_DIR)
#             output_path = OUTPUT_DIR / meta['filename']
#             meta['plotter'](data, scores, save_path=output_path)

# test_plot.py
import os
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score

from src.dl.plot_results.ext import extract_info_from_column


def extract_data_from_folder(data_dir, metric_type):
    folder_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    scores = {}
    data = {}

    for name in folder_names:
        with open(os.path.join(data_dir, name, 'test', 'testing_records.json'), 'r') as f:
            records = json.load(f)
            y_test = records['y_test']
            y_scores = [proba[1] for proba in records['y_pred_pro']]

            if metric_type == "roc":
                x, y, _ = roc_curve(y_test, y_scores)
                score = roc_auc_score(y_test, y_scores)
            elif metric_type == "pr":
                x, y, _ = precision_recall_curve(y_test, y_scores)
                score = average_precision_score(y_test, y_scores)
            else:
                raise ValueError("Invalid metric_type")

            data[name] = {'x': x, 'y': y}
            scores[name] = score
    print(scores)
    return data, scores

# def plot_curve(data, folder, scores, title_prefix, xlabel, ylabel, save_path=None):
#     global label
#     plt.figure(figsize=(6, 6*0.618))
#     for name, values in data.items():
#         info = extract_info_from_column(name)
#         if folder == "thresholds":
#             label = f"Threshold: {info['thresholds']}, Features: {info['features_num']}, Score: {scores[name]:.3f}"
#         elif folder == "models":
#             label = f"Models: {info['model_name']}, Features: {info['features_num']}, Score: {scores[name]:.3f}"
#         elif folder == "criterions":
#             label = f"Criterions: {info['criterions_name']}, Features: {info['features_num']}, Score: {scores[name]:.3f}"
#         plt.plot(values['x'], values['y'], label=label)
#
#     plt.title(f'{title_prefix} for Different {folder}')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend(loc='lower right')
#     plt.grid(True)
#
#     if save_path:
#         plt.savefig(save_path, dpi=300)
#     else:
#         plt.show()

import matplotlib.pyplot as plt
import seaborn as sns


def plot_curve(data, folder, scores, title_prefix, xlabel, ylabel, save_path=None):
    global label
    plt.figure(figsize=(6, 6 * 0.618))

    line_styles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
    palette = sns.color_palette("husl", len(data))  # Choose a different palette if needed

    # Define a function for the key argument in sorted
    def get_score(item):
        name, values = item
        return scores[name]

    # Sort data items by label
    sorted_data = sorted(data.items(), key=get_score, reverse=True)

    for style_index, (name, values) in enumerate(sorted_data):
        info = extract_info_from_column(dataset, name)
        if folder == "thresholds":
            label = f"Threshold: {info['thresholds']}, Features: {info['features_num']}, Score: {scores[name]:.3f}"
        elif folder == "models":
            label = f"Models: {info['model_name']}, Features: {info['features_num']}, Score: {scores[name]:.3f}"
        elif folder == "criterions":
            label = f"Criterions: {info['criterions_name']}, Features: {info['features_num']}, Score: {scores[name]:.3f}"

        plt.plot(values['x'], values['y'], label=label, linestyle=line_styles[style_index % len(line_styles)],
                 color=palette[style_index])

    plt.title(f'{title_prefix} for Different {folder}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()



if __name__ == "__main__":
    BASE_DIR = Path(r'F:\new')
    for dataset in ['UNSW_NB15_10', 'NSL_KDD']:
        for folder in ['thresholds', 'models', 'criterions']:
            DATA_DIR = BASE_DIR / 'results' / 'dl' / dataset / folder
            OUTPUT_DIR = BASE_DIR / 'result_plots' / 'dl' / dataset / folder / 'test'
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            metrics = {
                'pr_curve': {
                    'extractor': lambda dir: extract_data_from_folder(dir, "pr"),
                    'plotter': lambda data, scores, save_path: plot_curve(data, folder, scores, "Precision-Recall Curve", "Recall", "Precision", save_path),
                    'filename': f"pr_curve_{folder}.png",
                },
                'roc_curve': {
                    'extractor': lambda dir: extract_data_from_folder(dir, "roc"),
                    'plotter': lambda data, scores, save_path: plot_curve(data, folder, scores, "ROC Curve", "False Positive Rate", "True Positive Rate", save_path),
                    'filename': f"roc_curve_{folder}.png",
                }
            }

            for key, meta in metrics.items():
                data, scores = meta['extractor'](DATA_DIR)
                output_path = OUTPUT_DIR / meta['filename']
                meta['plotter'](data, scores, save_path=output_path)
