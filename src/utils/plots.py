# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 9/19/2023 6:49 PM
# @Last Modified by: zhenwan
# @Last Modified time: 9/19/2023  6:49 PM
# @file_name: plots.
# @IDE: PyCharm
# @copyright: zhenwan
import json
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix


# def plot_training(file_name):
#     # 读取JSON文件并提取数据
#     with open(file_name, 'r') as file:
#         training_records = json.load(file)
#
#     # 提取epoch、train_loss、val_loss、train_accuracy和val_accuracy数据
#     epochs = [record['epoch'] for record in training_records]
#     train_losses = [record['train_loss'] for record in training_records]
#     val_losses = [record['val_loss'] for record in training_records]
#     train_accuracies = [record['train_accuracy'] for record in training_records]
#     val_accuracies = [record['val_accuracy'] for record in training_records]
#
#     min_train_loss_value = min(train_losses)
#     min_val_loss_value = min(val_losses)
#     max_train_acc_value = max(train_accuracies)
#     max_val_acc_value = max(val_accuracies)
#
#     # 找到最低Train Loss和最低Validation Loss的epoch
#     min_train_loss_epoch = epochs[train_losses.index(min_train_loss_value)]
#     min_val_loss_epoch = epochs[val_losses.index(min_val_loss_value)]
#
#     # 找到最高Train Accuracy和最高Validation Accuracy的epoch
#     max_train_acc_epoch = epochs[train_accuracies.index(max_train_acc_value)]
#     max_val_acc_epoch = epochs[val_accuracies.index(max_val_acc_value)]
#
#     # 绘制Loss图形
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_losses, marker='o', linestyle='-', label='Train Loss', markersize=1)
#     plt.plot(epochs, val_losses, marker='o', linestyle='-', label='Validation Loss', markersize=1)
#     plt.scatter(min_train_loss_epoch, min(train_losses), color='red', marker='o', label='Min Train Loss', s=100)
#     plt.scatter(min_val_loss_epoch, min(val_losses), color='blue', marker='o', label='Min Validation Loss', s=100)
#     # 在最低Train Loss和Validation Loss处添加文本标签
#     plt.annotate(f'Train: {min_train_loss_value:.2f}', xy=(min_train_loss_epoch, min_train_loss_value),
#                  xytext=(min_train_loss_epoch - 5, min_train_loss_value + 0.03), textcoords='data',
#                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.5'),
#                  color='red')  # 将文字颜色设置为红色
#     plt.annotate(f'Validation: {min_val_loss_value:.2f}', xy=(min_val_loss_epoch, min(val_losses)),
#                  xytext=(min_val_loss_epoch - 5, min(val_losses) + 0.03), textcoords='data',
#                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.5'),
#                  color='blue')  # 将文字颜色设置为蓝色
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#
#     # 绘制Accuracy图形
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, train_accuracies, marker='o', linestyle='-', label='Train Accuracy', markersize=1)
#     plt.plot(epochs, val_accuracies, marker='o', linestyle='-', label='Validation Accuracy', markersize=1)
#     plt.scatter(max_train_acc_epoch, max(train_accuracies), color='red', marker='o', label='Max Train Accuracy', s=100)
#     plt.scatter(max_val_acc_epoch, max(val_accuracies), color='blue', marker='o', label='Max Validation Accuracy',
#                 s=100)
#     # 在最高Train Accuracy和Validation Accuracy处添加文本标签
#     plt.annotate(f'Train: {max_train_acc_value:.2f}%', xy=(max_train_acc_epoch, max(train_accuracies)),
#                  xytext=(max_train_acc_epoch - 5, max(train_accuracies) + 2), textcoords='data',
#                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.5'),
#                  color='red')
#     plt.annotate(f'Validation: {max_val_acc_value:.2f}%', xy=(max_val_acc_epoch, max(val_accuracies)),
#                  xytext=(max_val_acc_epoch - 5, max(val_accuracies) + 2), textcoords='data',
#                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.5'),
#                  color='blue')
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.savefig(f'{os.path.dirname(file_name)}/Training and Validation.png', dpi=500)
#     plt.show()


def plot_testing(file_name):
    # 读取JSON文件并提取数据
    with open(file_name, 'r') as file:
        training_records = json.load(file)

    model_name = training_records['model_name']
    target_names = training_records['target_names']
    test_directory = training_records['test_directory']
    y_pred = np.array(training_records['y_pred'])
    y_pred_pro = np.array(training_records['y_pred_pro'])
    y_test = np.array(training_records['y_test'])
    # print(f"y_test type: {type(y_test)}")
    # print(f"y_pred type: {type(y_pred)}")
    # print(f"y_pred_pro type: {type(y_pred_pro)}")

    plot_PR_curve(model_name, y_test, y_pred_pro, target_names, test_directory)
    plot_ROC_curve(model_name, y_test, y_pred_pro, target_names, test_directory)
    plot_confusion_matrix(model_name, y_test, y_pred, target_names, test_directory)


def plot_PR_curve(model_name, y_test, y_pred_pro, target_names, dataset_folder):
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(len(target_names)):
        precision[i], recall[i], _ = precision_recall_curve((y_test == i).astype(int), y_pred_pro[:, i])
        average_precision[i] = auc(recall[i], precision[i])
        plt.plot(recall[i], precision[i], marker='.', label=f'{target_names[i]} (AP = {average_precision[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(dataset_folder, f'{model_name}_pr.png'), dpi=500)
    plt.show()


def plot_ROC_curve(model_name, y_test, y_pred_pro, target_names, dataset_folder):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(target_names)):
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred_pro[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], marker='.', label=f'{target_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(dataset_folder, f'{model_name}_roc.png'), dpi=500)
    plt.show()


def plot_confusion_matrix(model_name, y_test, y_pred, target_names, dataset_folder):
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(target_names)), target_names)
    plt.yticks(np.arange(len(target_names)), target_names)

    for i in range(len(target_names)):
        for j in range(len(target_names)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(dataset_folder, f'{model_name}_cm.png'), dpi=500)
    plt.show()


def plot_PR_curve_from_json(json_file_path):
    # Load the data from JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    model_name = data['model_name']
    target_names = data['target_names']
    y_test = np.array(data['y_test'])
    y_pred_pro = np.array(data['y_pred_pro'])
    dataset_folder = data['test_directory']

    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(len(target_names)):
        precision[i], recall[i], _ = precision_recall_curve((y_test == i).astype(int), y_pred_pro[:, i])
        average_precision[i] = auc(recall[i], precision[i])
        plt.plot(recall[i], precision[i], marker='.', label=f'{target_names[i]} (AP = {average_precision[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(dataset_folder, f'{model_name}_pr.png'), dpi=500)
    plt.show()


def plot_ROC_curve_from_json(json_file_path):
    # Load the data from JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    model_name = data['model_name']
    target_names = data['target_names']
    y_test = np.array(data['y_test'])
    y_pred_pro = np.array(data['y_pred_pro'])
    dataset_folder = data['test_directory']

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(target_names)):
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred_pro[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], marker='.', label=f'{target_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(dataset_folder, f'{model_name}_roc.png'), dpi=500)
    plt.show()


def plot_confusion_matrix_from_json(json_file_path):
    # Load the data from JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    model_name = data['model_name']
    target_names = data['target_names']
    y_test = np.array(data['y_test'])
    y_pred = np.array(data['y_pred'])
    dataset_folder = data['test_directory']

    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(target_names)), target_names, rotation=45)
    plt.yticks(np.arange(len(target_names)), target_names)

    for i in range(len(target_names)):
        for j in range(len(target_names)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_folder, f'{model_name}_cm.png'), dpi=500)
    plt.show()


def plot_training(file_name):
    # 读取JSON文件并提取数据
    with open(file_name, 'r') as file:
        training_records = json.load(file)

    # 提取epoch、train_loss、val_loss、train_accuracy和val_accuracy数据
    epochs = [record['epoch'] for record in training_records]
    train_losses = [record['train_loss'] for record in training_records]
    val_losses = [record['val_loss'] for record in training_records]
    train_accuracies = [record['train_accuracy'] for record in training_records]
    val_accuracies = [record['val_accuracy'] for record in training_records]

    min_train_loss_value = min(train_losses)
    min_val_loss_value = min(val_losses)
    max_train_acc_value = max(train_accuracies)
    max_val_acc_value = max(val_accuracies)

    # 找到最低Train Loss和最低Validation Loss的epoch
    min_train_loss_epoch = epochs[train_losses.index(min_train_loss_value)]
    min_val_loss_epoch = epochs[val_losses.index(min_val_loss_value)]

    # 找到最高Train Accuracy和最高Validation Accuracy的epoch
    max_train_acc_epoch = epochs[train_accuracies.index(max_train_acc_value)]
    max_val_acc_epoch = epochs[val_accuracies.index(max_val_acc_value)]

    # 设置更具辨识度的颜色
    colors = plt.cm.get_cmap("tab20c", len(epochs))

    # 绘制Loss图形
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    for i, epoch in enumerate(epochs):
        label = f'Epoch {epoch}'
        plt.plot(epoch, train_losses[i], marker='o', markersize=4, linestyle='-', label=label, color=colors(i))
    plt.scatter(min_train_loss_epoch, min_train_loss_value, color='red', marker='o', s=100, label='Min Train Loss')
    plt.scatter(min_val_loss_epoch, min_val_loss_value, color='blue', marker='o', s=100, label='Min Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制Accuracy图形
    plt.subplot(1, 2, 2)
    for i, epoch in enumerate(epochs):
        label = f'Epoch {epoch}'
        plt.plot(epoch, train_accuracies[i], marker='o', markersize=4, linestyle='-', label=label, color=colors(i))
    plt.scatter(max_train_acc_epoch, max_train_acc_value, color='red', marker='o', s=100, label='Max Train Accuracy')
    plt.scatter(max_val_acc_epoch, max_val_acc_value, color='blue', marker='o', s=100, label='Max Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()

    # 保存和显示图形
    save_path = os.path.join(os.path.dirname(file_name), 'Training_and_Validation.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
