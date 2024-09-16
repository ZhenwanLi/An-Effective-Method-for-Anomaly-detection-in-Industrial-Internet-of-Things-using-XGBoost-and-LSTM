# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 9/19/2023 7:15 PM
# @Last Modified by: zhenwan
# @Last Modified time: 9/19/2023  7:15 PM
# @file_name: plot.
# @IDE: PyCharm
# @copyright: zhenwan
import json
import matplotlib.pyplot as plt

# 读取JSON文件并提取数据
with open('training_records.json', 'r') as file:
    training_records = json.load(file)

# 提取epoch、loss和accuracy数据
epochs = [record['epoch'] for record in training_records]
losses = [record['train_loss'] for record in training_records]
accuracies = [record['train_accuracy'] for record in training_records]

min_loss_value = min(losses)
max_acc_value = max(accuracies)
# 找到最低Loss和最高Accuracy的epoch
min_loss_epoch = epochs[losses.index(min_loss_value)]
max_acc_epoch = epochs[accuracies.index(max_acc_value)]

# 绘制Loss和Accuracy图形
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, marker='o', linestyle='-', label='Loss', markersize=1)
plt.scatter(min_loss_epoch, min(losses), color='red', marker='o', label='Min Loss', s=100)
# 在Loss最低值处添加文本标签（微调位置和改变文字颜色）
plt.annotate(f'{min_loss_value:.2f}', xy=(min_loss_epoch, min_loss_value),
             xytext=(min_loss_epoch - 5, min_loss_value + 0.03), textcoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.5'),
             color='red')  # 将文字颜色设置为红色
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracies, marker='o', linestyle='-', label='Accuracy', markersize=1)
plt.scatter(max_acc_epoch, max(accuracies), color='red', marker='o', label='Max Accuracy', s=100)
# 在Accuracy最大值处添加文本标签
plt.annotate(f'{max_acc_value:.2f}%', xy=(max_acc_epoch, max_acc_value),
             xytext=(max_acc_epoch - 5, max_acc_value + 2), textcoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.5'),
             color='red')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()

plt.show()