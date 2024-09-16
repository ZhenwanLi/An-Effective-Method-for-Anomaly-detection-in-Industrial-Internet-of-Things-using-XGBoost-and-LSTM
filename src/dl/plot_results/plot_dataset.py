# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 12/20/2023 4:35 PM
# @Last Modified by: zhenwan
# @Last Modified time: 12/20/2023  4:35 PM
# @file_name: plot_dataset.
# @IDE: PyCharm
# @copyright: zhenwan
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch.utils.data import random_split



def load_and_split_data(file_name, train_ratio=0.7):
    dataset = pd.read_csv(file_name)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_df = dataset.iloc[train_dataset.indices].reset_index(drop=True)
    val_df = dataset.iloc[val_dataset.indices].reset_index(drop=True)

    return train_df, val_df


def combine_datasets(train_df, val_df, test_df):
    train_df['dataset_type'] = 'train'
    val_df['dataset_type'] = 'val'
    test_df['dataset_type'] = 'test'
    return pd.concat([train_df, val_df, test_df])


def visualize_counts(combined_df):
    plt.figure(figsize=(15, 7))
    sns.countplot(data=combined_df, x=LABEL, hue='dataset_type')
    plt.title('Number of instances for each category in train, val, and test datasets')
    plt.show()


def apply_tsne(data, n_components=2, random_state=42):
    if os.path.exists(TSNE_FILE):
        tsne_results = np.load(TSNE_FILE)
    else:
        tsne = TSNE(n_components=n_components, random_state=random_state)
        tsne_results = tsne.fit_transform(data.drop([LABEL, 'dataset_type'], axis=1))
        np.save(TSNE_FILE, tsne_results)

    data['tsne-2d-one'] = tsne_results[:, 0]
    data['tsne-2d-two'] = tsne_results[:, 1]
    return data


def visualize_tsne(combined_df, BASE_DIR, dataset, num_classes):
    plot = sns.relplot(data=combined_df, x="tsne-2d-one", y="tsne-2d-two", hue=LABEL, col="dataset_type")
    if dataset == "UNSW_NB15_10":
        pattern = re.compile(r'^([A-Za-z0-9_]+)_\d+$')
        match = pattern.match(dataset)
        if match:
            extracted_string = match.group(1)
            print(extracted_string)
            dataset = extracted_string
        else:
            print("No match found.")
    plt.suptitle(f't-SNE Visualization of {dataset} Datasets', y=1.02)
    plot.savefig(f"{BASE_DIR}\{dataset}_{num_classes}_tsne_visualization.png", dpi=300)
    plt.show()


def main(training_dataset, testing_dataset, dataset, num_classes, BASE_DIR):
    train_df, val_df = load_and_split_data(training_dataset)
    test_df = pd.read_csv(testing_dataset)
    combined_df = combine_datasets(train_df, val_df, test_df)
    combined_df[LABEL].to_csv(fr"{BASE_DIR}\{dataset}_LABEL.csv", index=False)
    visualize_counts(combined_df)

    combined_df = apply_tsne(combined_df)
    visualize_tsne(combined_df, BASE_DIR, dataset, num_classes)

    # 设置统一的颜色调色板
    palette = sns.color_palette("viridis", n_colors=combined_df[LABEL].nunique())
    plt.figure(figsize=(10, 5))
    # ax = sns.countplot(data=combined_df, x=LABEL, hue='dataset_type', palette=palette)

    sns.countplot(data=combined_df, x=LABEL, hue='dataset_type', palette=palette)
    if dataset == "UNSW_NB15_10":
        pattern = re.compile(r'^([A-Za-z0-9_]+)_\d+$')
        match = pattern.match(dataset)
        if match:
            extracted_string = match.group(1)
            print(extracted_string)
            dataset = extracted_string
        else:
            print("No match found.")
    plt.title(f'Distribution of Labels in {dataset} Datasets')
    plt.ylabel('Count')
    plt.xlabel('Label')
    plt.legend(title='Dataset Type')
    plt.tight_layout()
    # Annotate each bar with its count
    # for p in ax.patches:
    #     ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
    #                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    print(fr"{BASE_DIR}\{dataset}_{num_classes}_label_distribution.png")
    plt.savefig(fr"{BASE_DIR}\{dataset}_{num_classes}_label_distribution.png", dpi=300)
    plt.show()




if __name__ == "__main__":
    # for dataset, LABEL in [('UNSW_NB15_10', 'attack_cat'), ('NSL_KDD', 'label')]:
    for dataset, LABEL in [('UNSW_NB15_10', 'attack_cat'), ('NSL_KDD', 'label')]:
        for num_classes in ['2']:
            BASE_DIR = Path(fr"F:\new\data\{dataset}\processed")
            print(BASE_DIR)
            TSNE_FILE = fr"{BASE_DIR}\tsne_results.npy"
            training_dataset = BASE_DIR / 'train' / f'train_un_{num_classes}_label.csv'
            testing_dataset = BASE_DIR / 'test' / f'test_{num_classes}_label.csv'
            main(training_dataset, testing_dataset, dataset, num_classes, BASE_DIR)
