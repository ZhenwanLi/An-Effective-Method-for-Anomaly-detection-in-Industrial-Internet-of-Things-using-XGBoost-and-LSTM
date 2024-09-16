import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

LABEL = 'attack_cat'
# TSNE_DIR = r'/home/tyxk/Desktop/ZhenWan/new/results/dl/NSL_KDD/best/NSL_KDD_2_0.003_23_MIX_LSTM_Adam_WeightClassBalancedLoss_30'
TSNE_DIR = r'/home/tyxk/Desktop/ZhenWan/new/results/dl/UNSW_NB15_10/best/UNSW_NB15_10_2_0.002_31_MIX_LSTM_Adam_WeightClassBalancedLoss_10'

def load_data(file_name):
    return np.load(file_name)

def visualize_tsne(data, labels, folder_name):
    df = pd.DataFrame(data, columns=['tsne-2d-one', 'tsne-2d-two'])
    df[LABEL] = labels

    plt.figure(figsize=(10, 5))
    sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue=LABEL, data=df)
    plt.title(f't-SNE Visualization for {folder_name}')
    plt.savefig(f"{TSNE_DIR}/{folder_name}_tsne_visualization.png", dpi=300)
    plt.close()

def main():
    # 遍历 results/tsne 下的文件夹
    for folder in os.listdir(TSNE_DIR):
        folder_path = os.path.join(TSNE_DIR, folder)
        if os.path.isdir(folder_path):
            label_file = os.path.join(folder_path, 'label.npy')
            if os.path.exists(label_file):
                labels = load_data(label_file)
                for file in os.listdir(folder_path):
                    if file.endswith('.npy') and 'label' not in file:
                        data = load_data(os.path.join(folder_path, file))
                        visualize_tsne(data, labels, file.split('.')[0])

if __name__ == "__main__":
    main()
