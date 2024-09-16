# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 10/17/2023 7:48 PM
# @Last Modified by: zhenwan
# @Last Modified time: 10/17/2023  7:48 PM
# @file_name: best_plot.
# @IDE: PyCharm
# @copyright: zhenwan
from pathlib import Path

from src.dl.plots import plot_training, plot_testing


# BASE_DIR = Path(r"/home/tyxk/Desktop/ZhenWan/new/results/dl/NSL_KDD/best/NSL_KDD_2_0.003_23_MIX_LSTM_Adam_WeightClassBalancedLoss_30")
# BASE_DIR = Path(r"/home/tyxk/Desktop/ZhenWan/new/results/dl/UNSW_NB15_10/best/UNSW_NB15_10_2_0.002_31_MIX_LSTM_Adam_WeightClassBalancedLoss_10")
BASE_DIR = Path(r"F:\new\results\dl\UNSW_NB15_10\best\UNSW_NB15_10_2_0.002_31_MIX_LSTM_Adam_WeightClassBalancedLoss_10")

training_records = BASE_DIR/'train'/'training_records.json'
testing_records = BASE_DIR/'test'/'testing_records.json'

plot_training(training_records)
plot_testing(testing_records)
