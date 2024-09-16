# metrics_extractor.py
import json
import os

import pandas as pd
import re
from pathlib import Path
import logging
from tabulate import tabulate

from src.dl.plot_results.ext import extract_info_from_column

logging.basicConfig(level=logging.INFO)


def generate_latex_table(csv_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    df = df.round(4)
    df.replace(to_replace='_', value='\_', regex=True, inplace=True)

    # 处理第一列缺失值
    if df[df.columns[0]].dtype == 'float64' or df[df.columns[0]].dtype == 'int64':
        df[df.columns[0]].fillna(0, inplace=True)
    else:
        df[df.columns[0]].fillna('\/', inplace=True)

    # 对第一列进行处理，只在中间位置显示标签
    label_counts = df[df.columns[0]].value_counts()
    label_pos = {}  # 用于存储每个标签应该显示的位置
    for label, count in label_counts.items():
        mid_point = count // 2
        if count % 2 == 0:
            mid_point -= 1
        label_pos[label] = mid_point

    prev_label = None
    count = 0
    for i, row in df.iterrows():
        label = row[df.columns[0]]
        if prev_label == label:
            count += 1
        else:
            count = 0

        if count != label_pos[label]:
            df.at[i, df.columns[0]] = ''

        prev_label = label

    df.set_index(df.columns[0], inplace=True)

    # 将数据框转换为LaTeX三线表
    latex_table = tabulate(df, tablefmt='latex_raw', headers='keys', showindex='always')

    return latex_table


def get_value(folder, info):
    value = ''
    if folder == "thresholds":
        value = info['thresholds']
    elif folder == "models":
        value = info['model_name']
    elif folder == "criterions":
        value = info['criterions_name']
    return value



def extract_metrics(file_path):
    """Tries to open and read a json file, logging a warning if the file is not found."""
    try:
        with open(file_path, 'r') as f:
            records = json.load(f)
            return records
    except FileNotFoundError:
        logging.warning(f'File not found: {file_path}. Skipping this file.')
        return None

def extract_and_save_data(dataset, data_dir, output_dir, current_folder_name, files_to_read, subfolder_name):
    """Extracts data from specified files in subfolders, and saves the data to a CSV file."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    folders = [name for name in data_dir.iterdir() if name.is_dir()]

    rows = []
    for folder in folders:
        info = extract_info_from_column(dataset, folder.name)  # Extract info from folder name
        all_records = {}
        for file_name in files_to_read:
            records = extract_metrics(folder / subfolder_name / file_name)
            if records is not None:
                for category, metrics in records.items():
                    all_records.setdefault(category, {}).update(metrics)

        if not all_records:
            continue

        for category, metrics in all_records.items():

            row = {current_folder_name.title(): get_value(current_folder_name, info), 'Category': category}
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(files_to_read) > 1:
        output_file_name = f'{current_folder_name}.csv'
        output_path = output_dir / output_file_name
        df.to_csv(output_path, index=False)

        latex_table = generate_latex_table(output_path)
        # print(latex_table)  # 打印LaTeX表格
        with open(os.path.join(output_dir, f'{current_folder_name}.tex'), 'w') as file:  # 保存到.tex文件
            file.write(latex_table)
    if len(files_to_read) == 1:
        print(files_to_read)
        output_file_name = f"{current_folder_name}_{files_to_read[0].replace('.json', '')}.csv"
        output_path = output_dir / output_file_name
        df.to_csv(output_path, index=False)

        latex_table = generate_latex_table(output_path)
        # print(latex_table)  # 打印LaTeX表格
        with open(os.path.join(output_dir, f"{current_folder_name}_{files_to_read[0].replace('.json', '')}.tex"), 'w') as file:  # 保存到.tex文件
            file.write(latex_table)



if __name__ == "__main__":

    BASE_DIR = Path(fr'F:\new')
    for dataset in ['UNSW_NB15_10', 'NSL_KDD']:

        # files_to_read = ['derived_metrics.json']  # specify the files to read here
        # files_to_read = ['advanced_metrics.json']  # specify the files to read here
        files_to_read = ['derived_metrics.json', 'advanced_metrics.json']  # specify the files to read here
        subfolder_name = 'test'  # specify the subfolder name here

        # for folder in ['thresholds', 'models', 'criterions']:
        for folder in ['balance_test']:
        # for folder in ['best']:
            DATA_DIR = BASE_DIR / 'results' / 'dl' / dataset / folder
            OUTPUT_DIR = BASE_DIR / 'result_tables' / 'dl' / dataset / folder / 'test'

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            extract_and_save_data(dataset, DATA_DIR, OUTPUT_DIR, folder, files_to_read, subfolder_name)
