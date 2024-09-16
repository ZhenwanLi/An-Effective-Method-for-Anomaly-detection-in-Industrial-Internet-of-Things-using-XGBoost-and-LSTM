# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 9/5/2023 下午 11:05
# @Last Modified by: zhenwan
# @Last Modified time: 9/5/2023  下午 11:05
# @file_name: kdd99_tsne.py
# @IDE: PyCharm
# @copyright: zhenwan

import os.path
from abc import ABC
import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data.utils import BaseDataset
from src.components.data.unsw_nb15_10_label_types import NUM_CLASSES_COLUMNS, NUM_CLASSES_TO_LABEL_TYPE


class UNSW_NB15_10(BaseDataset, ABC):
    """
    `unsw_nb15_10 <https://research.unsw.edu.au/projects/unsw-nb15-dataset>`_ Dataset.
    https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys?path=%2FUNSW-NB15%20-%20CSV%20Files
    """

    # https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=NUSW-NB15_features.csv
    # https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=NUSW-NB15_GT.csv
    # https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=The%20UNSW-NB15%20description.pdf
    # https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=UNSW-NB15_1.csv
    # https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=UNSW-NB15_LIST_EVENTS.csv
    # https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW_NB15_testing-set.csv
    # https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW_NB15_training-set.csv

    mirrors = [
        "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=",
    ]

    resources = [
        ("UNSW_NB15_testing-set.csv", "e55caabaa6cd4a8f1c06a227bcfababc"),
        ("UNSW_NB15_training-set.csv", "e0beea40262e46168cdb81476dbc27b4"),
        # ("UNSW-NB15_3.csv", "75e8567341edb760a23f803b920b69d4"),
        # ("UNSW-NB15_4.csv", "f55875cab7f037fdad40b5b737e7db5e"),
        # ("NUSW-NB15_features.csv", "9e6f08b95bc19e4c4986d4d5729f3ce9"),
        # ("NUSW-NB15_GT.csv", "efa04c34ec50b7ed10e6673f7ac67b59"),
        # ("UNSW-NB15_LIST_EVENTS.csv", "fb6a2eb2efb7e0a0e7abe2435db1d3f9"),
        # ("The%20UNSW-NB15%20description.pdf", "8ff130d94bc8c54f97019e5e24561323"),
    ]

    UNSW_NB15_10_COLUMNS = ['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
                            'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
                            'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
                            'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
                            'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
                            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
                            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
                            'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label']  # 45

    LABEL = 'attack_cat'

    DROP_COLUMNS = ['id', 'label']

    CAT_COLUMNS = ['proto', 'service', 'state']  # 3

    CONT_COLUMNS = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl',
                    'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit',
                    'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
                    'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
                    'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
                    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
                    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']  # 39

    NUM_CLASSES_TO_LABEL_TYPE = NUM_CLASSES_TO_LABEL_TYPE

    NUM_CLASSES_COLUMNS = NUM_CLASSES_COLUMNS

    MAX_NUM_CLASSES = 10

    def __init__(self, root: str, download: bool = False, processed: bool = False, num_classes: int = None,
                 method: str = None, threshold: float = None, augmenter: str = None, transform: str = None) -> None:
        """
        num_classes: 2, 5
        transform: 'N', 'Z', 'M'
        augmenter: smote, adasyn, smoteenn
        """

        super().__init__(root, download, processed, num_classes, method, threshold, augmenter, transform)
        self.min_augment_num = 5
        self.root = root
        self.num_classes = num_classes
        self.method = method
        self.threshold = threshold
        self.augmenter = augmenter
        self.transform = transform
        self.data_file = self.get_train_csvfile()
        print(self.data_file)

    def _check_raw_exists(self) -> bool:
        for file in ['UNSW_NB15_testing-set.csv', 'UNSW_NB15_training-set.csv']:
            file_path = os.path.join(self.raw_folder, file)
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def get_train_un_csv(self):
        super().get_train_un_csv()

        train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'UNSW_NB15_testing-set.csv'), encoding="utf-8"))
        train_df.drop(self.DROP_COLUMNS, axis=1, inplace=True)
        train_df.drop_duplicates(keep='last').reset_index(drop=True)
        train_df.dropna(axis=0, subset=[self.LABEL], inplace=True)
        train_df.to_csv(os.path.join(self.train_processed_folder, f'train_un.csv'), index=False)
        self.features_num = train_df.shape[1] - 1

    def get_train_val_un_csv(self):
        super().get_train_val_un_csv()

        train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'UNSW_NB15_testing-set.csv'), encoding="utf-8"))
        train_df.drop(self.DROP_COLUMNS, axis=1, inplace=True)
        train_df.drop_duplicates(keep='last').reset_index(drop=True)
        train_df.dropna(axis=0, subset=[self.LABEL], inplace=True)
        data_train, data_val = train_test_split(train_df, test_size=0.3, random_state=42)
        data_train.to_csv(os.path.join(self.train_processed_folder, f'train_un.csv'), index=False)
        data_val.to_csv(os.path.join(self.val_processed_folder, f'val_un.csv'), index=False)
        self.features_num = train_df.shape[1] - 1

    def get_test_csv(self):
        super().get_test_csv()
        test_df = pd.read_csv(open(os.path.join(self.raw_folder, 'UNSW_NB15_training-set.csv'), encoding="utf-8"))
        test_df = test_df.drop(self.DROP_COLUMNS, axis=1)
        # test_df.drop_duplicates(keep='last').reset_index(drop=True)
        test_df = test_df.dropna(axis=0, subset=[self.LABEL])
        test_df.to_csv(os.path.join(self.test_processed_folder, f'test.csv'), index=False)

        # if self.name == "test_known":
        #     train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'UNSW_NB15_training-set.csv'), encoding="utf-8"))
        #     known_categories = train_df[self.LABEL].unique()
        #     test_df = pd.read_csv(open(os.path.join(self.raw_folder, 'UNSW_NB15_testing-set.csv'), encoding="utf-8"))
        #     known_test_df = test_df[test_df[self.LABEL].isin(known_categories)]
        #     known_test_df = known_test_df.reset_index(drop=True)
        #     known_test_df = known_test_df.drop(self.DROP_COLUMNS, axis=1)
        #     # test_df.drop_duplicates(keep='last').reset_index(drop=True)
        #     known_test_df = known_test_df.dropna(axis=0, subset=[self.LABEL])
        #     known_test_df.to_csv(os.path.join(self.test_processed_folder, f'{self.name}.csv'), index=False)
        #
        # if self.name == "test_unknown":
        #     train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'UNSW_NB15_training-set.csv'), encoding="utf-8"))
        #     known_categories = train_df[self.LABEL].unique()
        #     test_df = pd.read_csv(open(os.path.join(self.raw_folder, 'UNSW_NB15_testing-set.csv'), encoding="utf-8"))
        #     unknown_test_df = test_df[~test_df[self.LABEL].isin(known_categories)]
        #     normal_category_data = test_df[test_df[self.LABEL] == 'normal.']
        #     unknown_test_df = pd.concat([unknown_test_df, normal_category_data], ignore_index=True)
        #     unknown_test_df = unknown_test_df.reset_index(drop=True)
        #     unknown_test_df = unknown_test_df.drop(self.DROP_COLUMNS, axis=1)
        #     # test_df.drop_duplicates(keep='last').reset_index(drop=True)
        #     unknown_test_df = unknown_test_df.dropna(axis=0, subset=[self.LABEL])
        #     unknown_test_df.to_csv(os.path.join(self.test_processed_folder, f'{self.name}.csv'), index=False)


if __name__ == '__main__':
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    print(project_root)
    data_dir = os.path.join(project_root, 'data')
    print(data_dir)

    for num_classes in [2]:
        for method in ["xgboost"]:
            for threshold in [0.002]:
            # for threshold in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012]:
            # for threshold in [0.0005, 0.0007, 0.0008, 0.0009, 0.0011, 0.0012]:
            #     for augmenter in ["", "smote", "smoteenn", "random"]:
                for augmenter in [""]:
                    for transform in [""]:
                        UNSW_NB15_10(root=data_dir, download=True, processed=True, num_classes=num_classes,
                                     method=method, threshold=threshold, augmenter=augmenter, transform=transform)
