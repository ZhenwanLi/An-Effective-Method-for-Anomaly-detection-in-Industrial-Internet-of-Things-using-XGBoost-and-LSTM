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

from src.components.data.utils import BaseDataset, duplicate_samples
from src.components.data.nsl_kdd_label_types import NUM_CLASSES_COLUMNS, NUM_CLASSES_TO_LABEL_TYPE


class NSL_KDD(BaseDataset, ABC):
    """`nsl_kdd <http://205.174.165.80/CICDataset/NSL-KDD/Dataset/>`_ Dataset.

    Args:
        root (string): Root tsne_directory of dataset where ``KDD99/raw/train-images-idx3-ubyte``
            and  ``KDD99/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root tsne_directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    mirrors = [
        "http://205.174.165.80/CICDataset/NSL-KDD/Dataset/",
    ]

    resources = [
        ("NSL-KDD.zip", "a761a9d1ec6fd2ab017fc505c555f5b9"),
    ]
    NSL_KDD_COLUMNS = ["duration", "protocol_type", "service_type", "flag_type", "src_bytes", "dst_bytes", "land",
                       "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
                       "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
                       "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count2",
                       "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                       "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                       "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                       "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                       "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "leval"]  # 42

    LABEL = 'label'

    DROP_COLUMNS = ['leval']

    CAT_COLUMNS = ['protocol_type', 'service_type', 'flag_type']  # 3

    CONT_COLUMNS = ["duration", "src_bytes", "dst_bytes", "land",
                    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
                    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
                    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count2",
                    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]  # 38

    NUM_CLASSES_TO_LABEL_TYPE = NUM_CLASSES_TO_LABEL_TYPE

    NUM_CLASSES_COLUMNS = NUM_CLASSES_COLUMNS

    MAX_NUM_CLASSES = 23

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
        for file in ['NSL-KDD.zip']:
            file_path = os.path.join(self.raw_folder, file)
        return os.path.isfile(file_path) and file_path.endswith('.zip')

    def get_train_un_csv(self):
        super().get_train_un_csv()
        train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'KDDTrain+.txt'), encoding="utf-8"),
                               names=self.NSL_KDD_COLUMNS)
        train_df.drop(self.DROP_COLUMNS, axis=1, inplace=True)
        train_df.drop_duplicates(keep='last').reset_index(drop=True)
        train_df.dropna(axis=0, subset=[self.LABEL], inplace=True)
        train_df.to_csv(os.path.join(self.train_processed_folder, f'train_un.csv'), index=False)
        self.features_num = train_df.shape[1] - 1

    # def get_train_max_csv(self):
    #     super().get_train_max_csv()
    #     train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'KDDTrain+.txt'), encoding="utf-8"),
    #                            names=self.NSL_KDD_COLUMNS)
    #     train_df.drop(self.DROP_COLUMNS, axis=1, inplace=True)
    #     train_df.drop_duplicates(keep='last').reset_index(drop=True)
    #     train_df.dropna(axis=0, subset=[self.LABEL], inplace=True)
    #     duplicated_df = duplicate_samples(train_df, self.LABEL, self.min_augment_num)
    #     duplicated_df.to_csv(os.path.join(self.train_processed_augment_folder, f'{self.name}.csv'), index=False)

    def get_train_val_un_csv(self):
        super().get_train_val_un_csv()
        train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'KDDTrain+.txt'), encoding="utf-8"),
                               names=self.NSL_KDD_COLUMNS)
        train_df.drop(self.DROP_COLUMNS, axis=1, inplace=True)
        train_df.drop_duplicates(keep='last').reset_index(drop=True)
        train_df.dropna(axis=0, subset=[self.LABEL], inplace=True)
        data_train, data_val = train_test_split(train_df, test_size=0.3, random_state=42)
        data_train.to_csv(os.path.join(self.train_processed_folder, f'train_un.csv'), index=False)
        data_val.to_csv(os.path.join(self.val_processed_folder, f'val_un.csv'), index=False)
        self.features_num = train_df.shape[1] - 1

    def get_test_csv(self):
        super().get_test_csv()
        test_df = pd.read_csv(open(os.path.join(self.raw_folder, 'KDDTest+.txt'), encoding="utf-8"),
                              names=self.NSL_KDD_COLUMNS)
        test_df = test_df.drop(self.DROP_COLUMNS, axis=1)
        # test_df.drop_duplicates(keep='last').reset_index(drop=True)
        test_df = test_df.dropna(axis=0, subset=[self.LABEL])
        test_df.to_csv(os.path.join(self.test_processed_folder, f'test.csv'), index=False)

        # if self.name == "test":
        #     test_df = pd.read_csv(open(os.path.join(self.raw_folder, 'KDDTest+.txt'), encoding="utf-8"),
        #                           names=self.NSL_KDD_COLUMNS)
        #     test_df = test_df.drop(self.DROP_COLUMNS, axis=1)
        #     # test_df.drop_duplicates(keep='last').reset_index(drop=True)
        #     test_df = test_df.dropna(axis=0, subset=[self.LABEL])
        #     test_df.to_csv(os.path.join(self.test_processed_folder, f'{self.name}.csv'), index=False)
        #
        # if self.name == "test_known":
        #     train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'KDDTrain+.txt'), encoding="utf-8"),
        #                            names=self.NSL_KDD_COLUMNS)
        #     known_categories = train_df[self.LABEL].unique()
        #     test_df = pd.read_csv(open(os.path.join(self.raw_folder, 'KDDTest+.txt'), encoding="utf-8"),
        #                           names=self.NSL_KDD_COLUMNS)
        #     known_test_df = test_df[test_df[self.LABEL].isin(known_categories)]
        #     known_test_df = known_test_df.reset_index(drop=True)
        #     known_test_df = known_test_df.drop(self.DROP_COLUMNS, axis=1)
        #     # test_df.drop_duplicates(keep='last').reset_index(drop=True)
        #     known_test_df = known_test_df.dropna(axis=0, subset=[self.LABEL])
        #     known_test_df.to_csv(os.path.join(self.test_processed_folder, f'{self.name}.csv'), index=False)
        #
        # if self.name == "test_unknown":
        #     train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'KDDTrain+.txt'), encoding="utf-8"),
        #                            names=self.NSL_KDD_COLUMNS)
        #     known_categories = train_df[self.LABEL].unique()
        #     test_df = pd.read_csv(open(os.path.join(self.raw_folder, 'KDDTest+.txt'), encoding="utf-8"),
        #                           names=self.NSL_KDD_COLUMNS)
        #     unknown_test_df = test_df[~test_df[self.LABEL].isin(known_categories)]
        #     normal_category_data = test_df[test_df[self.LABEL] == 'normal']
        #     unknown_test_df = pd.concat([unknown_test_df, normal_category_data], ignore_index=True)
        #     unknown_test_df = unknown_test_df.reset_index(drop=True)
        #     unknown_test_df = unknown_test_df.drop(self.DROP_COLUMNS, axis=1)
        #     # test_df.drop_duplicates(keep='last').reset_index(drop=True)
        #     unknown_test_df = unknown_test_df.dropna(axis=0, subset=[self.LABEL])
        #     unknown_test_df.to_csv(os.path.join(self.test_processed_folder, f'{self.name}.csv'), index=False)
        #


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
                    for transform in ["Z"]:
                        NSL_KDD(root=data_dir, download=True, processed=True, num_classes=num_classes,
                                     method=method, threshold=threshold, augmenter=augmenter, transform=transform)
