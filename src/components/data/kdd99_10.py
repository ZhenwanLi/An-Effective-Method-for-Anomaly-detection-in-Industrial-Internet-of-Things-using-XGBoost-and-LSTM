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
from src.components.data.utils import BaseDataset, duplicate_samples
from kdd99_10_label_types import NUM_CLASSES_COLUMNS, NUM_CLASSES_TO_LABEL_TYPE

class KDD99_10(BaseDataset, ABC):
    """`KDD99_10 <http://kdd.ics.uci.edu/databases/kddcup99/>`_ Dataset."""

    mirrors = [
        "http://kdd.ics.uci.edu/databases/kddcup99/",
    ]

    resources = [
        ("corrected.gz", "7a5e0f6e66b8b5bdf9b232a074834751"),
        # ("kddcup.data.gz", "3745289f84bdd907c03baca24f9f81bc"),
        ("kddcup.data_10_percent.gz", "c421989ff187d340c1265ac3080a3229"),
        # ("training_attack_types", "f6d96584fb6d744adab35a0473fa15b4"),
        # ("typo-correction.txt", "4286c32a4dcc2bc9a8ad5b8f2fcc3b9e"),
        # ("kddcup.names", "19e3ed2afd7b83e2268599816e973c63"),

    ]

    KDD99_10_COLUMNS = ["duration", "protocol_type", "service_type", "flag_type", "src_bytes", "dst_bytes", "land",
                     "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
                     "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
                     "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count2",
                     "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                     "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                     "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                     "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                     "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]  # 42

    LABEL = 'label'

    DROP_COLUMNS = []

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

    def __init__(self, root: str, download: bool = False, processed: bool = False, name: str = "train", num_classes: int = None,
                 augmenter: str = None, transform: str = None, best_augment: bool = False) -> None:
        """
        num_classes: 2, 5
        transform: 'N', 'Z', 'M'
        augmenter: smote, adasyn, smoteenn
        """

        super().__init__(root, download, processed, name, num_classes, augmenter, transform, best_augment)
        self.min_augment_num = 5
        self.root = root
        self.name = name
        self.num_classes = num_classes
        self.augmenter = augmenter
        self.transform = transform
        self.best_augment = best_augment
        self.data_file = self.get_train_csvfile()
        print(self.data_file)


    def _check_raw_exists(self) -> bool:
        for file in ['corrected', 'corrected.gz', 'kddcup.data_10_percent', 'kddcup.data_10_percent.gz']:
            file_path = os.path.join(self.raw_folder, file)
        return os.path.isfile(file_path)

    def get_train_max_csv(self):
        super().get_train_max_csv()
        train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'kddcup.data_10_percent'), encoding="utf-8"),
                               names=self.KDD99_10_COLUMNS)
        train_df.drop(self.DROP_COLUMNS, axis=1, inplace=True)
        train_df.drop_duplicates(keep='last').reset_index(drop=True)
        train_df.dropna(axis=0, subset=[self.LABEL], inplace=True)
        duplicated_df = duplicate_samples(train_df, self.LABEL, self.min_augment_num)
        duplicated_df.to_csv(os.path.join(self.train_processed_augment_folder, f'{self.name}.csv'), index=False)

    def get_train_un_csv(self):
        super().get_train_un_csv()

        train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'kddcup.data_10_percent'), encoding="utf-8"),
                               names=self.KDD99_10_COLUMNS)
        train_df.drop(self.DROP_COLUMNS, axis=1, inplace=True)
        train_df.drop_duplicates(keep='last').reset_index(drop=True)
        train_df.dropna(axis=0, subset=[self.LABEL], inplace=True)
        train_df.to_csv(os.path.join(self.train_processed_folder, f'{self.name}_un.csv'), index=False)

    def get_test_csv(self):
        super().get_test_csv()
        if self.name == "test":
            test_df = pd.read_csv(open(os.path.join(self.raw_folder, 'corrected'), encoding="utf-8"),
                                  names=self.KDD99_10_COLUMNS)
            test_df = test_df.drop(self.DROP_COLUMNS, axis=1)
            # test_df.drop_duplicates(keep='last').reset_index(drop=True)
            test_df = test_df.dropna(axis=0, subset=[self.LABEL])
            test_df.to_csv(os.path.join(self.test_processed_folder, f'{self.name}.csv'), index=False)

        if self.name == "test_known":
            train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'kddcup.data_10_percent'), encoding="utf-8"),
                                   names=self.KDD99_10_COLUMNS)
            known_categories = train_df[self.LABEL].unique()
            test_df = pd.read_csv(open(os.path.join(self.raw_folder, 'corrected'), encoding="utf-8"),
                                  names=self.KDD99_10_COLUMNS)
            known_test_df = test_df[test_df[self.LABEL].isin(known_categories)]
            known_test_df = known_test_df.reset_index(drop=True)
            known_test_df = known_test_df.drop(self.DROP_COLUMNS, axis=1)
            # test_df.drop_duplicates(keep='last').reset_index(drop=True)
            known_test_df = known_test_df.dropna(axis=0, subset=[self.LABEL])
            known_test_df.to_csv(os.path.join(self.test_processed_folder, f'{self.name}.csv'), index=False)

        if self.name == "test_unknown":
            train_df = pd.read_csv(open(os.path.join(self.raw_folder, 'kddcup.data_10_percent'), encoding="utf-8"),
                                   names=self.KDD99_10_COLUMNS)
            known_categories = train_df[self.LABEL].unique()
            test_df = pd.read_csv(open(os.path.join(self.raw_folder, 'corrected'), encoding="utf-8"),
                                  names=self.KDD99_10_COLUMNS)
            unknown_test_df = test_df[~test_df[self.LABEL].isin(known_categories)]
            normal_category_data = test_df[test_df[self.LABEL] == 'normal.']
            unknown_test_df = pd.concat([unknown_test_df, normal_category_data], ignore_index=True)
            unknown_test_df = unknown_test_df.reset_index(drop=True)
            unknown_test_df = unknown_test_df.drop(self.DROP_COLUMNS, axis=1)
            # test_df.drop_duplicates(keep='last').reset_index(drop=True)
            unknown_test_df = unknown_test_df.dropna(axis=0, subset=[self.LABEL])
            unknown_test_df.to_csv(os.path.join(self.test_processed_folder, f'{self.name}.csv'), index=False)


if __name__ == '__main__':
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    print(project_root)
    data_dir = os.path.join(project_root, '')
    print(data_dir)

    for name in ["train", "test_known", "test", "test_unknown"]:
        for num_classes in [2, 5]:
            for augmenter in ["", "smote"]:
                for transform in ["", "Z"]:
                    for best_augment in [True, False]:
                        KDD99_10(root=data_dir, download=True, processed=True, name=name, num_classes=num_classes,
                                augmenter=augmenter, transform=transform, best_augment=best_augment)

    for name in ["train", "test_known"]:
        for num_classes in [23]:
            for augmenter in ["", "smote"]:
                for transform in ["", "Z"]:
                    for best_augment in [True, False]:

                        KDD99_10(root=data_dir, download=True, processed=True, name=name, num_classes=num_classes,
                                augmenter=augmenter, transform=transform, best_augment=best_augment)