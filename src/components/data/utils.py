# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 10/5/2023 上午 11:24
# @Last Modified by: zhenwan
# @Last Modified time: 10/5/2023  上午 11:24
# @file_name: utils.
# @IDE: PyCharm
# @copyright: zhenwan
import math
import os
import numbers
import os.path
from typing import Dict, List
from urllib.error import URLError
import numpy as np
import pandas as pd
# import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from abc import abstractmethod
from torchvision.datasets.utils import download_url


def duplicate_samples(df: pd.DataFrame, label_column: str, min_count: int) -> pd.DataFrame:
    """
    Duplicate samples in a DataFrame for label categories with count2 less than the minimum count2.

    Args:
        df (pd.DataFrame): Input DataFrame.
        label_column (str): Name of the label column.
        min_count (int): Minimum count2 required for each label category.

    Returns:
        pd.DataFrame: DataFrame with duplicated samples.
    """
    label_counts = df[label_column].value_counts()
    labels_to_duplicate = label_counts[label_counts < min_count].index.tolist()

    duplicated_samples = []
    for label in labels_to_duplicate:
        samples_to_duplicate = df[df[label_column] == label]
        num_duplicates = min_count - len(samples_to_duplicate)
        n = ((num_duplicates // len(samples_to_duplicate)) + 1) * len(samples_to_duplicate)
        duplicated_samples.append(samples_to_duplicate.sample(n=n, replace=True))

    duplicated_df = pd.concat([df] + duplicated_samples)
    return duplicated_df


class BaseDataset(Dataset):
    mirrors = None

    resources = None

    COLUMNS = None

    LABEL = None

    DROP_COLUMNS = None

    CAT_COLUMNS = None

    CONT_COLUMNS = None

    LABEL_TYPE = None

    NUM_CLASSES_TO_LABEL_TYPE = None

    NUM_CLASSES_COLUMNS = None

    MAX_NUM_CLASSES = None

    def __init__(
            self,
            root: str,
            download: bool = False,
            processed: bool = False,
            num_classes: int = None,
            method: str = None,
            threshold: float = None,
            augmenter: str = None,
            transform: str = None,
    ) -> None:
        """
        num_classes: 2, 3, 5, 32, 38, 40
        transform: 'N', 'Z', 'M'
        augmenter: smote, adasyn, smoteenn
        """

        self.min_augment_num = 5
        self.root = root
        self.num_classes = num_classes
        self.method = method
        self.threshold = threshold
        self.augmenter = augmenter
        self.transform = transform
        self.data_file = self.get_train_csvfile()

        if download:
            self.download()

        if processed:
            self.processed()

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def train_processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed/train")

    @property
    def val_processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed/val")

    @property
    def test_processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed/test")

    @abstractmethod
    def _check_raw_exists(self) -> bool:
        pass

    def _check_train_processed_exists(self) -> bool:
        for file in ['train_un.csv']:
            file_path = os.path.join(self.train_processed_folder, file)
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_csv_exists(self) -> bool:
        train_file_path = os.path.join(self.train_processed_folder, f'train_un.csv')
        val_file_path = os.path.join(self.val_processed_folder, f'val_un.csv')
        return os.path.isfile(train_file_path) and train_file_path.endswith('.csv') and os.path.isfile(
            val_file_path) and val_file_path.endswith('.csv')

    def _check_test_csv_exists(self) -> bool:
        file_path = os.path.join(self.test_processed_folder, f'test.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_classes_exists(self) -> bool:
        file_path = os.path.join(self.train_processed_folder, f'train_un_{self.num_classes}.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_val_classes_exists(self) -> bool:
        file_path = os.path.join(self.val_processed_folder, f'val_un_{self.num_classes}.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_test_classes_exists(self) -> bool:
        file_path = os.path.join(self.test_processed_folder, f'test_{self.num_classes}.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_train_test_label_exists(self) -> bool:
        train_file_path = os.path.join(self.train_processed_folder, f'train_un_{self.num_classes}_label.csv')
        test_file_path = os.path.join(self.test_processed_folder, f'test_{self.num_classes}_label.csv')
        return os.path.isfile(train_file_path) and os.path.isfile(test_file_path) and train_file_path.endswith(
            '.csv') and test_file_path.endswith('.csv')

    def _check_train_val_test_label_exists(self) -> bool:
        train_file_path = os.path.join(self.train_processed_folder, f'train_un_{self.num_classes}_label.csv')
        val_file_path = os.path.join(self.val_processed_folder, f'val_un_{self.num_classes}_label.csv')
        test_file_path = os.path.join(self.test_processed_folder, f'test_{self.num_classes}_label.csv')
        return os.path.isfile(train_file_path) and os.path.isfile(test_file_path) and train_file_path.endswith(
            '.csv') and test_file_path.endswith('.csv') and val_file_path.endswith('.csv') and val_file_path.endswith(
            '.csv')

    def _check_num_exists(self) -> bool:
        file_path = os.path.join(self.train_processed_folder, f'train_un_{self.num_classes}_num.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_val_num_exists(self) -> bool:
        file_path = os.path.join(self.val_processed_folder, f'val_un_{self.num_classes}_num.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_test_num_exists(self) -> bool:
        file_path = os.path.join(self.test_processed_folder, f'test_{self.num_classes}_num.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_val_num_augmenter_exists(self) -> bool:
        file_path = os.path.join(self.val_processed_folder,
                                 f'val_un_{self.num_classes}_num_{self.method}_{self.threshold}_{self.augmenter}.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_test_num_augmenter_exists(self) -> bool:
        file_path = os.path.join(self.test_processed_folder,
                                 f'test_{self.num_classes}_num_{self.method}_{self.threshold}_{self.augmenter}.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_aug_exists(self) -> bool:
        file_path = os.path.join(self.train_processed_folder,
                                 f'train_un_{self.num_classes}_num_{self.method}_{self.threshold}_{self.augmenter}.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_select_exists(self) -> bool:
        print("_check_select_exists")
        train_file_path = os.path.join(self.train_processed_folder,
                                       f'train_un_{self.num_classes}_num_{self.method}_{self.threshold}.csv')
        print(train_file_path)
        val_file_path = os.path.join(self.val_processed_folder,
                                     f'val_un_{self.num_classes}_num_{self.method}_{self.threshold}.csv')
        print(val_file_path)
        test_file_path = os.path.join(self.test_processed_folder,
                                      f'test_{self.num_classes}_num_{self.method}_{self.threshold}.csv')
        print(test_file_path)
        return os.path.isfile(train_file_path) and train_file_path.endswith('.csv') and os.path.isfile(
            val_file_path) and val_file_path.endswith('.csv') and os.path.isfile(
            test_file_path) and test_file_path.endswith('.csv')

    def _check_train_transform_exists(self) -> bool:
        file_path = os.path.join(self.train_processed_folder, f'train_un_num_{self.transform}.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_val_transform_exists(self) -> bool:
        file_path = os.path.join(self.train_processed_folder, f'val_un_num_{self.transform}.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_test_transform_exists(self) -> bool:
        file_path = os.path.join(self.train_processed_folder, f'test_num_{self.transform}.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def _check_train_test_select_exists(self) -> bool:
        train_file_path = os.path.join(self.train_processed_folder,
                                       f'train_un_{self.num_classes}_num_{self.method}_{self.threshold}.csv')
        # print(train_file_path)
        test_file_path = os.path.join(self.test_processed_folder,
                                      f'test_{self.num_classes}_num_{self.method}_{self.threshold}.csv')
        # print(test_file_path)
        return os.path.isfile(train_file_path) and train_file_path.endswith('.csv') and os.path.isfile(
            test_file_path) and test_file_path.endswith('.csv')

    def _check_select_test_exists(self) -> bool:
        print("_check_select_test_exists")
        file_path = os.path.join(self.test_processed_folder,
                                 f'test_{self.num_classes}_num_{self.method}_{self.threshold}.csv')
        return os.path.isfile(file_path) and file_path.endswith('.csv')

    def download(self) -> None:
        if self._check_raw_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    if str(filename).split('.')[-1] == "csv":
                        download_url(url, root=self.raw_folder, filename=filename, md5=md5)
                    else:
                        download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def processed(self) -> None:
        """Processed the data if it doesn't exist already."""
        self.get_train_un_csv()
        # self.get_train_val_un_csv()
        self.get_test_csv()

        self.get_train_un_classifier_csv()
        # self.get_val_un_classifier_csv()
        self.get_test_classifier_csv()

        # self.get_train_val_test_label_csv()
        self.get_train_test_label_csv()

        self.get_train_un_num_csv()
        # self.get_val_un_num_csv()
        self.get_test_num_csv()

        # self.get_num_train_val_test_select_features_csv()
        self.get_num_train_test_select_features_csv()

        # self.get_train_un_num_transform_csv()
        # # self.get_val_un_num_transform_csv()
        # self.get_test_num_transform_csv()

        self.get_train_un_num_augmenter_csv()
        self.get_test_num_augmenter_csv()
        # self.get_val_un_num_augmenter_csv()

    def get_train_un_csv(self):
        if self._check_train_processed_exists():
            print("get_train_un_csv is exist!")
            return
        os.makedirs(self.train_processed_folder, exist_ok=True)

    def get_train_val_un_csv(self):
        if self._check_csv_exists():
            print("get_train_val_un_csv is exist!")
            return
        os.makedirs(self.train_processed_folder, exist_ok=True)
        os.makedirs(self.val_processed_folder, exist_ok=True)

    def get_test_csv(self):
        if self._check_test_csv_exists():
            print("get_test_csv is exist!")
            return
        os.makedirs(self.test_processed_folder, exist_ok=True)

    def get_train_un_classifier_csv(self):
        if self._check_classes_exists():
            print("get_train_un_classifier_csv is exist!")
            return

        os.makedirs(self.train_processed_folder, exist_ok=True)

        data_df = pd.read_csv(open(os.path.join(self.train_processed_folder, f'train_un.csv'), encoding="utf-8"))
        class_counts = data_df[self.LABEL].value_counts()

        train_un_class_counts_df = class_counts.reset_index()
        train_un_class_counts_df.columns = ['Label', 'Count']
        # train_un_class_counts_df.to_csv(f'{self.data_file}_train_un_class_counts.csv', index=False)
        train_un_class_counts_df.to_csv(os.path.join(self.train_processed_folder, f'train_un_class_counts.csv'), index=False)

        data_df = data_df[data_df[self.LABEL].isin(self.NUM_CLASSES_COLUMNS[self.num_classes])]
        print(data_df)
        class_counts = data_df[self.LABEL].value_counts()
        print(class_counts)

        for target_label, source_labels in self.NUM_CLASSES_TO_LABEL_TYPE[self.num_classes].items():
            data_df[self.LABEL] = data_df[self.LABEL].replace(source_labels, target_label)

        # data_df = data_df.groupby(LABEL).filter(lambda x: len(x) >= 3)
        # class_counts = data_df[LABEL].value_counts()
        # print(class_counts)

        data_df.to_csv(os.path.join(self.train_processed_folder, f'train_un_{self.num_classes}.csv'), index=False)

    def get_val_un_classifier_csv(self):
        if self._check_val_classes_exists():
            return

        os.makedirs(self.val_processed_folder, exist_ok=True)

        data_df = pd.read_csv(open(os.path.join(self.val_processed_folder, f'val_un.csv'), encoding="utf-8"))
        class_counts = data_df[self.LABEL].value_counts()
        print(class_counts)

        data_df = data_df[data_df[self.LABEL].isin(self.NUM_CLASSES_COLUMNS[self.num_classes])]
        print(data_df)
        class_counts = data_df[self.LABEL].value_counts()
        print(class_counts)

        for target_label, source_labels in self.NUM_CLASSES_TO_LABEL_TYPE[self.num_classes].items():
            data_df[self.LABEL] = data_df[self.LABEL].replace(source_labels, target_label)

        # data_df = data_df.groupby(LABEL).filter(lambda x: len(x) >= 3)
        # class_counts = data_df[LABEL].value_counts()
        # print(class_counts)

        data_df.to_csv(os.path.join(self.val_processed_folder, f'val_un_{self.num_classes}.csv'), index=False)

    def get_test_classifier_csv(self):
        if self._check_test_classes_exists():
            print("get_test_classifier_csv is exist!")
            return

        os.makedirs(self.test_processed_folder, exist_ok=True)

        data_df = pd.read_csv(open(os.path.join(self.test_processed_folder, f'test.csv'), encoding="utf-8"))
        class_counts = data_df[self.LABEL].value_counts()
        print(class_counts)
        test_class_counts_df = class_counts.reset_index()
        test_class_counts_df.columns = ['Label', 'Count']
        # test_class_counts_df.to_csv(f'{self.data_file}_test_class_counts.csv', index=False)
        test_class_counts_df.to_csv(os.path.join(self.test_processed_folder, f'test_class_counts.csv'), index=False)

        data_df = data_df[data_df[self.LABEL].isin(self.NUM_CLASSES_COLUMNS[self.num_classes])]
        print(data_df)
        class_counts = data_df[self.LABEL].value_counts()
        print(class_counts)

        for target_label, source_labels in self.NUM_CLASSES_TO_LABEL_TYPE[self.num_classes].items():
            data_df[self.LABEL] = data_df[self.LABEL].replace(source_labels, target_label)

        # data_df = data_df.groupby(LABEL).filter(lambda x: len(x) >= 3)
        # class_counts = data_df[LABEL].value_counts()
        # print(class_counts)

        data_df.to_csv(os.path.join(self.test_processed_folder, f'test_{self.num_classes}.csv'), index=False)

    def get_train_val_test_label_csv(self):
        if self._check_train_val_test_label_exists():
            print("_check_label_exists")
            return

        os.makedirs(self.train_processed_folder, exist_ok=True)
        os.makedirs(self.val_processed_folder, exist_ok=True)
        os.makedirs(self.test_processed_folder, exist_ok=True)

        train_df = pd.read_csv(
            open(os.path.join(self.train_processed_folder, f'train_un_{self.num_classes}.csv'), encoding="utf-8"))
        val_df = pd.read_csv(
            open(os.path.join(self.val_processed_folder, f'val_un_{self.num_classes}.csv'), encoding="utf-8"))
        test_df = pd.read_csv(
            open(os.path.join(self.test_processed_folder, f'test_{self.num_classes}.csv'), encoding="utf-8"))

        combined_data = pd.concat([train_df, val_df, test_df])

        encoded_data = self.convert_features_to_numerical(combined_data)
        train_encoded_data = encoded_data.iloc[:len(train_df)]
        val_encoded_data = encoded_data.iloc[len(train_df):len(train_df) + len(val_df)]
        test_encoded_data = encoded_data.iloc[len(train_df) + len(val_df):]
        train_encoded_data.to_csv(os.path.join(self.train_processed_folder, f'train_un_{self.num_classes}_label.csv'),
                                  index=False)
        val_encoded_data.to_csv(os.path.join(self.val_processed_folder, f'val_un_{self.num_classes}_label.csv'),
                                index=False)
        test_encoded_data.to_csv(os.path.join(self.test_processed_folder, f'test_{self.num_classes}_label.csv'),
                                 index=False)

    def get_train_test_label_csv(self):
        if self._check_train_test_label_exists():
            print("get_train_test_label_csv is exist!")
            return

        os.makedirs(self.train_processed_folder, exist_ok=True)
        os.makedirs(self.test_processed_folder, exist_ok=True)

        train_df = pd.read_csv(
            open(os.path.join(self.train_processed_folder, f'train_un_{self.num_classes}.csv'), encoding="utf-8"))
        test_df = pd.read_csv(
            open(os.path.join(self.test_processed_folder, f'test_{self.num_classes}.csv'), encoding="utf-8"))

        combined_data = pd.concat([train_df, test_df])

        encoded_data = self.convert_features_to_numerical(combined_data)
        train_encoded_data = encoded_data.iloc[:len(train_df)]
        test_encoded_data = encoded_data.iloc[len(train_df):]
        train_encoded_data.to_csv(os.path.join(self.train_processed_folder, f'train_un_{self.num_classes}_label.csv'),
                                  index=False)
        test_encoded_data.to_csv(os.path.join(self.test_processed_folder, f'test_{self.num_classes}_label.csv'),
                                 index=False)

    def get_train_un_num_csv(self):
        if self._check_num_exists():
            print("get_train_un_num_csv is exist!")
            return

        os.makedirs(self.train_processed_folder, exist_ok=True)

        data_df = pd.read_csv(
            open(os.path.join(self.train_processed_folder, f'train_un_{self.num_classes}_label.csv'),
                 encoding="utf-8"))
        data = self.convert_label_to_numerical(data_df)

        data.to_csv(os.path.join(self.train_processed_folder, f'train_un_{self.num_classes}_num.csv'),
                    index=False)

    def get_val_un_num_csv(self):
        if self._check_val_num_exists():
            return

        os.makedirs(self.val_processed_folder, exist_ok=True)

        data_df = pd.read_csv(
            open(os.path.join(self.val_processed_folder, f'val_un_{self.num_classes}_label.csv'),
                 encoding="utf-8"))
        data = self.convert_label_to_numerical(data_df)

        data.to_csv(os.path.join(self.val_processed_folder, f'val_un_{self.num_classes}_num.csv'),
                    index=False)

    def get_test_num_csv(self):
        if self._check_test_num_exists():
            print("get_test_num_csv is exist!")
            return

        os.makedirs(self.test_processed_folder, exist_ok=True)

        data_df = pd.read_csv(
            open(os.path.join(self.test_processed_folder, f'test_{self.num_classes}_label.csv'),
                 encoding="utf-8"))
        data = self.convert_label_to_numerical(data_df)

        data.to_csv(os.path.join(self.test_processed_folder, f'test_{self.num_classes}_num.csv'), index=False)

    def get_train_un_num_transform_csv(self):
        if self._check_train_transform_exists():
            return
        elif self.transform is None:
            return
        else:
            transform_in_csv = f'train_un_{self.num_classes}_num_{self.method}_{self.threshold}.csv'
            transform_out_csv = f'train_un_{self.num_classes}_num_{self.method}_{self.threshold}_{self.transform}.csv'

            df = pd.read_csv(os.path.join(self.train_processed_folder, transform_in_csv))
            df_all, _ = self.normalize(df)
            df_all.to_csv(os.path.join(self.train_processed_folder, transform_out_csv), index=False)

    def get_val_un_num_transform_csv(self):
        if self._check_val_transform_exists():
            return
        else:
            transform_in_csv = f'val_un_{self.num_classes}_num.csv'
            transform_out_csv = f'val_un_{self.num_classes}_num_{self.transform}.csv'

            df = pd.read_csv(os.path.join(self.val_processed_folder, transform_in_csv))
            df_all, _ = self.normalize(df)
            df_all.to_csv(os.path.join(self.val_processed_folder, transform_out_csv), index=False)

    def get_test_num_transform_csv(self):
        if self._check_test_transform_exists():
            return
        elif self.transform is None:
            return
        else:
            transform_in_csv = f'test_{self.num_classes}_num_{self.method}_{self.threshold}.csv'
            transform_out_csv = f'test_{self.num_classes}_num_{self.method}_{self.threshold}_{self.transform}.csv'

            df = pd.read_csv(os.path.join(self.test_processed_folder, transform_in_csv))
            df_all, _ = self.normalize(df)
            df_all.to_csv(os.path.join(self.test_processed_folder, transform_out_csv), index=False)

    def get_num_train_val_test_select_features_csv(self):
        if self._check_select_exists():
            print(
                "self._check_select_exists() and self._check_select_val_exists() and self._check_select_test_exists()")
            return
        if self.method not in ['xgboost', 'shap', None]:
            raise ValueError(f"{self.augmenter} is not a valid method!")
        # print("why")
        train_un_select_in_csv = f'train_un_{self.num_classes}_num.csv'
        val_un_select_in_csv = f'val_un_{self.num_classes}_num.csv'
        test_select_in_csv = f'test_{self.num_classes}_num.csv'

        train_un_df = pd.read_csv(os.path.join(self.train_processed_folder, train_un_select_in_csv))
        val_un_df = pd.read_csv(os.path.join(self.val_processed_folder, val_un_select_in_csv))
        test_df = pd.read_csv(os.path.join(self.test_processed_folder, test_select_in_csv))

        train_un_df_all, val_un_df_all, test_df_all = self.train_val_test_select_features(train_un_df, val_un_df,
                                                                                          test_df)

        train_un_select_out_csv = f'train_un_{self.num_classes}_num_{self.method}_{self.threshold}.csv'
        print(train_un_select_out_csv)
        train_un_df_all.to_csv(os.path.join(self.train_processed_folder, train_un_select_out_csv), index=False)

        val_un_select_out_csv = f'val_un_{self.num_classes}_num_{self.method}_{self.threshold}.csv'
        print(train_un_select_out_csv)
        val_un_df_all.to_csv(os.path.join(self.val_processed_folder, val_un_select_out_csv), index=False)

        test_select_out_csv = f'test_{self.num_classes}_num_{self.method}_{self.threshold}.csv'
        print(test_select_out_csv)
        test_df_all.to_csv(os.path.join(self.test_processed_folder, test_select_out_csv), index=False)

    def get_num_train_test_select_features_csv(self):
        if self._check_train_test_select_exists():
            print("get_num_train_test_select_features_csv is exist!")
            return
        if self.method not in ['xgboost', 'shap', None]:
            raise ValueError(f"{self.augmenter} is not a valid method!")
        # print("why")
        train_un_select_in_csv = f'train_un_{self.num_classes}_num.csv'
        test_select_in_csv = f'test_{self.num_classes}_num.csv'

        train_un_df = pd.read_csv(os.path.join(self.train_processed_folder, train_un_select_in_csv))
        test_df = pd.read_csv(os.path.join(self.test_processed_folder, test_select_in_csv))

        train_un_df_all, test_df_all = self.train_test_select_features(train_un_df, test_df)

        train_un_select_out_csv = f'train_un_{self.num_classes}_num_{self.method}_{self.threshold}.csv'
        print(train_un_select_out_csv)
        train_un_df_all.to_csv(os.path.join(self.train_processed_folder, train_un_select_out_csv), index=False)

        test_select_out_csv = f'test_{self.num_classes}_num_{self.method}_{self.threshold}.csv'
        print(test_select_out_csv)
        test_df_all.to_csv(os.path.join(self.test_processed_folder, test_select_out_csv), index=False)

    def get_train_un_num_augmenter_csv(self):
        if self._check_aug_exists():
            print("get_train_un_num_augmenter_csv is exist!")
            return
        elif self.num_classes and self.augmenter:
            aug_in_csv = f'train_un_{self.num_classes}_num_{self.method}_{self.threshold}.csv'
            aug_out_csv = f'train_un_{self.num_classes}_num_{self.method}_{self.threshold}_{self.augmenter}.csv'
            if self.augmenter not in ['smote', 'smoteenn', 'random']:
                raise ValueError(f"{self.augmenter} is not a valid augmenter!")
            df = pd.read_csv(os.path.join(self.train_processed_folder, aug_in_csv))
            duplicated_df = duplicate_samples(df, self.LABEL, self.min_augment_num)
            df_all = self.augment(duplicated_df)
            df_all.to_csv(os.path.join(self.train_processed_folder, aug_out_csv), index=False)
        else:
            return

    def get_val_un_num_augmenter_csv(self):
        if self._check_val_num_augmenter_exists():
            return
        elif self.num_classes and self.augmenter:
            aug_in_csv = f'val_un_{self.num_classes}_num_{self.method}_{self.threshold}.csv'
            aug_out_csv = f'val_un_{self.num_classes}_num_{self.method}_{self.threshold}_{self.augmenter}.csv'
            if self.augmenter not in ['smote', 'smoteenn', 'random']:
                raise ValueError(f"{self.augmenter} is not a valid augmenter!")
            df = pd.read_csv(os.path.join(self.val_processed_folder, aug_in_csv))
            duplicated_df = duplicate_samples(df, self.LABEL, self.min_augment_num)
            df_all = self.augment(duplicated_df)
            df_all.to_csv(os.path.join(self.val_processed_folder, aug_out_csv), index=False)
        else:
            return

    def get_test_num_augmenter_csv(self):
        if self._check_test_num_augmenter_exists():
            print("get_test_num_augmenter_csv is exist!")
            return
        elif self.augmenter:
            os.makedirs(self.test_processed_folder, exist_ok=True)

            data_df = pd.read_csv(
                open(os.path.join(self.test_processed_folder,
                                  f'test_{self.num_classes}_num_{self.method}_{self.threshold}.csv'),
                     encoding="utf-8"))
            data_df.to_csv(
                os.path.join(self.test_processed_folder,
                             f'test_{self.num_classes}_num_{self.method}_{self.threshold}_{self.augmenter}.csv'),
                index=False)
        else:
            return

    def train_val_test_select_features(self, train_un_df: pd.DataFrame, val_un_df: pd.DataFrame,
                                       test_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        if self.method == 'xgboost':
            if self.threshold == 0:
                return train_un_df, val_un_df, test_df
            else:
                return self.train_val_test_select_features_xgboost(train_un_df, val_un_df, test_df)
        elif self.method == 'shap':
            return self.train_val_test_select_features_shap(train_un_df, val_un_df, test_df)
        else:
            return

    def train_test_select_features(self, train_un_df: pd.DataFrame, test_df: pd.DataFrame) -> (
            pd.DataFrame, pd.DataFrame):
        if self.method == 'xgboost':
            if self.threshold == 0:
                return train_un_df, test_df
            else:
                return self.train_test_select_features_xgboost(train_un_df, test_df)
        elif self.method == 'shap':
            return self.train_test_select_features_shap(train_un_df, test_df)
        else:
            return

    def train_val_test_select_features_xgboost(self, train_un_df, val_un_df, test_df):
        train_un_X = train_un_df.drop(columns=[self.LABEL])
        train_un_y = train_un_df[self.LABEL]

        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(train_un_X, train_un_y)

        # xgb.plot_importance(xgb_classifier)
        # plt.savefig(f"{self.get_train_csvfile().split()[0]}.png", dpi=300)
        # plt.show()

        feature_selector = SelectFromModel(xgb_classifier, threshold=self.threshold, prefit=True)
        selected_feature_indices = feature_selector.get_support()
        print("train_un_df shape:", train_un_df.shape)
        print("selected_feature_indices shape:", selected_feature_indices.shape)
        print(selected_feature_indices)
        print(type(selected_feature_indices))

        selected_feature_indices = np.append(selected_feature_indices, True)
        train_un_selected_features_df = train_un_df.iloc[:, selected_feature_indices]
        val_un_selected_features_df = val_un_df.iloc[:, selected_feature_indices]
        test_selected_features_df = test_df.iloc[:, selected_feature_indices]
        return train_un_selected_features_df, val_un_selected_features_df, test_selected_features_df

    def train_val_test_select_features_shap(self, train_un_df, val_un_df, test_df):
        train_un_X = train_un_df.drop(columns=[self.LABEL])
        train_un_y = train_un_df[self.LABEL]

        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(train_un_X, train_un_y)
        explainer = shap.Explainer(xgb_classifier)
        shap_values = explainer.shap_values(train_un_X)
        mean_abs_shap_values = abs(shap_values).mean(axis=0)
        selected_feature_indices = mean_abs_shap_values > self.threshold

        selected_feature_indices = np.append(selected_feature_indices, True)
        train_un_selected_features_df = train_un_df.iloc[:, selected_feature_indices]
        val_un_selected_features_df = val_un_df.iloc[:, selected_feature_indices]
        test_selected_features_df = test_df.iloc[:, selected_feature_indices]
        return train_un_selected_features_df, val_un_selected_features_df, test_selected_features_df

    def train_test_select_features_xgboost(self, train_un_df, test_df):
        train_un_X = train_un_df.drop(columns=[self.LABEL])
        train_un_y = train_un_df[self.LABEL]

        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(train_un_X, train_un_y)

        # xgb.plot_importance(xgb_classifier)
        # plt.savefig(f"{self.get_train_csvfile().split()[0]}.png", dpi=300)
        # plt.show()

        feature_selector = SelectFromModel(xgb_classifier, threshold=self.threshold, prefit=True)
        selected_feature_indices = feature_selector.get_support()
        print("train_un_df shape:", train_un_df.shape)
        print("selected_feature_indices shape:", selected_feature_indices.shape)
        print(selected_feature_indices)
        print(type(selected_feature_indices))

        selected_feature_indices = np.append(selected_feature_indices, True)
        train_un_selected_features_df = train_un_df.iloc[:, selected_feature_indices]
        test_selected_features_df = test_df.iloc[:, selected_feature_indices]
        return train_un_selected_features_df, test_selected_features_df

    def train_test_select_features_shap(self, train_un_df, test_df):
        train_un_X = train_un_df.drop(columns=[self.LABEL])
        train_un_y = train_un_df[self.LABEL]

        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(train_un_X, train_un_y)
        explainer = shap.Explainer(xgb_classifier)
        shap_values = explainer.shap_values(train_un_X)
        mean_abs_shap_values = abs(shap_values).mean(axis=0)
        selected_feature_indices = mean_abs_shap_values > self.threshold

        selected_feature_indices = np.append(selected_feature_indices, True)
        train_un_selected_features_df = train_un_df.iloc[:, selected_feature_indices]
        test_selected_features_df = test_df.iloc[:, selected_feature_indices]
        return train_un_selected_features_df, test_selected_features_df

    def augment(self, df: pd.DataFrame) -> (pd.DataFrame, List[str]):

        X = df.drop(columns=[self.LABEL])
        y = df[self.LABEL]

        if self.augmenter == 'smote':
            resampler = SMOTE(random_state=12345)
        # elif self.augmenter == 'adasyn':
        #     resampler = ADASYN(random_state=12345)
        elif self.augmenter == 'smoteenn':
            resampler = SMOTEENN(random_state=12345)
        elif self.augmenter == 'random':
            resampler = RandomOverSampler(random_state=12345)
        else:
            return

        X_resampled, y_resampled = resampler.fit_resample(X, y)

        df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)

        return df_resampled
        # df_all.to_csv(os.path.join(self.train_processed_folder, aug_out_csv), index=False)

    def count_classes(self, filename):
        dff = pd.read_csv(os.path.join(self.train_processed_folder, filename))
        # Get the count2 of each class in the label column
        class_counts = dff[self.LABEL].value_counts()
        print(class_counts)

    def normalize(self, data_df_un: pd.DataFrame) -> (pd.DataFrame, List[str]):
        scaler_params = {}
        for col in data_df_un.columns:
            if col in self.CONT_COLUMNS:
                if self.transform == 'N':
                    scaler_params = None
                    continue

                elif self.transform == 'Z':
                    scaler = StandardScaler()
                    data_df_un[col] = scaler.fit_transform(data_df_un[col].values.reshape(-1, 1))
                    scaler_params[col] = {
                        'mean': scaler.mean_[0],
                        'std': scaler.scale_[0]
                    }
                elif self.transform == 'M':
                    scaler = MinMaxScaler()
                    data_df_un[col] = scaler.fit_transform(data_df_un[col].values.reshape(-1, 1))
                    scaler_params[col] = {
                        'min': scaler.data_min_[0],
                        'max': scaler.data_max_[0]
                    }
                elif self.transform == 'R':
                    scaler = RobustScaler()
                    data_df_un[col] = scaler.fit_transform(data_df_un[col].values.reshape(-1, 1))
                    scaler_params[col] = {
                        'center': scaler.center_[0],
                        'scale': scaler.scale_[0]
                    }
                # elif self.transform == 'L':
                #     data_df_un[col] = np.log1p(data_df_un[col])
                # elif self.transform == 'P':
                #     data_df_un[col] = np.log1p(data_df_un[col])
                #     scaler = PowerTransformer()
                #     data_df_un[col] = scaler.fit_transform(data_df_un[col].values.reshape(-1, 1))
                #     scaler_params[col] = {
                #         'lambda': scaler.lambdas_[0]
                #     }
                # elif self.transform == 'LZ':
                #     data_df_un[col] = np.log1p(data_df_un[col])
                #     scaler = StandardScaler()
                #     data_df_un[col] = scaler.fit_transform(data_df_un[col].values.reshape(-1, 1))
                #     scaler_params[col] = {
                #         'mean': scaler.mean_[0],
                #         'std': scaler.scale_[0]
                #     }
                # elif self.transform == 'LM':
                #     data_df_un[col] = np.log1p(data_df_un[col])
                #     scaler = MinMaxScaler()
                #     data_df_un[col] = scaler.fit_transform(data_df_un[col].values.reshape(-1, 1))
                #     scaler_params[col] = {
                #         'min': scaler.data_min_[0],
                #         'max': scaler.data_max_[0]
                #     }
                # elif self.transform == 'LR':
                #     data_df_un[col] = np.log1p(data_df_un[col])
                #     scaler = RobustScaler()
                #     data_df_un[col] = scaler.fit_transform(data_df_un[col].values.reshape(-1, 1))
                #     scaler_params[col] = {
                #         'center': scaler.center_[0],
                #         'scale': scaler.scale_[0]
                #     }
        return data_df_un, scaler_params

    def test_normalize(self, data_df_un: pd.DataFrame, file_path) -> (pd.DataFrame, List[str]):
        scaler_params = pd.read_csv(file_path, index_col=0).to_dict()
        for col in data_df_un.columns:
            if col in scaler_params:
                if self.transform == 'N':
                    continue
                elif self.transform == 'Z':
                    data_df_un[col] = np.log1p(data_df_un[col])
                    scaler = StandardScaler()
                    scaler.mean_ = scaler_params[col]['mean']
                    scaler.scale_ = scaler_params[col]['std']
                    data_df_un[col] = scaler.transform(data_df_un[col].values.reshape(-1, 1))
                elif self.transform == 'M':
                    data_df_un[col] = np.log1p(data_df_un[col])
                    scaler = MinMaxScaler()
                    scaler.min_ = scaler_params[col]['min']
                    scaler.scale_ = scaler_params[col]['max'] - scaler_params[col]['min']
                    data_df_un[col] = scaler.transform(data_df_un[col].values.reshape(-1, 1))
                elif self.transform == 'R':
                    data_df_un[col] = np.log1p(data_df_un[col])
                    scaler = RobustScaler()
                    scaler.center_ = scaler_params[col]['center']
                    scaler.scale_ = scaler_params[col]['scale']
                    data_df_un[col] = scaler.transform(data_df_un[col].values.reshape(-1, 1))
                # elif self.transform == 'L':
                #     data_df_un[col] = np.log1p(data_df_un[col])
                # elif self.transform == 'P':
                #     data_df_un[col] = np.log1p(data_df_un[col])
                #     scaler = PowerTransformer()
                #     scaler.lambdas_ = scaler_params[col]['lambda']
                #     data_df_un[col] = scaler.transform(data_df_un[col].values.reshape(-1, 1))

        return data_df_un

    def convert_string_to_numerical(self, data: pd.DataFrame) -> (pd.DataFrame, List[str]):
        for col in data.columns:
            if col in self.CAT_COLUMNS:
                data[col].fillna('---')
                data[col] = LabelEncoder().fit_transform(data[col].astype(str))
            if col in self.CONT_COLUMNS:
                data[col] = data[col].fillna(0)
            if col == self.LABEL:
                LABEL_TYPE_2_num = {index: [item] for index, item in
                                    enumerate(list(self.NUM_CLASSES_TO_LABEL_TYPE[self.num_classes].keys()))}
                class_mapping = self.create_class_mapping(label_type=LABEL_TYPE_2_num)
                data[col] = np.array([class_mapping[label] for label in data[self.LABEL]])

        return data

    def convert_features_to_numerical(self, data: pd.DataFrame) -> (pd.DataFrame, List[str]):
        encoded_data = data.iloc[:, :-1].copy()
        one_hot_counts = []
        for col in data.columns:
            if col in self.CAT_COLUMNS:
                value_to_index = {value: index for index, value in enumerate(data[col].unique())}
                encoded_data[col] = data[col].map(value_to_index)
                one_hot = pd.get_dummies(encoded_data[col], prefix=col)
                print(col)
                print(len(data[col].unique()))
                one_hot_counts.append({'Label': col, 'Unique_Count': len(data[col].unique())})

                one_hot = one_hot.astype(int)
                encoded_data = pd.concat([encoded_data, one_hot], axis=1)
                encoded_data.drop(col, axis=1, inplace=True)
            if col in self.CONT_COLUMNS:
                encoded_data[col] = data[col].fillna(0)
            if col in self.LABEL:
                encoded_data[col] = data[col]
        # print(encoded_data.head())
        one_hot_counts_df = pd.DataFrame(one_hot_counts)
        one_hot_counts_df.columns = ['One_Hot', 'Count']
        one_hot_counts_df.to_csv(os.path.join(self.train_processed_folder, f'one_hot_counts.csv'), index=False)
        return encoded_data

    def max_convert_numerical_to_label(self, data: pd.DataFrame) -> (pd.DataFrame, List[str]):
        for col in data.columns:
            if col == self.LABEL:
                num_2_LABEL_TYPE = {index: item for index, item in
                                    enumerate(list(self.NUM_CLASSES_TO_LABEL_TYPE[self.MAX_NUM_CLASSES].keys()))}
                # class_mapping = self.create_class_mapping(label_type=num_2_LABEL_TYPE)
                data[col] = np.array([num_2_LABEL_TYPE[label] for label in data[self.LABEL]])

        return data

    def convert_numerical_to_label(self, data: pd.DataFrame) -> (pd.DataFrame, List[str]):
        for col in data.columns:
            if col == self.LABEL:
                num_2_LABEL_TYPE = {index: item for index, item in
                                    enumerate(list(self.NUM_CLASSES_TO_LABEL_TYPE[self.num_classes].keys()))}
                # class_mapping = self.create_class_mapping(label_type=num_2_LABEL_TYPE)
                data[col] = np.array([num_2_LABEL_TYPE[label] for label in data[self.LABEL]])

        return data

    def convert_label_to_numerical(self, data: pd.DataFrame) -> (pd.DataFrame, List[str]):
        for col in data.columns:
            if col == self.LABEL:
                LABEL_TYPE_2_num = {index: [item] for index, item in
                                    enumerate(list(self.NUM_CLASSES_TO_LABEL_TYPE[self.num_classes].keys()))}
                class_mapping = self.create_class_mapping(label_type=LABEL_TYPE_2_num)
                data[col] = np.array([class_mapping[label] for label in data[self.LABEL]])
        return data

    def convert_label_to_numerical_augment(self, data: pd.DataFrame) -> (pd.DataFrame, List[str]):
        for col in data.columns:
            if col == self.LABEL:
                LABEL_TYPE_2_num = {index: [item] for index, item in
                                    enumerate(list(self.NUM_CLASSES_TO_LABEL_TYPE[self.MAX_NUM_CLASSES].keys()))}
                class_mapping = self.create_class_mapping(label_type=LABEL_TYPE_2_num)
                print(data[self.LABEL])
                data[col] = np.array([class_mapping[label] for label in data[self.LABEL]])

        return data

    @abstractmethod
    def get_class_names(self):
        pass

    def create_class_mapping(self, label_type) -> Dict[str, int]:
        class_mapping = {}
        for i in label_type:
            for label in label_type[i]:
                class_mapping[label] = i
        return class_mapping

    def get_train_csvfile(self) -> str:
        filename = f"train_un"
        # if num_classes is not None:
        if isinstance(self.num_classes, numbers.Number):
            filename += f"_{self.num_classes}_num"
            if self.method:
                filename += f"_{self.method}_{self.threshold}"
            if self.augmenter:
                filename += f"_{self.augmenter.lower()}"
            if self.transform:
                filename += f"_{self.transform.upper()}"
        return filename + ".csv"

    def get_test_csvfile(self) -> str:
        filename = f"test"
        # if num_classes is not None:
        if isinstance(self.num_classes, numbers.Number):
            filename += f"_{self.num_classes}_num"
            if self.method:
                filename += f"_{self.method}_{self.threshold}"
            if self.augmenter:
                filename += f"_{self.augmenter.lower()}"
            if self.transform:
                filename += f"_{self.transform.upper()}"
        return filename + ".csv"

    def robust_scale(self, data, median, quartile_1, quartile_3):
        IQR = quartile_3 - quartile_1

        scaled_data = (data - median) / IQR

        return scaled_data


def get_csvfile(name: str, num_classes: int, augmenter: str, transform: str) -> str:
    filename = f"{name}_un"
    # if num_classes is not None:
    if isinstance(num_classes, numbers.Number):
        filename += f"_{num_classes}_num"
        if augmenter:
            filename += f"_{augmenter.lower()}"
        if transform:
            filename += f"_{transform.upper()}"
    else:
        filename += "_num"
        if transform:
            filename += f"_{transform.upper()}"
    return filename + ".csv"


def split_data_train_val(trainset):
    allset_num = len(trainset)
    print(allset_num)
    train_split = math.floor(allset_num * 11 / 14)
    val_split = allset_num - train_split
    # test_split = allset_num - train_split - val_split
    # print(f'allset: {allset_num}')
    # print(f'train_split: {train_split}')
    # print(f'val_split: {val_split}')
    # print(f'test_split: {test_split}')
    # print(f'train_val_test_split: [{train_split}, {val_split}, {test_split}]')
    # print(f'train_val_test_split: [{train_split}, {val_split}, {test_split}]')
    print(f'train_val_split: [{train_split}, {val_split}]')


def split_data(trainset):
    allset_num = len(trainset)
    train_split = math.floor(allset_num * 11 / 14)
    val_split = math.floor(allset_num / 14)
    test_split = allset_num - train_split - val_split
    # print(f'allset: {allset_num}')
    # print(f'train_split: {train_split}')
    # print(f'val_split: {val_split}')
    # print(f'test_split: {test_split}')
    # print(f'train_val_test_split: [{train_split}, {val_split}, {test_split}]')
    # print(f'train_val_test_split: [{train_split}, {val_split}, {test_split}]')
    print(f'train_val_test_split: [{train_split}, {val_split}, {test_split}]')


if __name__ == '__main__':
    file_name = get_csvfile(num_classes=2, augmenter=None, transform='Z')
    print(file_name)
