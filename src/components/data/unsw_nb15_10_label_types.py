# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 7/13/2023 3:34 PM
# @Last Modified by: zhenwan
# @Last Modified time: 7/13/2023  3:34 PM
# @file_name: label_types_nsl_kdd.
# @IDE: PyCharm
# @copyright: zhenwan
import numbers

LABEL_TYPE = ['Normal', 'Reconnaissance', 'Backdoor', 'DoS', 'Exploits', 'Analysis', 'Fuzzers', 'Worms',
              'Shellcode', 'Generic']  # 10

LABEL_TYPE_2 = {'Normal': ['Normal'],
                'Attack': ['Reconnaissance', 'Backdoor', 'DoS', 'Exploits', 'Analysis', 'Fuzzers', 'Worms',
                           'Shellcode', 'Generic']}  # 2

LABEL_TYPE_10 = {label: [label] for label in LABEL_TYPE}  # 10

NUM_CLASSES_TO_LABEL_TYPE = {
    2: LABEL_TYPE_2,
    10: LABEL_TYPE_10,
}

NUM_CLASSES_COLUMNS = {
    2: LABEL_TYPE,
    10: LABEL_TYPE,
}


def get_labels(num_classes):
    if num_classes == 2:
        class_names = list(LABEL_TYPE_2.keys())
    elif num_classes == 10:
        class_names = list(LABEL_TYPE_10.keys())
    else:
        raise ValueError("Unsupported number of classes: {}".format(num_classes))
    return class_names



def get_augment_csvfile(num_classes: int, name: str, augmenter: str, transform: str) -> str:
    filename = f"{name}"
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


def get_csvfile(num_classes: int, name: str, augmenter: str, transform: str) -> str:
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

