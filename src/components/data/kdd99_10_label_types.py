# -*- coding: utf-8 -*-
# @Author: zhenwan
# @Time: 7/13/2023 3:34 PM
# @Last Modified by: zhenwan
# @Last Modified time: 7/13/2023  3:34 PM
# @file_name: label_types_nsl_kdd.
# @IDE: PyCharm
# @copyright: zhenwan
import numbers

LABEL_TYPE = ['normal.', 'ipsweep.', 'mscan.', 'nmap.', 'portsweep.', 'saint.', 'satan.', 'apache2.', 'back.',
              'land.', 'mailbomb.', 'neptune.', 'pod.', 'processtable.', 'smurf.', 'teardrop.', 'udpstorm.',
              'buffer_overflow.', 'httptunnel.', 'loadmodule.', 'perl.', 'ps.', 'rootkit.', 'sqlattack.', 'xterm.',
              'ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'named.', 'phf.', 'sendmail.', 'snmpgetattack.',
              'snmpguess.', 'spy.', 'warezclient.', 'warezmaster.', 'worm.', 'xlock.', 'xsnoop.']  # 40

LABEL_TYPE_2 = {'Normal': ['normal.'],
                'Attack': ['ipsweep.', 'mscan.', 'nmap.', 'portsweep.', 'saint.', 'satan.',
                           'apache2.', 'back.', 'land.', 'mailbomb.', 'neptune.', 'pod.', 'processtable.', 'smurf.',
                           'teardrop.', 'udpstorm.',
                           'buffer_overflow.', 'httptunnel.', 'loadmodule.', 'perl.', 'ps.', 'rootkit.',
                           'sqlattack.',
                           'xterm.',
                           'ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'named.', 'phf.', 'sendmail.',
                           'snmpgetattack.',
                           'snmpguess.', 'spy.', 'warezclient.', 'warezmaster.', 'worm.', 'xlock.', 'xsnoop.']}  # 2

LABEL_TYPE_5 = {'Normal': ['normal.'],
                'Probing': ['ipsweep.', 'mscan.', 'nmap.', 'portsweep.', 'saint.', 'satan.'],
                'DOS': ['apache2.', 'back.', 'land.', 'mailbomb.', 'neptune.', 'pod.', 'processtable.',
                        'smurf.', 'teardrop.', 'udpstorm.'],
                'U2R': ['buffer_overflow.', 'httptunnel.', 'loadmodule.', 'perl.', 'ps.', 'rootkit.',
                        'sqlattack.', 'xterm.'],
                'R2L': ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'named.', 'phf.', 'sendmail.',
                        'snmpgetattack.', 'snmpguess.', 'spy.', 'warezclient.', 'warezmaster.', 'worm.', 'xlock.',
                        'xsnoop.']}  # 5


LABEL_TYPE23 = ['normal.', 'neptune.', 'warezclient.', 'ipsweep.', 'portsweep.', 'teardrop.', 'nmap.', 'satan.', 'smurf.', 'pod.', 'back.', 'guess_passwd.', 'ftp_write.', 'multihop.', 'rootkit.', 'buffer_overflow.', 'imap.', 'warezmaster.', 'phf.', 'land.', 'loadmodule.', 'spy.', 'perl.']
LABEL_TYPE_23 = {(label): [label] for label in LABEL_TYPE23}  # 40
LABEL_TYPE_40 = {(label.capitalize()): [label] for label in LABEL_TYPE}  # 40

NUM_CLASSES_TO_LABEL_TYPE = {
    2: LABEL_TYPE_2,
    5: LABEL_TYPE_5,
    23: LABEL_TYPE_23,
    40: LABEL_TYPE_40,
}

NUM_CLASSES_COLUMNS = {
    2: LABEL_TYPE,
    5: LABEL_TYPE,
    23: LABEL_TYPE,
    40: LABEL_TYPE,
}


def get_labels(num_classes):
    if num_classes == 2:
        class_names = list(LABEL_TYPE_2.keys())
    elif num_classes == 5:
        class_names = list(LABEL_TYPE_5.keys())
    elif num_classes == 23:
        class_names = list(LABEL_TYPE_23.keys())
    elif num_classes == 40:
        class_names = list(LABEL_TYPE_40.keys())
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