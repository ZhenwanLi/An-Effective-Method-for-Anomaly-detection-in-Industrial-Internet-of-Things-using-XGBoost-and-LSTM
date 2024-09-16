import re


def extract_info_from_column(dataset, column):
    global pattern
    if dataset == "UNSW_NB15_10":
        pattern = r"UNSW_NB15_10_(\d+)_(\d+(?:\.\d+)?)_(\d+)_([A-Z_]+)_(\w+)_(\w+)_(\d+)"
    if dataset == "NSL_KDD":
        pattern = r"NSL_KDD_(\d+)_(\d+(?:\.\d+)?)_(\d+)_([A-Z_]+)_(\w+)_(\w+)_(\d+)"
    match = re.match(pattern, column)

    if match:
        info = {
            'num_classes': match.group(1),
            'thresholds': match.group(2),
            'features_num': match.group(3),
            'model_name': match.group(4),
            'optimizer_name': match.group(5),
            'criterions_name': match.group(6),
            'mix_epoch': match.group(7)
        }
        return info

    return {
        'num_classes': "",
        'thresholds': "",
        'features_num': "",
        'model_name': "",
        'optimizer_name': "",
        'criterions_name': "",
        'mix_epoch': ""
    }