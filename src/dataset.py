from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np


def load_col_idx_map(path):
    """
    load column index map text file , file format --> lines of [column-name,idx]
    :param path: file path
    :return: dictionary of column name and index

    """
    col_idx_map = {}
    with open(path, mode='r') as map_txt_file:
        lines = map_txt_file.readlines()
        for line in lines:
            words = line.split(',')
            col_idx_map[words[0]] = int(words[1])
    return col_idx_map


def data_preprocessing(raw_data_df):
    fields_to_drop = ['yr', 'casual', 'registered', 'atemp', 'dteday', 'instant']
    preprocessed_data_df = raw_data_df.drop(fields_to_drop, axis=1)

    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday', 'holiday', 'workingday']
    target = preprocessed_data_df['cnt']
    preprocessed_data_df.drop(['cnt'], axis=1, inplace=True)
    for field in dummy_fields:
        dummies = pd.get_dummies(preprocessed_data_df[field], prefix=field, drop_first=False)
        preprocessed_data_df = pd.concat([preprocessed_data_df, dummies], axis=1)

    preprocessed_data_df.drop(dummy_fields, axis=1, inplace=True)
    # add target col to the last idx
    preprocessed_data_df = pd.concat([preprocessed_data_df, target], axis=1)

    return preprocessed_data_df

