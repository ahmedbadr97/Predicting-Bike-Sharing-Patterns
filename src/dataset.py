from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import torch

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
    """
    drop pointless features
    do hot_encoding to categorical fields
    :param raw_data_df: raw data dataframe
    :return: preprocessed data dataframe
    """
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


class BikeSharingPatterns(Dataset):
    def __init__(self, csv_path, col_idx_dict, target_label):
        """
        BikeSharingPatterns custom pytorch dataset splits data into numpy features and target
        :param csv_path: dataset csv file
        :param col_idx_dict: dictionary with keys col_names and values the index of the column in the expected input
         to the model
        :param target_label: the name of the target label in the given csv_file
        """
        # load data
        df = pd.read_csv(csv_path)
        labels_df = df[target_label]
        df.drop(target_label, axis=1, inplace=True)
        self.np_data = np.zeros_like(df)
        self.no_rows = df.shape[0]

        self.np_labels = labels_df.to_numpy(dtype=float)

        # convert data to numpy with column idx same like col_idx_dict

        for col_name, col_idx in col_idx_dict.items():
            for row_idx in range(self.no_rows):
                self.np_data[row_idx][col_idx] = df[col_name][row_idx]

    def __getitem__(self, idx):
        return torch.tensor(self.np_data[idx], dtype=torch.float), self.np_labels[idx]

    def __len__(self):
        return self.no_rows
