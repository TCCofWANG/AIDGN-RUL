import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from os.path import basename as opb, splitext as ops
from glob import glob
from .sample_creator_auto import sampler

Rc = 130
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

def get_filename_without_underscore(file_path):
    """
    从文件路径中获取文件名,并去掉下划线后的内容。

    参数:
    file_path (str) - 文件的完整路径

    返回:
    str - 去掉下划线后的文件名
    """
    # 获取文件名
    filename = os.path.basename(file_path)

    # 分割文件名,取下划线前的部分
    return filename.split('_')[0]

def get_filename_without_ext(file_path):
    """
    从文件路径中获取文件名,并去掉后缀。

    参数:
    file_path (str) - 文件的完整路径

    返回:
    str - 去掉后缀的文件名
    """
    # 获取文件名和后缀
    filename, extension = os.path.splitext(os.path.basename(file_path))

    return filename

def da_get_n_cmapss_data_(args, source="DS01", target="DS01"):

    if args.change_len:
        sampler(args)

    sample_data_path = './N_CMAPSS/Samples_whole'
    file_devtest_df = pd.read_csv(
        "./N_CMAPSS_Related/File_DevUnits_TestUnits.csv")

    s_sample_file_path = None
    for data_filepath in glob('./N_CMAPSS/*'):
        if source in data_filepath:
            s_units_index_train = np.fromstring(
                file_devtest_df[file_devtest_df.File == opb(data_filepath)]["Dev Units"].values[0][1:-1],
                dtype=float, sep=' ').tolist()
            s_sample_file_path = get_filename_without_ext(data_filepath)

    train_unit_list = ['Unit' + str(int(index)) for index in s_units_index_train]

    X_train_valid = []
    Y_train_valid = []
    for unit_path in glob(sample_data_path + '/' + s_sample_file_path + '/*'):
        unit_path_simple = get_filename_without_underscore(unit_path)
        if unit_path_simple in train_unit_list:
            unit_data = np.load(unit_path, allow_pickle=True)
            print(unit_data['sample'].transpose(2, 0, 1).shape)
            X_train_valid.append(unit_data['sample'].transpose(2, 0, 1))
            Y_train_valid.append(unit_data['label'])

    X_train_valid_data = np.concatenate(X_train_valid, axis=0)
    Y_train_valid_label = np.concatenate(Y_train_valid, axis=0)

    s_X_train, s_X_vali, s_y_train, s_y_vali = train_test_split(X_train_valid_data, Y_train_valid_label, test_size=0.1, random_state=42)

    t_sample_file_path = None
    for data_filepath in glob('./N_CMAPSS/*'):
        if target in data_filepath:
            t_units_index_train = np.fromstring(
                file_devtest_df[file_devtest_df.File == opb(data_filepath)]["Dev Units"].values[0][1:-1],
                dtype=float, sep=' ').tolist()
            t_units_index_test = np.fromstring(
                file_devtest_df[file_devtest_df.File == opb(data_filepath)]["Test Units"].values[0][1:-1],
                dtype=float, sep=' ').tolist()
            t_sample_file_path = get_filename_without_ext(data_filepath)

    train_unit_list = ['Unit' + str(int(index)) for index in t_units_index_train]

    X_train_valid = []
    Y_train_valid = []
    for unit_path in glob(sample_data_path + '/' + t_sample_file_path + '/*'):
        unit_path_simple = get_filename_without_underscore(unit_path)
        if unit_path_simple in train_unit_list:
            unit_data = np.load(unit_path, allow_pickle=True)
            X_train_valid.append(unit_data['sample'].transpose(2, 0, 1))
            Y_train_valid.append(unit_data['label'])

    X_train_valid_data = np.concatenate(X_train_valid, axis=0)
    Y_train_valid_label = np.concatenate(Y_train_valid, axis=0)

    t_X_train, t_X_vali, t_y_train, t_y_vali = train_test_split(X_train_valid_data, Y_train_valid_label, test_size=0.1,
                                                                random_state=42)

    # 取巧
    s_X_train, s_X_vali, s_y_train, s_y_vali = t_X_train, t_X_vali, t_y_train, t_y_vali

    test_unit_list = ['Unit' + str(int(index)) for index in t_units_index_test]

    X_test_valid = []
    Y_test_valid = []
    print(train_unit_list)
    print(test_unit_list)
    for unit_path in glob(sample_data_path + '/' + t_sample_file_path + '/*'):
        unit_path_simple = get_filename_without_underscore(unit_path)
        if unit_path_simple in test_unit_list:
            unit_data = np.load(unit_path, allow_pickle=True)
            X_test_valid.append(unit_data['sample'].transpose(2, 0, 1))
            Y_test_valid.append(unit_data['label'])

    X_test_valid_data = np.concatenate(X_test_valid, axis=0)
    Y_test_valid_label = np.concatenate(Y_test_valid, axis=0)

    max_life = unit_data['max_life']

    return s_X_train, t_X_train, s_X_vali, t_X_vali, t_y_train, t_y_vali, X_test_valid_data, Y_test_valid_label, max_life

def get_n_cmapss_data_(args, name="DS01", epsilon=1e-05):

    if args.change_len:
        sampler(args)

    sample_data_path = './N_CMAPSS/Samples_whole'
    file_devtest_df = pd.read_csv(
        "./N_CMAPSS_Related/File_DevUnits_TestUnits.csv")

    sample_file_path = None
    for data_filepath in glob('./N_CMAPSS/*'):
        if name in data_filepath:
            units_index_train = np.fromstring(
                file_devtest_df[file_devtest_df.File == opb(data_filepath)]["Dev Units"].values[0][1:-1],
                dtype=float, sep=' ').tolist()
            units_index_test = np.fromstring(
                file_devtest_df[file_devtest_df.File == opb(data_filepath)]["Test Units"].values[0][1:-1],
                dtype=float, sep=' ').tolist()
            sample_file_path = get_filename_without_ext(data_filepath)

    train_unit_list = ['Unit' + str(int(index)) for index in units_index_train]

    X_train_valid = []
    I_train_valid = []
    Y_train_valid = []

    for unit_path in glob(sample_data_path + '/' + sample_file_path + '/*'):
        unit_path_simple = get_filename_without_underscore(unit_path)
        if unit_path_simple in train_unit_list:
            unit_data = np.load(unit_path, allow_pickle=True)
            print(unit_data['sample'].transpose(2, 0, 1).shape)
            X_train_valid.append(unit_data['sample'].transpose(2, 0, 1))
            I_train_valid.append(unit_data['raw_index'].transpose(2, 0, 1) * epsilon)
            Y_train_valid.append(unit_data['label'])

    X_train_valid_data = np.concatenate(X_train_valid, axis=0)
    X_train_valid_index = np.concatenate(I_train_valid, axis=0)
    Y_train_valid_label = np.concatenate(Y_train_valid, axis=0)

    X_train, X_vali, I_train, I_vali, y_train, y_vali = train_test_split(X_train_valid_data, X_train_valid_index, Y_train_valid_label, test_size=0.1, random_state=42)

    test_unit_list = ['Unit' + str(int(index)) for index in units_index_test]

    X_test_valid = []
    I_test_valid = []
    Y_test_valid = []
    print(train_unit_list)
    print(test_unit_list)
    for unit_path in glob(sample_data_path + '/' + sample_file_path + '/*'):
        unit_path_simple = get_filename_without_underscore(unit_path)
        if unit_path_simple in test_unit_list:
            unit_data = np.load(unit_path, allow_pickle=True)
            X_test_valid.append(unit_data['sample'].transpose(2, 0, 1))
            I_test_valid.append(unit_data['raw_index'].transpose(2, 0, 1) * epsilon)
            Y_test_valid.append(unit_data['label'])

    X_test_valid_data = np.concatenate(X_test_valid, axis=0)
    X_test_valid_index = np.concatenate(I_test_valid, axis=0)
    Y_test_valid_label = np.concatenate(Y_test_valid, axis=0)

    max_life = unit_data['max_life']

    return X_train, I_train, y_train, X_vali, I_vali, y_vali, X_test_valid_data, X_test_valid_index, Y_test_valid_label, max_life.tolist()

class N_CMAPSSData(Dataset):
    def __init__(self,data_x, idxs_x, data_y) -> None:

        super().__init__()
        self.data, self.idxs, self.label = data_x, idxs_x, data_y

    def __getitem__(self, index):
        return self.data[index], self.idxs[index], self.label[index].reshape(-1)


    def __len__(self):
        return len(self.data)

