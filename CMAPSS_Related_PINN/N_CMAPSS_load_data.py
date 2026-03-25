import sys
import torch
sys.path.append("..")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import  os


def Cmapss_test_batch_generator(test_data, sequence_length=5,is_diff=False):
    engine_ids = list(test_data["engine"].unique())

    feature_number = test_data.shape[1] - 2

    x_batch = []
    y_batch = []

    for _id in set(test_data['engine']):
        test_of_one_id = test_data[test_data['engine'] == _id]
        #每个引擎id只取最末尾的一个wz，不切，相当于test_date就去wz的长度  # 一个enging只取一个窗口
        if test_of_one_id.shape[0] >= sequence_length:
            x_batch.append(test_of_one_id.iloc[-sequence_length:, 2:].values)
            y_batch.append(test_of_one_id.iloc[-1:, 1].values)

        else:
            num_pad = sequence_length - len(test_of_one_id)
            new_sg = test_of_one_id
            for idx in range(num_pad):
                new_sg = pd.concat([new_sg.head(1), new_sg], axis=0)

            x_batch.append(new_sg.iloc[:,2:].values)
            y_batch.append(new_sg.iloc[-1:, 1].values)

    return np.array(x_batch), np.array(y_batch)


def Cmapss_train_vali_batch_generator(training_data, sequence_length=15):
    """
    data generate for turbofan dataset
    Generator function for creating random batches of training-data
    """
    # sequence_lenth 表示回顾窗口 输入时序定长
    engine_ids = list(training_data["engine"].unique())

    temp = training_data.copy().reset_index()
    for id_ in engine_ids:
        indexes = temp[temp["engine"] == id_].index
        traj_data = temp.loc[indexes]
        traj_data['cycle'] = np.arange(1, len(indexes) + 1, 1)
        cutoff_cycle = max(traj_data['cycle']) - sequence_length + 1
        # 把总cycle的切分位置的头尾定下来，然后开始步长为1的开始切分
        if cutoff_cycle <= 0:
            drop_range = indexes
            print("sequence_length + window_size is too large")
        else:
            cutoff_cycle_index = traj_data['cycle'][traj_data['cycle'] == cutoff_cycle + 1].index
            drop_range = list(range(cutoff_cycle_index[0], indexes[-1] + 1))

        temp.drop(drop_range, inplace=True)
    indexes = list(temp.index) #indexes表示一共能切出多少轮wz，切上面的操作就是保证每个id里面的切出来的wz不会多余
    del temp

    feature_number = training_data.shape[1] - 2

    x_shape = (len(indexes), sequence_length, feature_number)
    x_batch = np.zeros(shape=x_shape, dtype=np.float32)
    y_shape = (len(indexes), 1)
    y_batch = np.zeros(shape=y_shape, dtype=np.float32)

    for batch_index, index in enumerate(indexes):  # 0 是engine 1是rul
        y_batch[batch_index] = training_data.iloc[index + sequence_length-1:index + sequence_length, 1].values
        x_batch[batch_index] = training_data.iloc[index:index + sequence_length, 2:].values

    return x_batch, y_batch








def new_cmapss_data_train_vali_loader(data_path,
                  Data_id,
                  flag="train",
                  sequence_length=40,
                  MAXLIFE=100,
                  is_difference=False,
                  normalization="znorm",
                  validation=0.1):

    train_root = data_path + "/" + Data_id + "/" + 'train' + "/"
    test_root = data_path + "/" + Data_id + "/" + 'test' + "/"

    # Train dataset
    units = os.listdir(train_root)
    units = [i for i in units if i[-4:] == ".txt"]
    data, label = [], []
    seq_len = sequence_length
    idx = 1
    for i in units:
        rw = np.loadtxt(train_root + i)
        data_len = len(rw)
        max_rul = MAXLIFE
        lb = np.zeros(data_len)
        lb[::-1] = np.arange(data_len)
        for k in range(len(lb)):
            if lb[k] > max_rul:
                lb[k] = max_rul
        data_frame = pd.DataFrame(rw)
        data_frame['engine'] = idx
        data_frame['rul'] = lb
        idx+=1
        data.append(data_frame)

    train_FD = pd.concat(data)
    train_FD = train_FD.drop([28,29,30,31,32,33,35,36,37],axis=1)

    # TEST dataset
    units = os.listdir(test_root)
    units = [i for i in units if i[-4:] == ".txt"]
    data, label = [], []
    seq_len = sequence_length
    idx = 1
    for i in units:
        rw = np.loadtxt(test_root + i)
        data_len = len(rw)
        max_rul = MAXLIFE
        lb = np.zeros(data_len)
        lb[::-1] = np.arange(data_len)
        for k in range(len(lb)):
            if lb[k] > max_rul:
                lb[k] = max_rul
        test_data_frame = pd.DataFrame(rw)
        test_data_frame['engine'] = idx
        test_data_frame['rul'] = lb
        idx += 1
        data.append(test_data_frame)

    test_FD = pd.concat(data)
    test_FD = test_FD.drop([28, 29, 30, 31, 32, 33, 35, 36, 37], axis=1)


    scaler = MinMaxScaler()

    train_normalized = pd.DataFrame(train_FD.iloc[:,:-2])
    test_normalized = pd.DataFrame(test_FD.iloc[:,:-2])

    scaled_train = scaler.fit_transform(train_normalized)
    scaled_test = scaler.transform(test_normalized)

    train_normalized = pd.concat([train_FD.iloc[:,-2:],pd.DataFrame(data=scaled_train,index=train_normalized.index,columns=train_normalized.columns)],axis=1)
    test_normalized = pd.concat([test_FD.iloc[:,-2:],pd.DataFrame(data=scaled_test,index=test_normalized.index,columns=test_normalized.columns)],axis=1)





    if flag == "train":

        data_x, data_y = Cmapss_train_vali_batch_generator(train_normalized, sequence_length)
        X_train, X_vali, y_train, y_vali = train_test_split(data_x, data_y, test_size=validation, random_state=42)
        print("the shape of X_train is:", X_train.shape)
        print("the shape of y_train is:", y_train.shape)
        return X_train, y_train, X_vali, y_vali

    else:
        # process test_data
        data_x, data_y = Cmapss_test_batch_generator(test_normalized, sequence_length,is_diff=is_difference)

        return data_x, data_y




if __name__ == '__main__':
    new_cmapss_data_train_vali_loader(r'D:\用户目录\Desktop\实验\RUL\RUL_classify\N_CMAPSS','DS01')


