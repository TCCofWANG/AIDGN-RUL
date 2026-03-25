import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from scipy.stats import entropy

# this function used find features with poor correlation
def identify_and_remove_unique_columns(Dataframe):
    Dataframe = Dataframe.copy()
    del Dataframe["engine_id"]
    del Dataframe["cycle"]

    unique_counts = Dataframe.nunique()
    record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
        columns={'index': 'feature', 0: 'nunique'})
    unique_to_drop = list(record_single_unique['feature'])

    return unique_to_drop


def Cmapss_train_vali_batch_generator(training_data, sequence_length=15,is_diff=False, epsilon= 1e-06):
    """
    data generate for turbofan dataset
    Generator function for creating random batches of training-data
    """
    # sequence_lenth 表示回顾窗口 输入时序定长
    engine_ids = list(training_data["engine_id"].unique())


    if is_diff:
        sequence_length=sequence_length+1 #一次取多一个index
    # print(engine_ids)
    temp = training_data.copy()
    for id_ in engine_ids:
        indexes = temp[temp["engine_id"] == id_].index
        traj_data = temp.loc[indexes]
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

    feature_number = training_data.shape[1] - 3

    x_shape = (len(indexes), sequence_length, feature_number)
    x_batch = np.zeros(shape=x_shape, dtype=np.float32)
    x_basis = np.zeros(shape=(len(indexes), sequence_length, 1), dtype=np.float32)
    y_shape = (len(indexes), 1)
    y_batch = np.zeros(shape=y_shape, dtype=np.float32)

    for batch_index, index in enumerate(indexes):
        y_batch[batch_index] = training_data.iloc[index + sequence_length-1:index + sequence_length, -1].values
        x_batch[batch_index] = training_data.iloc[index:index + sequence_length, 2:-1].values
        x_basis[batch_index] = training_data.iloc[index:index + sequence_length, 1:2].values * epsilon

    return x_batch, y_batch, x_basis


def C_Cmapss_train_vali_batch_generator(training_data, sequence_length=15,is_diff=False, epsilon= 1e-06):
    """
    data generate for turbofan dataset
    Generator function for creating random batches of training-data
    """
    # sequence_lenth 表示回顾窗口 输入时序定长
    engine_ids = list(training_data["engine_id"].unique())


    if is_diff:
        sequence_length=sequence_length+1 #一次取多一个index
    # print(engine_ids)
    temp = training_data.copy()
    for id_ in engine_ids:
        indexes = temp[temp["engine_id"] == id_].index
        traj_data = temp.loc[indexes]
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

    feature_number = training_data.shape[1] - 3

    x_shape = (len(indexes), sequence_length, feature_number)
    x_batch = np.zeros(shape=x_shape, dtype=np.float32)
    x_basis = np.zeros(shape=(len(indexes), sequence_length, 1), dtype=np.float32)
    y_shape = (len(indexes), 1)
    y_batch = np.zeros(shape=y_shape, dtype=np.float32)

    for batch_index, index in enumerate(indexes):
        y_batch[batch_index] = training_data.iloc[index + sequence_length-1:index + sequence_length, -1].values
        x_batch[batch_index] = training_data.iloc[index:index + sequence_length, 2:-1].values
        x_basis[batch_index] = training_data.iloc[index:index + sequence_length, 1].values * epsilon

    return x_batch, y_batch, x_basis


def Cmapss_test_batch_generator(test_data, sequence_length=5,is_diff=False, epsilon= 1e-06):
    engine_ids = list(test_data["engine_id"].unique())
    if is_diff:
        sequence_length=sequence_length+1 #一次取多一个index

    feature_number = test_data.shape[1] - 3

    x_batch = []
    x_basis = []
    y_batch = []
    for _id in set(test_data['engine_id']):
        test_of_one_id = test_data[test_data['engine_id'] == _id]
        #每个引擎id只取最末尾的一个wz，不切，相当于test_date就去wz的长度
        if test_of_one_id.shape[0] >= sequence_length:
            x_batch.append(test_of_one_id.iloc[-sequence_length:, 2:-1].values)
            x_basis.append(test_of_one_id.iloc[-sequence_length:, 1:2].values * epsilon)
            y_batch.append(test_of_one_id.iloc[-1:, -1].values)

        else:
            num_pad = sequence_length - len(test_of_one_id)
            new_sg = test_of_one_id
            for idx in range(num_pad):
                new_sg = pd.concat([new_sg.head(1), new_sg], axis=0)

            x_batch.append(new_sg.iloc[:,2:-1].values)
            y_batch.append(new_sg.iloc[-1:, -1].values)
            x_basis.append(new_sg.iloc[:, 1:2].values * epsilon)

    return np.array(x_batch), np.array(y_batch), np.array(x_basis),


# 计算difference
def cal_diff(df, sensor_name, diff_periods=1):
    sensor_diff = []

    for _id in set(df['engine_id']):
        trainFD001_of_one_id = df[df['engine_id'] == _id]
        s = pd.Series(trainFD001_of_one_id[sensor_name])

        if len(s) > diff_periods:
            sensor_diff_temp = s.diff(periods=diff_periods)

            for i in range(diff_periods):
                sensor_diff.append(s.iloc[i] - s.iloc[0])

            for j in range(len(s) - diff_periods):
                sensor_diff.append(sensor_diff_temp.iloc[diff_periods + j])
        else:
            for h in range(len(s)):
                sensor_diff.append(s.iloc[h] - s.iloc[0])
    return sensor_diff


# load and process training&validation or test data
def hc_cmapss_data_train_vali_loader(data_path,
                  Data_id,
                  flag="train",
                  sequence_length=40,
                  MAXLIFE=120,
                  is_difference=False,
                  normalization="znorm",
                  validation=0.1):
    # --------------- read the train data, test data and labels for test ---------------

    column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']

    # 这里train test FD 的区别在于test缺少真实的寿命结果
    train_FD = pd.read_table("{}/train_{}.txt".format(data_path, Data_id), header=None, delim_whitespace=True)
    train_FD.columns = column_name

    test_FD = pd.read_table("{}/test_{}.txt".format(data_path, Data_id), header=None, delim_whitespace=True)
    test_FD.columns = column_name

    # test对应的真实寿命标签
    RUL_FD = pd.read_table("{}/RUL_{}.txt".format(data_path, Data_id), header=None, delim_whitespace=True)


    # ---------------- difference ------------------------
    # 默认不作这项操作
    # if difference:
    #     # 因为第一列和第二列表征id和cycle
    #     diff_columns = train_FD.columns[2:]
    #     for i in range(len(diff_columns)):
    #         sensor_name_temp = diff_columns[i]
    #         diff = cal_diff(train_FD, sensor_name=sensor_name_temp)
    #         name = sensor_name_temp + '_diff'
    #         train_FD[name] = diff
    #     for i in range(len(diff_columns)):
    #         sensor_name_temp = diff_columns[i]
    #         diff = cal_diff(test_FD, sensor_name=sensor_name_temp)
    #         name = sensor_name_temp + '_diff'
    #         test_FD[name] = diff


    # --------------- define the label for train and test ---------------
    # piecewise linear RUL  for Training data
    # MAXLIFE define after this timepoint, the work time is RUL

    id = 'engine_id'
    rul = []
    healthy_labels = []
    for _id in set(train_FD[id]):
        trainFD_of_one_id = train_FD[train_FD[id] == _id]
        cycle_list = trainFD_of_one_id['cycle'].tolist()
        max_cycle = max(cycle_list)
        # 限制最大寿命为MAXLIFE，也就是说cycle超出这个限制的，都被设置为MAXLIFE
        knee_point = max_cycle - MAXLIFE
        kink_RUL = []
        healthy_label = []
        for i in range(0, len(cycle_list)):
            #
            if i < knee_point:
                kink_RUL.append(MAXLIFE)
                healthy_label.append(0)
            else:
                tmp = max_cycle - i - 1
                kink_RUL.append(tmp)
                healthy_label.append(1)
        rul.extend(kink_RUL)
        healthy_labels.extend(healthy_label)
    #根据MAXLIFE 重新设置数据集的RUL
    train_FD["RUL"] = rul
    train_FD["h_label"] = healthy_labels
    # piecewise linear RUL  for Test data


    # train 和 test的本质处理其实是一样的
    id = 'engine_id'
    rul = []
    for _id_test in set(test_FD[id]):
        true_rul = int(RUL_FD.iloc[_id_test - 1])
        testFD_of_one_id = test_FD[test_FD[id] == _id_test]
        cycle_list = testFD_of_one_id['cycle'].tolist()
        max_cycle = max(cycle_list) + true_rul
        knee_point = max_cycle - MAXLIFE
        kink_RUL = []
        for i in range(0, len(cycle_list)):
            if i < knee_point:
                kink_RUL.append(MAXLIFE)
            else:
                tmp = max_cycle - i - 1
                kink_RUL.append(tmp)

        rul.extend(kink_RUL)

    test_FD["RUL"] = rul


    # --------------- acoording to the labels of training dataset, delete redundant input sensors ---------------
    # drop 掉没有用的feature 普遍因为是全是一个值没有考虑价值
    col_to_drop = identify_and_remove_unique_columns(train_FD)
    # 别人的code保留的传感器
    # ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

    # 强制去除固定的特征
    col_to_drop = ['setting3', 's1', 's5', 's6', 's10', 's16', 's18', 's19']#['setting2','setting3','s1','s5','s6','s10','s16','s18','s19']
    print(col_to_drop)
    train_FD = train_FD.drop(col_to_drop, axis=1)
    test_FD = test_FD.drop(col_to_drop, axis=1)


    # ---------------- Normalization --------------------------------

    if normalization == "znorm":
        mean = train_FD.iloc[:, 2:-2].mean()
        std  = train_FD.iloc[:, 2:-2].std()
        std.replace(0, 1, inplace=True)

        # training dataset
        train_FD.iloc[:, 2:-2] = (train_FD.iloc[:, 2:-2] - mean) / std

        # Testing dataset
        test_FD.iloc[:, 2:-1] = (test_FD.iloc[:, 2:-1] - mean) / std
    if normalization == "minmax":
        min_ = train_FD.iloc[:, 2:-2].min()
        max_ = train_FD.iloc[:, 2:-2].max()
        dis  = max_ - min_
        dis.replace(0, 1, inplace=True)
        # 把0替换为1 是为了避免分母为0 待会出的时候报错
        # training dataset
        train_FD.iloc[:, 2:-2] = (train_FD.iloc[:, 2:-2] - min_) / dis

        # Testing dataset
        test_FD.iloc[:, 2:-1] = (test_FD.iloc[:, 2:-1] - min_) / dis


    # ------------------- batch generator -------------------------------

    if flag == "train":

        data_x, data_y, basis_x = Cmapss_train_vali_batch_generator(train_FD, sequence_length,is_diff=is_difference)
        X_train, X_vali, y_train, y_vali = train_test_split(data_x, data_y, test_size=validation, random_state=42)
        print("the shape of X_train is:", X_train.shape)
        print("the shape of y_train is:", y_train.shape)
        return X_train, y_train, X_vali, y_vali

    else:
        # process test_data
        data_x, data_y, basis_x, = Cmapss_test_batch_generator(test_FD, sequence_length,is_diff=is_difference)

        return data_x, data_y, basis_x

def normalize_min_max(train_FD):
    min_ = train_FD.iloc[:, 2:-1].min()
    max_ = train_FD.iloc[:, 2:-1].max()
    dis = max_ - min_
    dis.replace(0, 1, inplace=True)
    # 把0替换为1 是为了避免分母为0 待会出的时候报错
    # training dataset
    train_FD.iloc[:, 2:-1] = (train_FD.iloc[:, 2:-1] - min_) / dis

    return train_FD,min_,max_,dis

def kl_divergence(p, q):
    """计算KL散度"""
    p = p / np.sum(p)  # 归一化
    q = q / np.sum(q)  # 归一化
    return entropy(p, q)

def get_global_input(train_values , sequence_length, type='mean'):

    min_num, max_num = np.min(train_values[:, 0]), np.max(train_values[:, 0])
    sample = []
    for i in range(int(min_num.tolist()), int((max_num+1).tolist())):
        sample.append(train_values[train_values[:, 0]==i])
    window_size = 60

    safe_sample = []
    for data in sample:
        # 用于存储KL散度的变化
        kl_changes = []

        # 滑动窗口计算KL散度
        for i in range(len(data) - 2 * window_size + 1):
            window1 = data[i:i + window_size]
            window2 = data[i + window_size:i + 2 * window_size]

            # 将每个窗口的分布归一化并计算KL散度
            hist1, bin_edges1 = np.histogram(window1, bins=10, density=True)
            hist2, bin_edges2 = np.histogram(window2, bins=10, density=True)

            kl = kl_divergence(hist1, hist2)
            kl_changes.append(kl)

        # 找到变化最大的KL散度点作为分隔点
        kl_changes = np.array(kl_changes)
        change_point = np.argmax(kl_changes) + window_size  # 由于滑动窗口的偏移，实际位置需要加上窗口大小

        # 获取分割后的两段数据
        if change_point>=sequence_length:
            change_point = sequence_length
            segment = data[:change_point]
        else:
            segment = np.pad(data[:change_point], ((0, sequence_length-change_point), (0, 0)), mode='constant', constant_values=(0, 0))

        length, sensor = segment.shape
        safe_sample.append(segment.reshape((1, length, sensor)))

    if type == 'mean':
        global_sample = np.mean(np.concatenate(safe_sample, axis=0), axis=0)
        return global_sample[:, 2:-1]
    else:
        global_sample = np.concatenate(safe_sample, axis=0)
        return global_sample[:, :, 2:-1]



def get_cmapss_data_(data_path,
                  Data_id,
                  flag="train",
                  sequence_length=40,
                  MAXLIFE=120,
                  is_difference=False,
                  normalization="znorm",
                  validation=0.1, epsilon= 1e-06):
    # --------------- read the train data, test data and labels for test ---------------

    column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']

    # 这里train test FD 的区别在于test缺少真实的寿命结果
    train_FD = pd.read_table("{}/train_{}.txt".format(data_path, Data_id), header=None, delim_whitespace=True)
    train_FD.columns = column_name

    test_FD = pd.read_table("{}/test_{}.txt".format(data_path, Data_id), header=None, delim_whitespace=True)
    test_FD.columns = column_name


    # test对应的真实寿命标签
    RUL_FD = pd.read_table("{}/RUL_{}.txt".format(data_path, Data_id), header=None, delim_whitespace=True)

    id = 'engine_id'
    rul = []

    for _id in set(train_FD[id]):
        trainFD_of_one_id = train_FD[train_FD[id] == _id]
        cycle_list = trainFD_of_one_id['cycle'].tolist()
        max_cycle = max(cycle_list)
        # 限制最大寿命为MAXLIFE，也就是说cycle超出这个限制的，都被设置为MAXLIFE
        knee_point = max_cycle - MAXLIFE
        kink_RUL = []

        for i in range(0, len(cycle_list)):
            #
            if i < knee_point:
                kink_RUL.append(MAXLIFE)

            else:
                tmp = max_cycle - i - 1
                kink_RUL.append(tmp)

        rul.extend(kink_RUL)

    #根据MAXLIFE 重新设置数据集的RUL
    train_FD["RUL"] = rul

    # piecewise linear RUL  for Test data


    # train 和 test的本质处理其实是一样的
    id = 'engine_id'
    rul = []
    for _id_test in set(test_FD[id]):
        true_rul = int(RUL_FD.iloc[_id_test - 1])
        testFD_of_one_id = test_FD[test_FD[id] == _id_test]
        cycle_list = testFD_of_one_id['cycle'].tolist()
        max_cycle = max(cycle_list) + true_rul
        knee_point = max_cycle - MAXLIFE
        kink_RUL = []
        for i in range(0, len(cycle_list)):
            if i < knee_point:
                kink_RUL.append(MAXLIFE)
            else:
                tmp = max_cycle - i - 1
                kink_RUL.append(tmp)

        rul.extend(kink_RUL)

    test_FD["RUL"] = rul


    # --------------- acoording to the labels of training dataset, delete redundant input sensors ---------------
    # 别人的code保留的传感器
    # ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

    # 强制去除固定的特征
    col_to_drop = ['setting3', 's1', 's5', 's6', 's10', 's16', 's18', 's19']
    train_FD = train_FD.drop(col_to_drop, axis=1)
    test_FD = test_FD.drop(col_to_drop, axis=1)

    train_FD['setting1'] = train_FD['setting1'].round(1)
    test_FD['setting1'] = test_FD['setting1'].round(1)

    grouped_train = train_FD.groupby('setting1')
    grouped_test = test_FD.groupby('setting1')

    # ---------------- Normalization --------------------------------
    scaler = MinMaxScaler()

    train_normalized = pd.DataFrame(columns=train_FD.columns[3:])
    test_normalized = pd.DataFrame(columns=test_FD.columns[3:])

    for train_idx, train in grouped_train:

        scaled_train = scaler.fit_transform(train.iloc[:, 3:])
        scaled_train_combine = pd.DataFrame(
            data=scaled_train,
            index=train.index,
            columns=train_FD.columns[3:])
        train_normalized = pd.concat([train_normalized, scaled_train_combine])

        for test_idx, test in grouped_test:
            if train_idx == test_idx:
                scaled_test = scaler.transform(test.iloc[:, 3:])
                scaled_test_combine = pd.DataFrame(
                    data=scaled_test,
                    index=test.index,
                    columns=test_FD.columns[3:])
                test_normalized = pd.concat([test_normalized, scaled_test_combine])


    train_normalized = train_normalized.sort_index()
    test_normalized = test_normalized.sort_index()


    train_FD.iloc[:, 3:-1]  = train_normalized.iloc[:,:]
    test_FD.iloc[:, 3:-1]  = test_normalized.iloc[:,:]


    # train_FD = train_FD.drop('setting1', axis=1)
    # test_FD = test_FD.drop('setting1', axis=1)

    # ------------------- batch generator -------------------------------
    global_input = get_global_input(train_FD.values, sequence_length, 'concat')

    data_x, data_y, basis_x = Cmapss_train_vali_batch_generator(train_FD, sequence_length,is_diff=is_difference)
    X_train, X_vali, y_train, y_vali, B_train, B_valid = train_test_split(data_x, data_y, basis_x, test_size=validation, random_state=42)
    print("the shape of X_train is:", X_train.shape)
    print("the shape of y_train is:", y_train.shape)
    print("the shape of B_train is:", B_train.shape)

    # basis_index = self.indexes[i]
    # index_list = np.arange(basis_index,index+(self.sequence_len),1)
    ## normalize the index
    norm_index = train_normalized * epsilon

    # process test_data
    data_x, data_y, basis_x = Cmapss_test_batch_generator(test_FD, sequence_length,is_diff=is_difference)

    return X_train, y_train, B_train, X_vali, y_vali, B_valid, data_x, data_y, basis_x, global_input


def da_get_cmapss_data_(data_path,
                   s_id,t_id,
                  sequence_length=40,
                  MAXLIFE=120,
                  is_difference=False,
                  normalization="znorm",
                  validation=0.1):
    # --------------- read the train data, test data and labels for test ---------------

    column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']

    # 这里train test FD 的区别在于test缺少真实的寿命结果
    s_train_FD = pd.read_table("{}/train_{}.txt".format(data_path, s_id), header=None, delim_whitespace=True)
    t_train_FD = pd.read_table("{}/train_{}.txt".format(data_path, t_id), header=None, delim_whitespace=True)

    s_train_FD.columns = column_name
    t_train_FD.columns = column_name

    test_FD = pd.read_table("{}/test_{}.txt".format(data_path, t_id), header=None, delim_whitespace=True)
    test_FD.columns = column_name

    # test对应的真实寿命标签
    RUL_FD = pd.read_table("{}/RUL_{}.txt".format(data_path, t_id), header=None, delim_whitespace=True)

    id = 'engine_id'
    s_rul = []

    for _id in set(s_train_FD[id]):
        trainFD_of_one_id = s_train_FD[s_train_FD[id] == _id]
        cycle_list = trainFD_of_one_id['cycle'].tolist()
        max_cycle = max(cycle_list)
        # 限制最大寿命为MAXLIFE，也就是说cycle超出这个限制的，都被设置为MAXLIFE
        knee_point = max_cycle - MAXLIFE
        kink_RUL = []

        for i in range(0, len(cycle_list)):
            #
            if i < knee_point:
                kink_RUL.append(MAXLIFE)

            else:
                tmp = max_cycle - i - 1
                kink_RUL.append(tmp)

        s_rul.extend(kink_RUL)

    #根据MAXLIFE 重新设置数据集的RUL
    s_train_FD["RUL"] = s_rul


    id = 'engine_id'
    t_rul = []

    for _id in set(t_train_FD[id]):
        trainFD_of_one_id = t_train_FD[t_train_FD[id] == _id]
        cycle_list = trainFD_of_one_id['cycle'].tolist()
        max_cycle = max(cycle_list)
        # 限制最大寿命为MAXLIFE，也就是说cycle超出这个限制的，都被设置为MAXLIFE
        knee_point = max_cycle - MAXLIFE
        kink_RUL = []

        for i in range(0, len(cycle_list)):
            #
            if i < knee_point:
                kink_RUL.append(MAXLIFE)

            else:
                tmp = max_cycle - i - 1
                kink_RUL.append(tmp)

        t_rul.extend(kink_RUL)

    # 根据MAXLIFE 重新设置数据集的RUL
    t_train_FD["RUL"] = t_rul

    # piecewise linear RUL  for Test data

    # train 和 test的本质处理其实是一样的
    id = 'engine_id'
    rul = []
    for _id_test in set(test_FD[id]):
        true_rul = int(RUL_FD.iloc[_id_test - 1])
        testFD_of_one_id = test_FD[test_FD[id] == _id_test]
        cycle_list = testFD_of_one_id['cycle'].tolist()
        max_cycle = max(cycle_list) + true_rul
        knee_point = max_cycle - MAXLIFE
        kink_RUL = []
        for i in range(0, len(cycle_list)):
            if i < knee_point:
                kink_RUL.append(MAXLIFE)
            else:
                tmp = max_cycle - i - 1
                kink_RUL.append(tmp)

        rul.extend(kink_RUL)

    test_FD["RUL"] = rul


    # --------------- acoording to the labels of training dataset, delete redundant input sensors ---------------
    # 别人的code保留的传感器
    # ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

    # 强制去除固定的特征['setting3', 's1', 's5', 's6', 's10', 's16', 's18', 's19'] , xb ['setting2','setting3','s1','s5','s6','s10','s16','s18','s19']
    col_to_drop = ['setting3', 's1', 's5', 's6', 's10', 's16', 's18', 's19']
    s_train_FD = s_train_FD.drop(col_to_drop, axis=1)
    t_train_FD = t_train_FD.drop(col_to_drop, axis=1)
    test_FD = test_FD.drop(col_to_drop, axis=1)

    s_train_FD['setting1'] = s_train_FD['setting1'].round(1)
    t_train_FD['setting1'] = t_train_FD['setting1'].round(1)
    test_FD['setting1'] = test_FD['setting1'].round(1)

    s_grouped_train = s_train_FD.groupby('setting1')
    t_grouped_train = t_train_FD.groupby('setting1')
    grouped_test = test_FD.groupby('setting1')

    # ---------------- Normalization --------------------------------
    scaler = MinMaxScaler()

    s_train_normalized = pd.DataFrame(columns=s_train_FD.columns[3:])
    test_normalized = pd.DataFrame(columns=test_FD.columns[3:])

    for train_idx, train in s_grouped_train:

        scaled_train = scaler.fit_transform(train.iloc[:, 3:])
        scaled_train_combine = pd.DataFrame(
            data=scaled_train,
            index=train.index,
            columns=s_train_FD.columns[3:])
        s_train_normalized = pd.concat([s_train_normalized, scaled_train_combine])

        for test_idx, test in grouped_test:
            if train_idx == test_idx:
                scaled_test = scaler.transform(test.iloc[:, 3:])
                scaled_test_combine = pd.DataFrame(
                    data=scaled_test,
                    index=test.index,
                    columns=test_FD.columns[3:])
                test_normalized = pd.concat([test_normalized, scaled_test_combine])

    train_normalized = s_train_normalized.sort_index()
    test_normalized = test_normalized.sort_index()


    s_train_FD.iloc[:, 3:-1]  = train_normalized.iloc[:,:]
    test_FD.iloc[:, 3:-1]  = test_normalized.iloc[:,:]

    # s_train_FD = s_train_FD.drop('setting1', axis=1)
    # test_FD = test_FD.drop('setting1', axis=1)

    # ------------------- batch generator -------------------------------
    # TODO 由于我们不使用domain方法 source和target一致，所以使用取巧方法
    t_train_FD = s_train_FD

    s_data_x, s_data_y = C_Cmapss_train_vali_batch_generator(s_train_FD, sequence_length,is_diff=is_difference)
    t_data_x, t_data_y = C_Cmapss_train_vali_batch_generator(t_train_FD, sequence_length,is_diff=is_difference)

    s_x_train, s_x_vali, s_y_train, s_y_vali = train_test_split(s_data_x, s_data_y, test_size=validation, random_state=42)
    t_x_train, t_x_vali, t_y_train, t_y_vali = train_test_split(t_data_x, t_data_y, test_size=validation, random_state=42)

    print("the shape of X_train is:", t_x_train.shape)
    print("the shape of y_train is:", t_y_train.shape)

    data_x, data_y = Cmapss_test_batch_generator(test_FD, sequence_length, is_diff=is_difference)


    return s_x_train, t_x_train, s_x_vali, t_x_vali, t_y_train.squeeze(-1), t_y_vali.squeeze(-1), data_x, data_y



def cmapss_data_train_Classify(X_train,Y_train):
    B,L,D = X_train.shape
    X_train_re = X_train.reshape((B*L,D))
    Y_train_re = Y_train.reshape((B*L,2))
    h_indices = np.where(Y_train_re[:,1]==0)
    b_indices = np.where(Y_train_re[:,1]==1)

    h_num = h_indices[0].shape[0]
    b_num = b_indices[0].shape[0]

    b_indices_limited = b_indices[0][:h_num]
    h_indices = h_indices[0]

    new_indices = np.concatenate((h_indices,b_indices_limited))
    limit_len = new_indices.size//L  * L

    new_indices = new_indices[:limit_len]

    X_train_c = X_train_re[new_indices,:]
    Y_train_c = Y_train_re[new_indices,:]


    X_train_c = X_train_c.reshape((-1,L,D))
    Y_train_c = Y_train_c.reshape((-1,L,2))

    return X_train_c,Y_train_c
