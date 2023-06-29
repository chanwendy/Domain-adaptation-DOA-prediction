import os
import numpy as np
import dataset.data_act as data_act
import pandas as pd
import dataset.data_weather as data_weather
import datetime
from ourmodel.loss_transfer import TransferLoss
import torch
import math
from dataset import data_process
from torch.utils.data import Dataset, DataLoader


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = data.shape[1]
    cols, names = list(), list()
    dataclass = data['BIS']
    # drop删除数据的某一行或则列，axis=1表示按列，axis=0表示按行 去掉序列号和标签
    data = data.drop(columns=['BIS'], axis=1)
    # 获取到data中的特征
    columns = data.columns
    # 移动sequence
    # input sequence (t-n, ... t-1)  #non arrivo all'osservazione corrente
    for i in range(n_in - 1, 0, -1):
        cols.append(data.shift(i))
        names += [(element + '(t-%d)' % (i)) for element in columns]

    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [(element + '(t)') for element in columns]
        else:
            names += [(element + '(t+%d)' % (i)) for element in columns]
    # 增加一个class类
    cols.append(dataclass)  # appendo le ultime cinque colonne
    names += ['BIS']

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # dropnan 删除数据中的NAN，按行删除
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def get_data(data_file, ):
    # data = torch.from_numpy(np.array(pd.read_csv(data_file,usecols=["RFTN20_VOL", "PPF20_VOL", "age", "sex", "weight", "height"])))
    data = pd.read_csv(data_file, usecols=["RFTN20_VOL", "PPF20_VOL", "BIS"])
    sequence_data = series_to_supervised(data, n_in=180, n_out=1, dropnan=True)
    times = len(sequence_data)
    label = torch.from_numpy(np.array(sequence_data["BIS"]))
    sequence_data = torch.from_numpy(np.array(sequence_data.drop(columns=['BIS'], axis=1))).view(-1, 180, 2)
    # label = torch.from_numpy(np.array(pd.read_csv(data_file,usecols=["BIS"]))).view(-1)
    return sequence_data, label, times


def TDC(num_domain, data_file, dis_type='coral'):
    start_time = 0
    # 要分的区间段
    split_N = 10
    # 读取数据文件
    data, label, num_times = get_data(data_file)
    # 提取数据
    feat = data[0:num_times]
    feat = torch.tensor(feat, dtype=torch.float32)

    feat_shape_1 = feat.shape[1]
    feat = feat.reshape(-1, feat.shape[2])
    feat = feat.cuda()
    # num_day_new = feat.shape[0]

    selected = [0, 10]
    candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    start = 0
    # 贪婪算法划分区间K
    if num_domain in [2, 3, 5, 7, 10]:
        while len(selected) - 2 < num_domain - 1:
            distance_list = []
            for can in candidate:
                selected.append(can)
                selected.sort()
                dis_temp = 0
                for i in range(1, len(selected) - 1):
                    for j in range(i, len(selected) - 1):
                        index_part1_start = start + math.floor(selected[i - 1] / split_N * num_times) * feat_shape_1
                        index_part1_end = start + math.floor(selected[i] / split_N * num_times) * feat_shape_1
                        feat_part1 = feat[index_part1_start: index_part1_end]
                        index_part2_start = start + math.floor(selected[j] / split_N * num_times) * feat_shape_1
                        index_part2_end = start + math.floor(selected[j + 1] / split_N * num_times) * feat_shape_1
                        feat_part2 = feat[index_part2_start:index_part2_end]
                        criterion_transder = TransferLoss(loss_type=dis_type, input_dim=feat_part1.shape[1])
                        dis_temp += criterion_transder.compute(feat_part1, feat_part2)
                        # dis_temp += CORAL(feat_part1, feat_part2)
                distance_list.append(dis_temp)
                selected.remove(can)
            can_index = distance_list.index(max(distance_list))
            selected.append(candidate[can_index])
            candidate.remove(candidate[can_index])
        selected.sort()
        res = []
        for i in range(1, len(selected)):
            if i == 1:
                sel_start_time = start_time + num_times / split_N * selected[i - 1]
            else:
                sel_start_time = start_time + num_times / split_N * selected[i - 1] + 1

            sel_end_time = start_time + num_times / split_N * selected[i]
            res.append((sel_start_time, sel_end_time))
        return res
    else:
        print("error in number of domain")

def new_TDC(num_domain,data, label, num_times, dis_type='coral'):
    start_time = 0
    # 要分的区间段
    split_N = 10
    # 读取数据文件
    # data, label, num_times = get_data(data_file)
    # 提取数据
    feat = data[0:num_times]
    feat = torch.tensor(feat, dtype=torch.float32)

    feat_shape_1 = feat.shape[1]
    feat = feat.reshape(-1, feat.shape[2])
    feat = feat.cuda()
    # num_day_new = feat.shape[0]

    selected = [0, 10]
    candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    start = 0
    # 贪婪算法划分区间K
    if num_domain in [2, 3, 5, 7, 10]:
        while len(selected) - 2 < num_domain - 1:
            distance_list = []
            for can in candidate:
                selected.append(can)
                selected.sort()
                dis_temp = 0
                for i in range(1, len(selected) - 1):
                    for j in range(i, len(selected) - 1):
                        index_part1_start = start + math.floor(selected[i - 1] / split_N * num_times) * feat_shape_1
                        index_part1_end = start + math.floor(selected[i] / split_N * num_times) * feat_shape_1
                        feat_part1 = feat[index_part1_start: index_part1_end]
                        index_part2_start = start + math.floor(selected[j] / split_N * num_times) * feat_shape_1
                        index_part2_end = start + math.floor(selected[j + 1] / split_N * num_times) * feat_shape_1
                        feat_part2 = feat[index_part2_start:index_part2_end]
                        criterion_transder = TransferLoss(loss_type=dis_type, input_dim=feat_part1.shape[1])
                        dis_temp += criterion_transder.compute(feat_part1, feat_part2)
                        # dis_temp += CORAL(feat_part1, feat_part2)
                distance_list.append(dis_temp)
                selected.remove(can)
            can_index = distance_list.index(max(distance_list))
            selected.append(candidate[can_index])
            candidate.remove(candidate[can_index])
        selected.sort()
        res = []
        for i in range(1, len(selected)):
            if i == 1:
                sel_start_time = start_time + num_times / split_N * selected[i - 1]
            else:
                sel_start_time = start_time + num_times / split_N * selected[i - 1] + 1

            sel_end_time = start_time + num_times / split_N * selected[i]
            res.append((sel_start_time, sel_end_time))
        return res
    else:
        print("error in number of domain")



class data_loader(Dataset):
    def __init__(self, df_feature, df_label_reg, t=None):
        assert len(df_feature) == len(df_label_reg)
        # df_feature = df_feature.reshape(df_feature.shape[0], df_feature.shape[1] // 6, df_feature.shape[2] * 6)
        self.df_feature = df_feature
        self.df_label_reg = df_label_reg
        self.T = t
        self.df_feature = torch.tensor(
            self.df_feature, dtype=torch.float32)
        self.df_label_reg = torch.tensor(
            self.df_label_reg, dtype=torch.float32)

    def __getitem__(self, index):
        sample, label_reg = self.df_feature[index], self.df_label_reg[index]
        return sample, label_reg

    def __len__(self):
        return len(self.df_feature)


def create_dataset(data_file, start_date, end_date, mean=None, std=None):
    feat, label_reg, times = get_data(data_file)
    referece_start_time = 0
    referece_end_time = 62580
    if end_date == 62580:
        end_date = times

    assert ((start_date) - referece_start_time) >= 0
    # assert ((end_date) - referece_end_time) <= 0
    assert ((end_date) - (start_date)) >= 0
    # 因为不同区间段的起始点不是一致的，所以需要将不同区间段的起始点减去最初起始点  这里表示啥？
    index_start = int(((start_date) - referece_start_time))
    # 不同区间段的末尾点减去起始点 表示数据的长度
    index_end = int(((end_date) - referece_start_time))
    feat = feat[index_start: index_end + 1]
    label_reg = label_reg[index_start: index_end + 1]

    # ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
    # feat=feat.reshape(-1, feat.shape[2])
    # feat=(feat - mean) / std
    # feat=feat.reshape(-1, ori_shape_1, ori_shape_2)

    return data_loader(feat, label_reg)


def get_weather_data(data_file, start_time, end_time, batch_size, shuffle=True, mean=None, std=None):
    dataset = create_dataset(data_file, start_time, end_time, mean=mean, std=std)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        return files  # 当前路径下所有非目录子文件,列表

# for one person
def load_weather_data_multi_domain2(train_file_path,valid_file_path, test_file_path, batch_size=6, number_domain=2,
                                   mode='pre_process', dis_type='coral'):
    # mode: 'tdc', 'pre_process'
    # data_file = os.path.join(file_path, "PRSA_Data_1.pkl")
    train_file_list = file_name(train_file_path)
    # print(train_file_list)
    valid_file_list = file_name(valid_file_path)
    test_file_list = file_name(test_file_path)
    train_list = []
    valid_list = []
    test_list = []
    for i in range(len(train_file_list)):
        # train_data_file = f"/home/user02/TUTMING/ming/adarnn/dataset/data/train/{train_file_list[j]}"
        train_data_file = os.path.join(train_file_path, train_file_list[i])
        split_time_list = TDC(number_domain, data_file=train_data_file, dis_type=dis_type)
        for j in range(len(split_time_list)):
            time_temp = split_time_list[j]
            train_loader = get_weather_data(train_data_file, start_time=time_temp[0], end_time=time_temp[1],
                                            batch_size=batch_size, )
            train_list.append(train_loader)
    for i in range(len(valid_file_list)):
        valid_data_file = os.path.join(valid_file_path, valid_file_list[i])
        valid_vld_loader = get_weather_data(valid_data_file, start_time=0, end_time=62580, batch_size=batch_size, )
        valid_list.append(valid_vld_loader)
    for i in range(len(test_file_list)):
        test_data_file = os.path.join(test_file_path, test_file_list[i])
        test_loader = get_weather_data(test_data_file, start_time=0, end_time=62580, batch_size=batch_size,
                                       shuffle=False)
        test_list.append(test_loader)
    return train_list, valid_list, test_list, valid_file_list, test_file_list






def load_weather_data_multi_domain(valid_file_path, test_file_path, batch_size=6, number_domain=2,
                                   mode='pre_process', dis_type='coral'):
    # mode: 'tdc', 'pre_process'
    # data_file = os.path.join(file_path, "PRSA_Data_1.pkl")
    # train_file_list = file_name(train_file_path)
    # print(train_file_list)
    valid_file_list = file_name(valid_file_path)
    test_file_list = file_name(test_file_path)
    # train_list = []
    valid_list = []
    test_list = []
    # for i in range(len(train_file_list)):
    #     # train_data_file = f"/home/user02/TUTMING/ming/adarnn/dataset/data/train/{train_file_list[j]}"
    #     train_data_file = os.path.join(train_file_path, train_file_list[i])
    #     split_time_list = TDC(number_domain, data_file=train_data_file, dis_type=dis_type)
    #     for j in range(len(split_time_list)):
    #         time_temp = split_time_list[j]
    #         train_loader = get_weather_data(train_data_file, start_time=time_temp[0], end_time=time_temp[1],
    #                                         batch_size=batch_size, )
    #         train_list.append(train_loader)
    for i in range(len(valid_file_list)):
        valid_data_file = os.path.join(valid_file_path, valid_file_list[i])
        valid_vld_loader = get_weather_data(valid_data_file, start_time=0, end_time=62580, batch_size=batch_size, )
        valid_list.append(valid_vld_loader)
    for i in range(len(test_file_list)):
        test_data_file = os.path.join(test_file_path, test_file_list[i])
        test_loader = get_weather_data(test_data_file, start_time=0, end_time=62580, batch_size=batch_size,
                                       shuffle=False)
        test_list.append(test_loader)
    return valid_list, test_list


def load_train_data(train_file_path, epoch=0, people_size=1, batch_size=6, number_domain=2, mode='pre_process', dis_type='coral'):
    train_file_list = file_name(train_file_path)
    train_list = []
    flag = int(60 / people_size)
    if epoch >= flag:
        epoch = epoch % flag
    for i in range(people_size):
        file_id = i + epoch * people_size
        # train_data_file = f"/home/user02/TUTMING/ming/adarnn/dataset/data/train/{train_file_list[j]}"
        train_data_file = os.path.join(train_file_path, train_file_list[file_id])
        split_time_list = TDC(number_domain, data_file=train_data_file, dis_type=dis_type)
        for j in range(len(split_time_list)):
            time_temp = split_time_list[j]
            train_loader = get_weather_data(train_data_file, start_time=time_temp[0], end_time=time_temp[1],
                                            batch_size=batch_size, )
            train_list.append(train_loader)
    return train_list


def load_norm_train_data(train_file_path, epoch=0, people_size=1, batch_size=6, number_domain=2, mode='pre_process', dis_type='coral'):
    train_file_list = file_name(train_file_path)
    train_list = []
    flag = int(60 / people_size)
    if epoch >= flag:
        epoch = epoch % flag
    for i in range(people_size):
        file_id = i + epoch * people_size
        # train_data_file = f"/home/user02/TUTMING/ming/adarnn/dataset/data/train/{train_file_list[j]}"
        train_data_file = os.path.join(train_file_path, train_file_list[file_id])
        # split_time_list = TDC(number_domain, data_file=train_data_file, dis_type=dis_type)
        train_loader = get_weather_data(train_data_file, start_time=0, end_time=62580,batch_size=batch_size,)
        train_list.append(train_loader)
    return train_list

def load_sum_train_data(train_file_path, epoch=0, people_size=1, batch_size=6, number_domain=2, mode='pre_process', dis_type='coral',):
    train_file_list = file_name(train_file_path)
    train_list = []
    for i in range(people_size):
        # train_data_file = f"/home/user02/TUTMING/ming/adarnn/dataset/data/train/{train_file_list[j]}"
        train_data_file = os.path.join(train_file_path, train_file_list[i])
        feat, label_reg, times = get_data(train_data_file)
        sum_times = 0
        if i == 0:
            label_list = label_reg.numpy()
            feat_list = feat.numpy()
            sum_times = times
        else:
            label_list = np.hstack((label_list, label_reg.numpy()))
            feat_list = np.hstack((feat_list, feat.numpy()))
            sum_times += times
    split_time_list = new_TDC(number_domain, feat_list, label_list, sum_times, dis_type=dis_type)
    for j in range(len(split_time_list)):
        time_temp = split_time_list[j]
        train_loader = get_weather_data(train_data_file, start_time=time_temp[0], end_time=time_temp[1],
                                        batch_size=batch_size, )
        train_list.append(train_loader)

    return train_list





