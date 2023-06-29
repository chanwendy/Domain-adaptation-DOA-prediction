import math
import time
import ipdb
import torch
import numpy as np
from pandas import read_csv
import os

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.utils.data as dat
import matplotlib.pyplot as plt

from ourmodel import TransferLoss
#A为原始列表，a为左移位数
def list_move_left(A,a):
    for i in range(a):
        A.insert(len(A),A[0])
        A.remove(A[0])
    return A


#A为原始列表，a为右移位数
def list_move_right(A,a):
    for i in range(a):
        A.insert(0,A.pop())
    return A

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files  # 当前路径下所有非目录子文件,列表

def normalizition(x, mu, sigma):
    # mu 均值 sigms 标准差
    x = (x - mu) / sigma
    return x

def dataload(len_sequence, case_nums=1, traindata="train", time_step=10, ):
    """
    :param case_nums:一次性加载的样本数
    :return: 带有bis,生理特征,用药量信息的字典列表,列表长度为 case_nums
    case_id:样本id列表
    case_information:样本的生理信息表
    case_in_information:样本在信息表中的位置，如 case3:->[0], case30:->[13]
    """
    case_information = read_csv('/home/user02/HYK/bis/information.csv')
    if traindata == "train":
        case_id = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/new_train_clean')
    elif traindata == "test":
        case_id = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/new_test_clean')
        # ni
        # case_id = file_name('/HDD_data/HYK/bis/database/ni_dataset/test')
        # case_id = file_name('/home/user02/HYK/bis/train')
    else:
        # case_id = file_name('/home/user02/HYK/bis/train')
        case_id = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/new_valid_clean')

    for i in range(len(case_id)):
        case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

    case_id = list(map(int, case_id))
    case_id.sort()
    print("file_name:", case_id)
    case_in_information = information_deal(case_id)

    data_seq = [0]*case_nums
    data_label = [0]*case_nums
    for i in tqdm(range(case_nums)):
        if traindata == "train":
            df = read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/new_train_clean/{case_id[i]}.csv')
        elif traindata == 'test':

            df = read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/new_test_clean/{case_id[i]}.csv')
            # NI
            # df = read_csv(f'/HDD_data/HYK/bis/database/ni_dataset/test/{case_id[i]}.csv')

            # df = read_csv(f'/home/user02/HYK/bis/train/{case_id[i]}.csv')
        else:
            df = read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/new_valid_clean/{case_id[i]}.csv')
            # df = read_csv(f'/home/user02/HYK/bis/train/{case_id[i]}.csv')

        x_len = int(len(df.BIS) / time_step)
        # body信息读取
        age = case_information.age[case_in_information[i]]
        if case_information.sex[case_in_information[i]] == 'F':
            sex = 0
        else:
            sex = 1
        height = case_information.height[case_in_information[i]]
        weight = case_information.weight[case_in_information[i]]
        body = torch.tensor([age, sex, height, weight]).float().repeat(x_len, 1)


        # 清除异常值
        modify_RFTN = df.RFTN20_VOL.values
        modify_PPF = df.PPF20_VOL.values
        diff_RFTN = np.diff(modify_RFTN)
        diff_PPF = np.diff(modify_PPF)
        for j in range(len(diff_RFTN)):
            if diff_RFTN[j] < 0:
                temp = (modify_RFTN[j] + modify_RFTN[j+2]) / 2
                df.loc[j + 1, "RFTN20_VOL"] = temp
            if diff_PPF[j] < 0:
                temp = (modify_PPF[j] + modify_PPF[j+2]) / 2
                df.loc[j + 1, "PPF20_VOL"] = temp

        # 为0时刻补上-1800s的零数据
        PPF = list(np.zeros(1200))
        PPF.extend(df.PPF20_VOL.values)
        RFTN = list(np.zeros(1200))
        RFTN.extend(df.RFTN20_VOL.values)

        # 特征制作
        X1 = torch.zeros((x_len, len_sequence))
        X2 = torch.zeros((x_len, len_sequence))

        for x in range(1200, len(PPF)-10, time_step):
            # 从补完数据1800s（实际0s）时刻开始取数据段
            PPF_10s, RFTN_10s = [], []
            for k in range(len_sequence - 1, -1, -1):
                # 第k个10s片段, 共180个
                PPF_10s.append((PPF[x-k*10]-PPF[x-(k+1)*10])*10)
                RFTN_10s.append((RFTN[x-k*10]-RFTN[x-(k+1)*10])*10)
            X1[int((x-1200)/time_step)] = torch.tensor(PPF_10s)
            X2[int((x-1200)/time_step)] = torch.tensor(RFTN_10s)

        X1 = torch.cat((X1, body), dim=1)
        X2 = torch.cat((X2, body), dim=1)
        seq = torch.zeros((x_len, len_sequence + 4, 2)).float()
        seq[:, :, 0] = X1
        seq[:, :, 1] = X2
        # mean = torch.mean(seq, dim=1).reshape((seq.shape[0], 1, seq.shape[2])).repeat(1, 184, 1)
        # std = torch.std(seq, dim=1).reshape((seq.shape[0], 1, seq.shape[2])).repeat(1, 184, 1) + 1e-3
        # seq = normalizition(x=seq, mu=mean, sigma=std)
        data_seq[i] = seq.float()
        label = np.zeros(x_len)

        for j in range(0, x_len, 1):
            label[int(j)] = df.BIS.values[j*time_step]

        data_label[i] = torch.tensor(label).float()

    print("data load finish!", 'case_nums = ', case_nums)
    return data_seq, data_label


# def nidataload(len_sequence, case_nums=1, traindata="train", time_step=10, ):
#     """
#     :param case_nums:一次性加载的样本数
#     :return: 带有bis,生理特征,用药量信息的字典列表,列表长度为 case_nums
#     case_id:样本id列表
#     case_information:样本的生理信息表
#     case_in_information:样本在信息表中的位置，如 case3:->[0], case30:->[13]
#     """
#     case_information = read_csv('/HDD_data/HYK/bis/database/ni_dataset/info.csv')
#     if traindata == "train":
#         case_id = file_name('/HDD_data/MING/adarnn/new_test')
#     elif traindata == "test":
#         # case_id = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/new_test_clean')
#         # ni
#         case_id = file_name('/HDD_data/MING/adarnn/new_test')
#         # case_id = file_name('/home/user02/HYK/bis/train')
#     else:
#         # case_id = file_name('/home/user02/HYK/bis/train')
#         case_id = file_name('/HDD_data/MING/adarnn/new_test')
#     case_id.sort()
#     data_seq = [0]*case_nums
#     data_label = [0]*case_nums
#     for i in tqdm(range(case_nums)):
#         if traindata == "train":
#             df = read_csv(f'/HDD_data/MING/adarnn/new_test/{case_id[i]}')
#         elif traindata == 'test':
#             # df = read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/new_test_clean/{case_id[i]}.csv')
#             # NI
#             df = read_csv(f'/HDD_data/MING/adarnn/new_test/{case_id[i]}')
#             # df = read_csv(f'/home/user02/HYK/bis/train/{case_id[i]}.csv')
#         else:
#             df = read_csv(f'/HDD_data/MING/adarnn/new_test/{case_id[i]}')
#             # df = read_csv(f'/home/user02/HYK/bis/train/{case_id[i]}.csv')
#
#         x_len = int(len(df.BIS) / time_step)
#         # body信息读取
#         age = case_information.age[i]
#         sex = case_information.sex[i]
#         height = case_information.height[i]
#         weight = case_information.weight[i]
#         body = torch.tensor([age, sex, height, weight]).float().repeat(x_len, 1)
#
#
#         # # 清除异常值
#         # modify_RFTN = df.RFTN20_VOL.values
#         # modify_PPF = df.PPF20_VOL.values
#         # diff_RFTN = np.diff(modify_RFTN)
#         # diff_PPF = np.diff(modify_PPF)
#         # for j in range(len(diff_RFTN)):
#         #     if diff_RFTN[j] < 0:
#         #         temp = (modify_RFTN[j] + modify_RFTN[j+2]) / 2
#         #         df.loc[j + 1, "RFTN20_VOL"] = temp
#         #     if diff_PPF[j] < 0:
#         #         temp = (modify_PPF[j] + modify_PPF[j+2]) / 2
#         #         df.loc[j + 1, "PPF20_VOL"] = temp
#
#         # 为0时刻补上-1800s的零数据
#         PPF = list(np.zeros(1200))
#         PPF.extend(df.PPF20_VOL.values)
#         RFTN = list(np.zeros(1200))
#         RFTN.extend(df.RFTN20_VOL.values)
#
#         # 特征制作
#         X1 = torch.zeros((x_len, len_sequence))
#         X2 = torch.zeros((x_len, len_sequence))
#
#         for x in range(1200, len(PPF)-10, time_step):
#             # 从补完数据1800s（实际0s）时刻开始取数据段
#             PPF_10s, RFTN_10s = [], []
#             for k in range(len_sequence - 1, -1, -1):
#                 # 第k个10s片段, 共180个
#                 PPF_10s.append((PPF[x-k*10]-PPF[x-(k+1)*10])*10)
#                 RFTN_10s.append((RFTN[x-k*10]-RFTN[x-(k+1)*10])*10)
#             X1[int((x-1200)/time_step)] = torch.tensor(PPF_10s)
#             X2[int((x-1200)/time_step)] = torch.tensor(RFTN_10s)
#
#         X1 = torch.cat((X1, body), dim=1)
#         X2 = torch.cat((X2, body), dim=1)
#         seq = torch.zeros((x_len, len_sequence + 4, 2)).float()
#         seq[:, :, 0] = X1
#         seq[:, :, 1] = X2
#         # mean = torch.mean(seq, dim=1).reshape((seq.shape[0], 1, seq.shape[2])).repeat(1, 184, 1)
#         # std = torch.std(seq, dim=1).reshape((seq.shape[0], 1, seq.shape[2])).repeat(1, 184, 1) + 1e-3
#         # seq = normalizition(x=seq, mu=mean, sigma=std)
#         data_seq[i] = seq.float()
#         label = np.zeros(x_len)
#
#         for j in range(0, x_len, 1):
#             label[int(j)] = df.BIS.values[j*time_step]
#
#         data_label[i] = torch.tensor(label).float()
#
#     print("data load finish!", 'case_nums = ', case_nums)
#     return data_seq, data_label
def nidataload(len_sequence, case_nums=1, traindata="train", time_step=10, ):
    """
    :param case_nums:一次性加载的样本数
    :return: 带有bis,生理特征,用药量信息的字典列表,列表长度为 case_nums
    case_id:样本id列表
    case_information:样本的生理信息表
    case_in_information:样本在信息表中的位置，如 case3:->[0], case30:->[13]
    """
    case_information = read_csv('/HDD_data/HYK/bis/database/ni_dataset/info.csv')
    if traindata == "train":
        # case_id = file_name('/HDD_data/MING/adarnn/new_train_nosmooth')
        # case_id = file_name('/HDD_data/HYK/bis/database/ni_dataset/train')
        case_id = file_name('/HDD_data/MING/adarnn/nidata/train')
    elif traindata == "test":
        # case_id = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/new_test_clean')
        # ni
        # case_id = file_name('/HDD_data/MING/adarnn/new_test')
        # case_id = file_name('/HDD_data/HYK/bis/database/ni_dataset/test')
        # case_id = file_name('/HDD_data/MING/adarnn/nidata/test')
        case_id = file_name('/HDD_data/MING/adarnn/baseline_model/data/proposed_ni')
        # case_id = file_name('/home/user02/HYK/bis/train')
    else:
        # case_id = file_name('/home/user02/HYK/bis/train')
        # case_id = file_name('/HDD_data/MING/adarnn/new_test')
        # case_id = file_name('/HDD_data/HYK/bis/database/ni_dataset/valid')
        # case_id = file_name('/HDD_data/MING/adarnn/nidata/valid')
        case_id = file_name('/HDD_data/MING/adarnn/baseline_model/data/proposed_ni')
    case_id.sort()
    data_seq = [0]*case_nums
    data_label = [0]*case_nums
    for i in tqdm(range(case_nums)):
        if traindata == "train":
            # df = read_csv(f'/HDD_data/MING/adarnn/new_train_nosmooth/{case_id[i]}')
            # df = read_csv(f'/HDD_data/HYK/bis/database/ni_dataset/train/{case_id[i]}')
            df = read_csv(f'/HDD_data/MING/adarnn/nidata/train/{case_id[i]}')
        elif traindata == 'test':
            # df = read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/new_test_clean/{case_id[i]}.csv')
            # NI
            # df = read_csv(f'/HDD_data/MING/adarnn/new_test/{case_id[i]}')
            # df = read_csv(f'/HDD_data/HYK/bis/database/ni_dataset/test/{case_id[i]}')
            # df = read_csv(f'/HDD_data/MING/adarnn/nidata/test/{case_id[i]}')
            df = read_csv(f'/HDD_data/MING/adarnn/baseline_model/data/proposed_ni/{case_id[i]}')
            # df = read_csv(f'/home/user02/HYK/bis/train/{case_id[i]}.csv')
        else:
            # df = read_csv(f'/HDD_data/MING/adarnn/new_test/{case_id[i]}')
            # df = read_csv(f'/HDD_data/HYK/bis/database/ni_dataset/valid/{case_id[i]}')
            df = read_csv(f'/HDD_data/MING/adarnn/baseline_model/data/proposed_ni/{case_id[i]}')
            # df = read_csv(f'/HDD_data/MING/adarnn/nidata/valid/{case_id[i]}')
            # df = read_csv(f'/home/user02/HYK/bis/train/{case_id[i]}.csv')

        x_len = int(len(df.BIS) / time_step)
        # body信息读取
        age = case_information.age[i]
        sex = case_information.sex[i]
        height = case_information.height[i]
        weight = case_information.weight[i]
        body = torch.tensor([age, sex, height, weight]).float().repeat(x_len, 1)

        alltime = len_sequence * 10
        PPF = list(np.zeros(alltime))
        PPF.extend(df.PPF20_VOL.values)
        RFTN = list(np.zeros(alltime))
        RFTN.extend(df.RFTN20_VOL.values)
        # RFTN_CP = list(np.zeros(alltime))
        # RFTN_CP.extend(df.RFTN20_CP.values)
        bis = df.loc[0, "BIS"]
        BIS = []
        for z in range(alltime):
            BIS.append(bis)
        BIS.extend(df.BIS.values)
        BIS = list_move_right(BIS, 1)
        BIS[0] = bis

        # 特征制作
        X1 = torch.zeros((x_len, len_sequence))
        X2 = torch.zeros((x_len, len_sequence))
        X3 = torch.zeros((x_len, len_sequence))
        X4 = torch.zeros((x_len, len_sequence))

        for x in range(alltime, len(PPF) - 10, time_step):
            # 从补完数据1800s（实际0s）时刻开始取数据段
            PPF_10s, RFTN_10s, BIS_10s = [], [], []
            RFTNCP_10s = []
            for k in range(len_sequence - 1, -1, -1):
                # 第k个10s片段, 共180个
                # 1800 - 1790  1180  010  020          1800 - 1200  1190 000  010
                # 1810 - 1190  1180 620    630          1810  - 1200 1190 610 620
                PPF_10s.append((PPF[x - k * 10] - PPF[x - (k + 1) * 10]) * 10)
                RFTN_10s.append((RFTN[x - k * 10] - RFTN[x - (k + 1) * 10]) * 10)
                BIS_10s.append((BIS[x - (k + 1) * 10]))

                # RFTNCP_10s.append((RFTN_CP[x-k*10] - RFTN_CP[x-(k+1)*10]) * 10)

            X1[int((x - alltime) / time_step)] = torch.tensor(PPF_10s)
            X2[int((x - alltime) / time_step)] = torch.tensor(RFTN_10s)
            X3[int((x - alltime) / time_step)] = torch.tensor(BIS_10s)
            # X3[int((x-alltime)/time_step)] = torch.tensor(RFTNCP_10s)
            # X4[int((x-alltime)/time_step)] = torch.tensor(BIS_10s)

        # for z in range()
        X1 = torch.cat((X1, body), dim=1)
        X2 = torch.cat((X2, body), dim=1)
        X3 = torch.cat((X3, body), dim=1)
        # X4 = torch.cat((X4, body), dim=1)
        seq = torch.zeros((x_len, len_sequence + 4, 3)).float()
        # seq = torch.zeros((x_len, len_sequence + 4, 4)).float()
        seq[:, :, 0] = X1
        seq[:, :, 1] = X2
        seq[:, :, 2] = X3
        # seq[:, :, 3] = X4
        # ipdb.set_trace()

        data_seq[i] = seq.float()
        label = np.zeros(x_len)
        # ?
        for j in range(0, x_len, 1):
            label[int(j)] = df.BIS.values[j * time_step]

        data_label[i] = torch.tensor(label).float()

    print("data load finish!", 'case_nums = ', case_nums)
    return data_seq, data_label

def information_deal(people_list):
    """
    :param people_list: 样本的id列表，如[3, 30, 67 ...]
    :return: 样本在information表中的位置
    """
    case_information = list(read_csv('/home/user02/HYK/bis/information.csv').caseid)
    case_location = list(np.zeros(len(people_list)))
    for i in range(len(people_list)):
        case_location[i] = case_information.index(people_list[i])
    print(case_location, people_list)
    print(case_location[48], people_list[48])
    return case_location  # clear3，30，36......等csv信息在information文件中的位置


def train_data_loader(lensequence, batch=1, batch_size=1):
    train_seq, train_label = dataload(lensequence, case_nums=batch, traindata='train')
    A = train_seq[0]
    B = train_label[0]
    for i in range(1, batch):
        A = torch.cat((A, train_seq[i]), 0)
        B = torch.cat((B, train_label[i]), 0)
    train_data = dat.TensorDataset(A, B)
    train_loader = dat.DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  num_workers=8,
                                  pin_memory=True)
    return train_loader



def test_data_loader(valid, lensequence, batch=1, batch_size=1, test_picture=0, timestep=1):
    # ipdb.set_trace()
    test_seq, test_label = dataload(lensequence, case_nums=batch, traindata=valid, time_step=timestep)

    test_data = list(np.zeros(batch))
    test_loader = list(np.zeros(batch))
    for i in range(batch):
        test_data[i] = dat.TensorDataset(test_seq[i], test_label[i])
        test_loader[i] = dat.DataLoader(dataset=test_data[i],
                                        batch_size=batch_size,
                                        drop_last=True,
                                        pin_memory=True,
                                        num_workers=8)
    return test_loader, test_label

def testni_data_loader(valid, lensequence, batch=1, batch_size=1, test_picture=0, timestep=1):
    # ipdb.set_trace()
    test_seq, test_label = nidataload(lensequence, case_nums=batch, traindata=valid, time_step=timestep)

    test_data = list(np.zeros(batch))
    test_loader = list(np.zeros(batch))
    for i in range(batch):
        test_data[i] = dat.TensorDataset(test_seq[i], test_label[i])
        test_loader[i] = dat.DataLoader(dataset=test_data[i],
                                        batch_size=batch_size,
                                        drop_last=True,
                                        pin_memory=True,
                                        num_workers=8)
    return test_loader, test_label

def TDC_nitrain_data_loader(lensequence, batch=1, batch_size=1):
    # train_seq, train_label = dataload(lensequence, case_nums=batch, traindata='train', time_step=1)
    train_seq, train_label = nidataload(lensequence, case_nums=batch, traindata='train', time_step=10)

    A = train_seq[0]
    B = train_label[0]
    sum_times = len(train_seq[0])
    train_list = []
    for i in range(1, batch):
        A = torch.cat((A, train_seq[i]), 0)
        B = torch.cat((B, train_label[i]), 0)
        sum_times += len(train_seq[i])
    # origin
    split_time_list = new_TDC(3, A, B, sum_times, dis_type="coral")
    # modify
    # split_time_list = new_TDC(5, A, B, sum_times, dis_type="cosine")
    for j in range(len(split_time_list)):
        time_temp = split_time_list[j]
        train_loader = new_get_weather_data(A, B, sum_times, start_time=time_temp[0],
                                            end_time=time_temp[1],
                                            batch_size=batch_size, )
        train_list.append(train_loader)

    return train_list
    # train_data = dat.TensorDataset(A, B)
    # train_loader = dat.DataLoader(dataset=train_data,
    #                               batch_size=batch_size,
    #                               drop_last=True,
    #                               num_workers=8,
    #                               pin_memory=True)
    # return train_loader

class data_loader(Dataset):
    def __init__(self, df_feature, df_label_reg, t=None):
        # print(df_feature.shape)
        # print(df_label_reg.shape)
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

def new_create_dataset(feat, label_reg, times, start_date, end_date, mean=None, std=None):
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

def new_get_weather_data(feat, label, times, start_time, end_time, batch_size, shuffle=True, mean=None, std=None):
    dataset = new_create_dataset(feat, label, times, start_time, end_time, mean=mean, std=std)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader


def TDC_train_data_loader(lensequence, batch=1, batch_size=1):
    train_seq, train_label = dataload(lensequence, case_nums=batch, traindata='train')
    A = train_seq[0]
    B = train_label[0]
    sum_times = len(train_seq[0])
    train_list = []
    for i in range(1, batch):
        A = torch.cat((A, train_seq[i]), 0)
        B = torch.cat((B, train_label[i]), 0)
        sum_times += len(train_seq[i])
    split_time_list = new_TDC(5, A, B, sum_times, dis_type="coral")
    for j in range(len(split_time_list)):
        time_temp = split_time_list[j]
        train_loader = new_get_weather_data(A, B, sum_times, start_time=time_temp[0],
                                            end_time=time_temp[1],
                                            batch_size=batch_size, )
        train_list.append(train_loader)

    return train_list
    # train_data = dat.TensorDataset(A, B)
    # train_loader = dat.DataLoader(dataset=train_data,
    #                               batch_size=batch_size,
    #                               drop_last=True,
    #                               num_workers=8,
    #                               pin_memory=True)
    # return train_loader



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



def time_devide(case_nums=1, traindata="test"):
    """
    :param traindata: 测试集或验证集
    :param case_nums:加载的样本数
    :return: istart:开始注射时间 istop: 停止注射时间
    """
    case_id = file_name(f'/home/user02/TUTMING/ming/adarnn/dataset/data/new_{traindata}_clean')

    for i in range(len(case_id)):
        case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

    case_id = list(map(int, case_id))
    case_id.sort()
    print("file_name:", case_id)
    infusion_start, infusion_stop = [0] * case_nums, [0] * case_nums
    # ipdb.set_trace()
    for i in tqdm(range(case_nums)):
        df = read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/new_{traindata}_clean/{case_id[i]}.csv')

        x_len = int(len(df.BIS))
        ppf = df.PPF20_VOL.values
        start_flag = True
        stop_flag = True
        for j in range(x_len):
            if ppf[j] > 0 and start_flag:
                infusion_start[i] = j
                start_flag = False
            if ppf[-j-1] != ppf[-j-2] and stop_flag:
                infusion_stop[i] = x_len-j+1
                stop_flag = False
            if not start_flag and not stop_flag:
                break

    print(f"{traindata}data load finish!", 'case_nums = ', case_nums)
    return case_id, infusion_start, infusion_stop


# infusion_start, infusion_stop = time_devide(10)
# print(infusion_start)
# print(infusion_stop)






