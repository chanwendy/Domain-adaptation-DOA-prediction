import ipdb
import torch
import numpy as np
from pandas import read_csv
import os
from tqdm import tqdm
import torch.utils.data as dat


class Dataloader:
    def __init__(self, database_wdir, nums, time_step, tw):
        self.database_wdir = database_wdir
        self.nums = nums
        self.time_step = time_step
        self.tw = tw

    def dataload(self, case_nums=1, traindata="train"):
        """
        :param case_nums:一次性加载的样本数
        :return: 带有bis,生理特征,用药量信息的字典列表,列表长度为 case_nums
        case_id:样本id列表
        case_information:样本的生理信息表
        case_in_information:样本在信息表中的位置，如 case3:->[0], case30:->[13]
        x1:ppf_vol
        x2:rftn_vol
        x3:pkpd_bis
        x4:real_bis
        x5-x8:body information(age, sex, height, weight)
        """
        case_information = read_csv(f'/HDD_data/HYK/bis/database/new_{traindata}_clean.csv')
        case_id = self.file_name(data=traindata)
        # ipdb.set_trace()
        for i in range(len(case_id)):
            case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

        case_id = list(map(int, case_id))
        case_id.sort()
        print("file_name:", case_id)
        case_in_information = self.information_deal(case_id, traindata)

        data_seq = [0] * case_nums
        data_label = [0] * case_nums

        for i in tqdm(range(case_nums)):
            df = read_csv(f'{self.database_wdir}/{traindata}/{case_id[i]}.csv')
            x_len = int(len(df.BIS) / self.time_step)
            # body信息读取
            age = case_information.age[case_in_information[i]]
            sex = case_information.sex[case_in_information[i]]
            height = case_information.height[case_in_information[i]]
            weight = case_information.weight[case_in_information[i]]
            body = torch.tensor([age, sex, height, weight]).float().reshape(1, 1, 4).repeat(x_len, self.tw, 1)

            # 清除异常值
            modify_RFTN = df.RFTN20_VOL.values
            modify_PPF = df.PPF20_VOL.values
            diff_RFTN = np.diff(modify_RFTN)
            diff_PPF = np.diff(modify_PPF)
            for j in range(len(diff_RFTN)):
                if diff_RFTN[j] < 0:
                    temp = (modify_RFTN[j] + modify_RFTN[j + 2]) / 2
                    df.loc[j + 1, "RFTN20_VOL"] = temp
                if diff_PPF[j] < 0:
                    temp = (modify_PPF[j] + modify_PPF[j + 2]) / 2
                    df.loc[j + 1, "PPF20_VOL"] = temp

            # 为0时刻补上-1800s的零数据
            PPF = list(np.zeros(self.tw * 10))
            PPF.extend(df.PPF20_VOL.values)
            RFTN = list(np.zeros(self.tw * 10))
            RFTN.extend(df.RFTN20_VOL.values)

            ppf_ce = df.PPF_CE.values
            rftn_ce = df.RFTN20_CE.values
            pkpd_bis = self.pkpd(ppf_ce, rftn_ce)
            PKPD_bis = list(np.ones(self.tw * 10)*98)
            PKPD_bis.extend(pkpd_bis)

            # 特征制作
            X1 = torch.zeros((x_len, self.tw))
            X2 = torch.zeros((x_len, self.tw))
            X3 = torch.zeros((x_len, self.tw))
            X4 = torch.zeros((x_len, self.tw))
            # X5 = torch.zeros((x_len, self.tw))
            for x in range(self.tw*10, len(PPF) - self.time_step, self.time_step):
                # 从补完数据1800s（实际0s）时刻开始取数据段
                PPF_10s, RFTN_10s, BIS_10s = [], [], []
                for k in range(self.tw-1, -1, -1):
                    # 第k个10s片段, 共180个
                    PPF_10s.append((PPF[x - k * 10] - PPF[x - (k + 1) * 10]) * 0.1)
                    RFTN_10s.append((RFTN[x - k * 10] - RFTN[x - (k + 1) * 10]) * 0.1)
                    BIS_10s.append((PKPD_bis[x - k * 10]))

                X1[int((x - self.tw * 10) / self.time_step)] = torch.tensor(PPF_10s)
                X2[int((x - self.tw * 10) / self.time_step)] = torch.tensor(RFTN_10s)
                X3[int((x - self.tw * 10) / self.time_step)] = torch.tensor(BIS_10s)

            bis = torch.tensor(df.BIS.values)


            for k in range(x_len):
                if k * self.time_step < self.tw:
                    X4[k, :] = torch.cat((torch.ones(self.tw - k * self.time_step) * 98, bis[:k * self.time_step]), dim=0)
                    # X3[k, :] = torch.cat((torch.zeros(self.tw - k * self.time_step), pkpd_bis[:k * self.time_step]), dim=0)
                    # X5[k, :] = torch.cat((torch.zeros(180 - k * self.time_step), rftn_ce[:k * self.time_step]), dim=0)

                else:
                    X4[k, :] = bis[k * self.time_step - self.tw:k * self.time_step]
                    # X3[k, :] = pkpd_bis[k * self.time_step - self.tw:k * self.time_step]
                    # X5[k, :] = rftn_ce[k * self.time_step - 180:k * self.time_step]

            seq = torch.zeros((x_len, self.tw, 4)).float()
            seq[:, :, 0] = X1  # ppf vol
            seq[:, :, 1] = X2  # rftn vol
            seq[:, :, 2] = X3  # pk-pd bis
            seq[:, :, 3] = X4  # bis history
            # seq[:, :, 3] = X5  # rftn ce

            # 归一化
            mean = torch.mean(seq, dim=1).reshape((seq.shape[0], 1, seq.shape[2])).repeat(1, self.tw, 1)
            std = torch.std(seq, dim=1).reshape((seq.shape[0], 1, seq.shape[2])).repeat(1, self.tw, 1) + 1e-3
            seq = self.normalizition(x=seq, mu=mean, sigma=std)

            out = torch.cat((seq, body), dim=2)
            # out = torch.cat((out, seq[:, :, 2].reshape(seq.shape[0], 180, 1)), dim=2)

            data_seq[i] = out.float()
            label = np.zeros(x_len)
            for j in range(0, x_len, 1):
                label[int(j)] = df.BIS.values[j * self.time_step]

            data_label[i] = torch.tensor(label).float()

        print(f"{traindata}data load finish!", 'case_nums = ', case_nums)
        return data_seq, data_label

    def train_data_loader(self, batch=1, batch_size=1, data="train", shuffle=True):
        train_seq, train_label = self.dataload(case_nums=batch, traindata=data)
        A = train_seq[0]
        B = train_label[0]
        for i in range(1, batch):
            A = torch.cat((A, train_seq[i]), 0)
            B = torch.cat((B, train_label[i]), 0)
        train_data = dat.TensorDataset(A, B)
        train_loader = dat.DataLoader(dataset=train_data,
                                      batch_size=batch_size,
                                      drop_last=True,
                                      num_workers=4,
                                      pin_memory=True,
                                      shuffle=shuffle)
        return train_loader

    def test_data_loader(self, batch=1, batch_size=1, data="test"):
        test_seq, test_label = self.dataload(case_nums=batch, traindata=data)
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

    def information_deal(self, people_list, data="train"):
        """
        :param people_list: 样本的id列表，如[3, 30, 67 ...]
        :return: 样本在information表中的位置
        """
        case_information = list(read_csv(f'/HDD_data/HYK/bis/database/new_{data}_clean.csv').caseid)
        case_location = list(np.zeros(len(people_list)))
        for i in range(len(people_list)):
            case_location[i] = case_information.index(people_list[i])
        return case_location  # clear3，30，36......等csv信息在information文件中的位置

    def time_devide(self, case_nums=1, traindata="test", dataset="vital"):
        """
        :param traindata: 测试集或验证集
        :param case_nums:加载的样本数
        :return: istart:开始注射时间 istop: 停止注射时间
        """
        case_id = self.file_name(traindata)

        for i in range(len(case_id)):
            case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

        case_id = list(map(int, case_id))
        case_id.sort()
        print("file_name:", case_id)
        infusion_start, infusion_stop = [0] * case_nums, [0] * case_nums
        for i in tqdm(range(case_nums)):
            if dataset == "vital":
                df = read_csv(f'/HDD_data/HYK/bis/database/{traindata}/{case_id[i]}.csv')
            else:
                df = read_csv(f'/data/HYK/DATASET/bis/database/{traindata}/{case_id[i]}.csv')

            x_len = int(len(df.BIS))
            ppf = df.PPF20_VOL.values
            start_flag = True
            stop_flag = True
            for j in range(x_len):
                if ppf[j] > 0 and start_flag:
                    infusion_start[i] = j
                    start_flag = False
                if ppf[-j - 1] != ppf[-j - 2] and stop_flag:
                    infusion_stop[i] = x_len - j + 1
                    stop_flag = False
                if not start_flag and not stop_flag:
                    break

        print(f"{traindata}data load finish!", 'case_nums = ', case_nums)
        return infusion_start, infusion_stop

    def file_name(self, data):
        for root, dirs, files in os.walk(f'{self.database_wdir}/{data}'):
            return files  # 当前路径下所有非目录子文件,列表

    @staticmethod
    def pkpd(Ec1, Ec2):
        ppf_ec50 = 4.47
        rftn_ec50 = 19.3
        gamma = 1.43
        p_gamma = (Ec1/ppf_ec50 + Ec2/rftn_ec50)**gamma
        bis = 98. - 98. * p_gamma / (1 + p_gamma)
        return bis

    @staticmethod
    def normalizition(x, mu, sigma):
        # mu 均值 sigms 标准差
        x = (x - mu) / sigma
        return x


if __name__ == "__main__":
    a = Dataloader(database_wdir="/home/user02/HYK/bis/database/clean", time_step=1, nums=1)
    A = a.dataload(1, "train")



