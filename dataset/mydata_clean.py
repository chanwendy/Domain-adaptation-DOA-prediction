import ipdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import torch
from sklearn.preprocessing import MinMaxScaler
import random
import tqdm
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        return files  # 当前路径下所有非目录子文件,列表


def add_feature(data, fileid, clinical, ):
    print("fileid is {}".format(fileid))
    clinical_data = clinical.loc[clinical.caseid == fileid, "age":"weight"]
    data['age'] = float(clinical_data['age'])
    data['sex'] = float(clinical_data['sex'])
    data['weight'] = float(clinical_data['weight'])
    data['height'] = float(clinical_data['height'])
    # data['age'] = np.array(clinical_data['age'], dtype='float32')[0]
    # data['sex'] = np.array(clinical_data['sex'], dtype='float32')[0]
    # data['weight'] = np.array(clinical_data['weight'], dtype='float32')[0]
    # data['height'] = np.array(clinical_data['height'], dtype='float32')[0]
    return data


def data_clean(file, file_nums, train, clinical):
    #  读取数据
    fileid = int(file.split('.csv')[0])
    people = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/medicine_data/{file}')
    # people = people.rename(columns=lambda x: x.replace("Solar8000/", "").replace("Orchestra/", "").replace("BIS/", ""))
    people = people.rename(columns=lambda x: x.replace("Orchestra/", "").replace("BIS/", ""))

    # 丢掉只有前半场手术数据的样本
    if people.BIS[len(people) - 1] <= 60:
        pass

    #  删除错误的心率数据
    # people.HR = people.HR.interpolate(method='linear', limit_direction='forward', axis=0)
    # people = people.dropna(subset=['HR'])
    # people.index = range(0, len(people))
    # people = people.drop(people[people.HR == 0].index, axis=0)
    # people.index = range(0, len(people))

    #  删除错误的BIS数据
    people.BIS = people.BIS.interpolate(method='linear', limit_direction='forward', axis=0)
    people = people.dropna(subset=['BIS'])
    people.index = range(0, len(people))
    people = people.drop(people[people.BIS == 0].index, axis=0)
    people.index = range(0, len(people))

    #  BIS平滑
    if train:
        lowess = sm.nonparametric.lowess
        people.BIS = lowess(people.BIS, people.index, frac=0.03)[:, 1]

    #  错误数据补齐
    # people.PPF20_RATE = people.PPF20_RATE.interpolate(method='linear', limit_direction='forward', axis=0)
    # people.RFTN20_RATE = people.RFTN20_RATE.interpolate(method='linear', limit_direction='forward', axis=0)
    people.RFTN20_VOL = people.RFTN20_VOL.interpolate(method='linear', limit_direction='forward', axis=0)
    people.PPF20_VOL = people.PPF20_VOL.interpolate(method='linear', limit_direction='forward', axis=0)
    people = people.fillna(0)
    for i in range(len(people) - 1):
        if np.abs(people.RFTN20_VOL[i + 1] - people.RFTN20_VOL[i]) >= 10:
            people.RFTN20_VOL[i + 1] = people.RFTN20_VOL[i]
        if np.abs(people.PPF20_VOL[i + 1] - people.PPF20_VOL[i]) >= 10:
            people.PPF20_VOL[i + 1] = people.PPF20_VOL[i]

    #  丢弃前100s内数据缺失超过30s的样本
    for i in range(100):
        if people.time[i + 1] - people.time[i] >= 30:
            pass

    people = add_feature(people, fileid, clinical)
    # print(people.head())

    #  保存数据
    if people.RFTN20_VOL[0] == 0 and people.PPF20_VOL[0] == 0 and people.BIS[0] >= 80:
        if train:
            people.to_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/clean_data/{file}', encoding='utf-8')
        if not train:
            people.to_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/clean_data/{file}', encoding='utf-8')
        print(file, "loading finish")
        return 1
    else:
        return 0

    # print(people)
    # plt.plot(people.BIS)
    # plt.plot(people.HR)
    # plt.plot(people.PPF20_VOL)
    # plt.show()
def get_test_clinical():
    # file_list = file_name('/home/user02/HYK/bis/test')
    file_list = file_name('/home/user02/HYK/bis/new_train')
    file_num = len(file_list)
    # clinical = pd.read_csv('/home/user02/HYK/bis/before_information.csv',
    #                        usecols=['caseid', 'age', 'sex', 'height', 'weight', "bmi"])
    clinical = pd.read_csv('/home/user02/HYK/bis/information.csv',
                           usecols=['caseid', 'age', 'sex', 'height', 'weight', "bmi"])
    for i in range(file_num):
        if clinical['sex'][i] == 'F':
            clinical['sex'][i] = 0
        else:
            clinical['sex'][i] = 1
    data = pd.DataFrame()

    for i in range(len(file_list)):
        fileid = int(file_list[i].split('.csv')[0])
        # ipdb.set_trace()
        clinical_data = clinical.loc[clinical.caseid == fileid, "age":"bmi"]
        if clinical_data['sex'].item() == 'F':
            clinical_data['sex'] = 0
        else:
            clinical_data['sex'] = 1
        data = data.append([{"caseid": fileid,'age': float(clinical_data['age']), "sex":float(clinical_data['sex']), "weight":float(clinical_data['weight']),"height":float(clinical_data['height']), "bmi": float(clinical_data['bmi'])}])
    # data.to_csv('/home/user02/TUTMING/ming/adarnn/dataset/data/clinicial_data/before_bodyinformation.csv', encoding='utf-8')
    data.to_csv('/home/user02/TUTMING/ming/adarnn/dataset/data/clinicial_data/after_bodyinformation.csv',
                encoding='utf-8')



def casefile_clean():
    file_list = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/medicine_data')
    file_num = len(file_list)
    print(len(file_list), "files was found")
    temp_list = np.random.permutation(range(100)).tolist()
    train_size = file_num * 0.6
    valid_size = file_num * 0.2
    test_size = file_num * 0.2
    temp = temp_list

    x = 0  # 加载的第x个case
    y = 0  # 符合要求的case，加载训练集
    train = True
    valid = False
    test = False
    # 添加特征
    clinical = pd.read_csv('/home/user02/TUTMING/ming/adarnn/dataset/data/clinicial_data/clinical.csv',
                           usecols=['caseid', 'age', 'sex', 'height', 'weight'])
    for i in range(file_num):
        if clinical['sex'][i] == 'F':
            clinical['sex'][i] = 0
        else:
            clinical['sex'][i] = 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    clinical[['age', 'height', 'weight']] = scaler.fit_transform(clinical[['age', 'height', 'weight']])
    while y < 100:
        print("loading files {} csv".format(file_list[x]))
        if data_clean(file_list[x], file_num, train, clinical) == 1:
            y += 1
        x += 1

    # y = 0   # 符合要求的case清零，加载测试集
    # # train = False
    # # while y < 100:
    # #     if data_clean(file_list[x], file_num, train) == 1:
    # #         y += 1
    # #     x += 1


# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range


# def informationfile_clean():
#     people = pd.read_csv('/home/user02/HYK/bis/information.csv')
#     people.age = normalization(people.age)
#     people.height = normalization(people.height)
#     people.weight = normalization(people.weight)
#     people.to_csv('/home/user02/HYK/bis/clean_data/information.csv', encoding='utf-8')

def train_valid_test(file_path):
    file_list = file_name(file_path)
    file_num = len(file_list)
    print(len(file_list), "files was found")
    temp_list = np.random.permutation(range(file_num)).tolist()
    train_size = int(file_num * 0.6)
    valid_size = int(file_num * 0.2)
    test_size = int(file_num * 0.2)
    train_file_list = random.sample(file_list, train_size)
    print(len(train_file_list))
    print(train_size)

    for i in range(100 - 1, -1, -1):
        for j in range(train_size - 1, -1, -1):
            if file_list[i] == train_file_list[j]:
                data = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/clean_data/{train_file_list[j]}')
                data.to_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/train/{train_file_list[j]}',
                            encoding='utf-8')
                temp = j
        file_list.remove(train_file_list[temp])
    valid_file_list = random.sample(file_list, valid_size)
    print(len(valid_file_list), valid_size)
    print(len(file_list))
    for i in range(len(file_list) - 1, -1, -1):
        for j in range(valid_size - 1, -1, -1):
            if file_list[i] == valid_file_list[j]:
                data = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/clean_data/{valid_file_list[j]}')
                data.to_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/valid/{valid_file_list[j]}',
                            encoding='utf-8')
                temp = j
        file_list.remove(valid_file_list[temp])
    for i in range(test_size):
        data = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/clean_data/{file_list[i]}')
        data.to_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/test/{file_list[i]}', encoding='utf-8')


# casefile_clean()
# informationfile_clean()
# file_path = "/home/user02/TUTMING/ming/adarnn/dataset/data/clean_data/"
# train_valid_test(file_path)
# get_test_clinical()
def plotBIS():
    file_list = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/new_test')
    file_num = len(file_list)
    for i in range(file_num):
        plt.figure(i)
        people = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/new_test/{file_list[i]}', usecols=["BIS"])

        people = np.array(people).squeeze()
        plt.plot(np.arange(len(people)), people)
        plt.savefig("/home/user02/TUTMING/ming/adarnn/dataset/data/bisfigure/{}.jpg".format(file_list[i]))

# plotBIS()

def plotmdeicine():
    test_file_list = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/new_test_clean')
    train_file_list = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/new_train_clean')
    valid_file_list = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/new_valid_clean')

    file_num = len(test_file_list)
    for i in range(file_num):
        plt.figure(i)
        test_people = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/new_test_clean/{test_file_list[i]}', usecols=["PPF20_VOL", "RFTN20_VOL"])
        valid_people = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/valid_new/{valid_file_list[i]}', usecols=["PPF20_VOL", "RFTN20_VOL"])
        test_people = np.array(test_people)
        plt.plot(np.arange(len(test_people)), test_people[:, 0])
        plt.plot(np.arange(len(test_people)), test_people[:, 1])
        plt.savefig("/home/user02/TUTMING/ming/adarnn/dataset/data/test_medicine/{}.jpg".format(test_file_list[i]))
        plt.close()
        plt.figure(2 * i + 1)
        valid_people = np.array(valid_people)
        plt.plot(np.arange(len(valid_people)), valid_people[:, 0])
        plt.plot(np.arange(len(valid_people)), valid_people[:, 1])
        plt.savefig("/home/user02/TUTMING/ming/adarnn/dataset/data/valid_medicine/{}.jpg".format(valid_file_list[i]))
        plt.close()

    file_num = len(train_file_list)
    for i in range(file_num):
        plt.figure(i)
        train_people = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/new_train_clean/{train_file_list[i]}', usecols=["PPF20_VOL", "RFTN20_VOL"])
        train_people = np.array(train_people)
        plt.plot(np.arange(len(train_people)), train_people[:, 0])
        plt.plot(np.arange(len(train_people)), train_people[:, 1])
        plt.savefig("/home/user02/TUTMING/ming/adarnn/dataset/data/train_medicine/{}.jpg".format(train_file_list[i]))
        plt.close()



def data_resave(case_nums=1, traindata="train", time_step=10):
    # case_id = file_name(f'/home/user02/HYK/bis/database/new_{traindata}_clean')
    #
    # for i in range(len(case_id)):
    #     case_id[i] = case_id[i].split('.')[0]  # 字符串转数字
    #
    # case_id = list(map(int, case_id))
    # case_id.sort()
    # print("file_name:", case_id)
    case_nums = 1
    case_id = [3]
    data_seq = [0]*case_nums
    data_label = [0]*case_nums
    for i in range(case_nums):
        df = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/new_{traindata}_clean/{case_id[i]}.csv')
        # ipdb.set_trace()
        x_len = int(len(df.BIS) / time_step)
        # 为0时刻补上-1800s的零数据
        PPF = list(np.zeros(1800))
        PPF.extend(df.PPF20_VOL.values)
        RFTN = list(np.zeros(1800))
        RFTN.extend(df.RFTN20_VOL.values)

        # 特征制作
        X1 = torch.zeros((x_len, 180))
        X2 = torch.zeros((x_len, 180))

        for x in range(1800, len(PPF)-10, time_step):
            # 从补完数据1800s（实际0s）时刻开始取数据段
            PPF_10s, RFTN_10s = [], []
            for k in range(179, -1, -1):
                # 第k个10s片段, 共180个
                PPF_10s.append((PPF[x-k*10]-PPF[x-(k+1)*10])*10)
                RFTN_10s.append((RFTN[x-k*10]-RFTN[x-(k+1)*10])*10)
            X1[int((x-1800)/time_step)] = torch.tensor(PPF_10s)
            X2[int((x-1800)/time_step)] = torch.tensor(RFTN_10s)

        pre_bis = {}
        for j in range(len(X1)):
            pre_bis[f"t{j}"] = X1[j, :]
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pre_bis.items()]))

        df.to_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/{i}.csv')

    print(f"{traindata}data resave finish!", 'case_nums = ', case_nums)
    return
# plotmdeicine()
# data_resave()




