import os
import matplotlib.pyplot as plt
import ipdb
import numpy as np
import pandas as pd
import numpy
from tqdm import tqdm

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files  # 当前路径下所有非目录子文件,列表

def copyfiles(traindata="train"):

    if traindata == "train":
        case_id = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/new_train_clean')
        # case_id = file_name('/home/user02/HYK/bis/database/ce_clean/train')
    elif traindata == "test":
        case_id = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/new_test_clean')
        # case_id = file_name('/home/user02/HYK/bis/database/ce_clean/test')
    else:
        case_id = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/new_valid_clean')
        # case_id = file_name('/home/user02/HYK/bis/database/ce_clean/test')

    for i in range(len(case_id)):
        case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

    case_id = list(map(int, case_id))
    case_id.sort()
    print("file_name:", case_id)
    for i in tqdm(range(len(case_id))):
        if traindata == "train":
            data = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_train/{case_id[i]}.csv')
            df = pd.read_csv(f'/HDD_data/HYK/bis/New Folder/{case_id[i]}.csv')
            data_times = data.time.values
            df_times = df.time.values
            length = len(data_times)
            RFTN_temp = np.zeros(length)
            PPF_temp = np.zeros(length)
            PPF_CP_temp = np.zeros(length)
            for j in range(length):
                RFTN_temp[j] = df.loc[df.time == data_times[j], "Orchestra/RFTN20_CE"]
            data["RFTN20_CE"] = RFTN_temp
            for j in range(length):
                PPF_temp[j] = df.loc[df.time == data_times[j], "Orchestra/PPF20_CE"]
            data["PPF_CE"] = PPF_temp
            for j in range(length):
                PPF_CP_temp[j] = df.loc[df.time == data_times[j], "Orchestra/PPF20_CP"]
            data["PPF_CP"] = PPF_CP_temp

            # data.RFTN20_CP = data.RFTN20_CP.interpolate(method='linear', limit_direction='forward', axis=0)
            data.RFTN20_CE = data.RFTN20_CE.interpolate(method='linear', limit_direction='forward', axis=0)
            data.PPF_CE = data.PPF_CE.interpolate(method='linear', limit_direction='forward', axis=0)
            data.PPF_CP = data.PPF_CP.interpolate(method='linear', limit_direction='forward', axis=0)
            temp = np.array(data.RFTN20_CE)
            PPF_temp = np.array(data.PPF_CE)
            PPF_CP_temp = np.array(data.PPF_CP)
            # ipdb.set_trace()
            for k in range(length):
                if np.isnan(temp[k]) == True:
                    temp[k] = temp[k - 1]
            data["RFTN20_CE"] = temp
            for k in range(length):
                if np.isnan(PPF_temp[k]) == True:
                    PPF_temp[k] = PPF_temp[k - 1]
            data["PPF_CE"] = PPF_temp
            for k in range(length):
                if np.isnan(PPF_CP_temp[k]) == True:
                    PPF_CP_temp[k] = PPF_CP_temp[k - 1]
            data["PPF_CP"] = PPF_CP_temp
            if data.PPF_CP.isnull().value_counts().values[0] != length:
                print("nan nums are {}".format(data.PPF_CP.isnull().value_counts().values[1]))
                print("train file name is {}".format(case_id[i]))
            data.to_csv(f"/home/user02/HYK/bis/database/waiting_train/{case_id[i]}.csv")
        elif traindata == 'test':
            data = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_test/{case_id[i]}.csv')
            df = pd.read_csv(f'/HDD_data/HYK/bis/New Folder/{case_id[i]}.csv')
            data_times = data.time.values
            df_times = df.time.values
            length = len(data_times)
            RFTN_temp = np.zeros(length)
            PPF_temp = np.zeros(length)
            PPF_CP_temp = np.zeros(length)
            for j in range(length):
                RFTN_temp[j] = df.loc[df.time == data_times[j], "Orchestra/RFTN20_CE"]
            data["RFTN20_CE"] = RFTN_temp
            for j in range(length):
                PPF_temp[j] = df.loc[df.time == data_times[j], "Orchestra/PPF20_CE"]
            data["PPF_CE"] = PPF_temp
            for j in range(length):
                PPF_CP_temp[j] = df.loc[df.time == data_times[j], "Orchestra/PPF20_CP"]
            data["PPF_CP"] = PPF_CP_temp

            # data.RFTN20_CP = data.RFTN20_CP.interpolate(method='linear', limit_direction='forward', axis=0)
            data.RFTN20_CE = data.RFTN20_CE.interpolate(method='linear', limit_direction='forward', axis=0)
            data.PPF_CE = data.PPF_CE.interpolate(method='linear', limit_direction='forward', axis=0)
            data.PPF_CP = data.PPF_CP.interpolate(method='linear', limit_direction='forward', axis=0)
            temp = np.array(data.RFTN20_CE)
            PPF_temp = np.array(data.PPF_CE)
            PPF_CP_temp = np.array(data.PPF_CP)
            # ipdb.set_trace()
            for k in range(length):
                if np.isnan(temp[k]) == True:
                    temp[k] = temp[k - 1]
            data["RFTN20_CE"] = temp
            for k in range(length):
                if np.isnan(PPF_temp[k]) == True:
                    PPF_temp[k] = PPF_temp[k - 1]
            data["PPF_CE"] = PPF_temp
            for k in range(length):
                if np.isnan(PPF_CP_temp[k]) == True:
                    PPF_CP_temp[k] = PPF_CP_temp[k - 1]
            data["PPF_CP"] = PPF_CP_temp
            if data.PPF_CP.isnull().value_counts().values[0] != length:
                print("nan nums are {}".format(data.PPF_CP.isnull().value_counts().values[1]))
                print("train file name is {}".format(case_id[i]))
            data.to_csv(f"/home/user02/HYK/bis/database/waiting_test/{case_id[i]}.csv")
        else:
            data = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_valid/{case_id[i]}.csv')
            df = pd.read_csv(f'/HDD_data/HYK/bis/New Folder/{case_id[i]}.csv')
            data_times = data.time.values
            df_times = df.time.values
            length = len(data_times)
            RFTN_temp = np.zeros(length)
            PPF_temp = np.zeros(length)
            PPF_CP_temp = np.zeros(length)
            for j in range(length):
                RFTN_temp[j] = df.loc[df.time == data_times[j], "Orchestra/RFTN20_CE"]
            data["RFTN20_CE"] = RFTN_temp
            for j in range(length):
                PPF_temp[j] = df.loc[df.time == data_times[j], "Orchestra/PPF20_CE"]
            data["PPF_CE"] = PPF_temp
            for j in range(length):
                PPF_CP_temp[j] = df.loc[df.time == data_times[j], "Orchestra/PPF20_CP"]
            data["PPF_CP"] = PPF_CP_temp

            # data.RFTN20_CP = data.RFTN20_CP.interpolate(method='linear', limit_direction='forward', axis=0)
            data.RFTN20_CE = data.RFTN20_CE.interpolate(method='linear', limit_direction='forward', axis=0)
            data.PPF_CE = data.PPF_CE.interpolate(method='linear', limit_direction='forward', axis=0)
            data.PPF_CP = data.PPF_CP.interpolate(method='linear', limit_direction='forward', axis=0)
            temp = np.array(data.RFTN20_CE)
            PPF_temp = np.array(data.PPF_CE)
            PPF_CP_temp = np.array(data.PPF_CP)
            # ipdb.set_trace()
            for k in range(length):
                if np.isnan(temp[k]) == True:
                    temp[k] = temp[k - 1]
            data["RFTN20_CE"] = temp
            for k in range(length):
                if np.isnan(PPF_temp[k]) == True:
                    PPF_temp[k] = PPF_temp[k - 1]
            data["PPF_CE"] = PPF_temp
            for k in range(length):
                if np.isnan(PPF_CP_temp[k]) == True:
                    PPF_CP_temp[k] = PPF_CP_temp[k - 1]
            data["PPF_CP"] = PPF_CP_temp
            if data.PPF_CP.isnull().value_counts().values[0] != length:
                print("nan nums are {}".format(data.PPF_CP.isnull().value_counts().values[1]))
                print("train file name is {}".format(case_id[i]))
            data.to_csv(f"/home/user02/HYK/bis/database/waiting_valid/{case_id[i]}.csv")
#
copyfiles("train")
copyfiles("test")
copyfiles("valid")
def plotmdeicine():
    test_file_list = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_test')
    train_file_list = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_train')
    valid_file_list = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_valid')

    file_num = len(test_file_list)
    for i in range(file_num):
        plt.figure(i)
        # test_people = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/new_test_clean/{test_file_list[i]}', usecols=["PPF20_VOL", "RFTN20_VOL"])
        test_people = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_test/{test_file_list[i]}')
        # print("nan nums are {}".format(test_people.isnull().value_counts().values[1]))
        test_people.RFTN20_CP = test_people.RFTN20_CP.interpolate(method='linear', limit_direction='forward', axis=0)
        if test_people.isnull().value_counts().values[0] != len(test_people):
            print("nan nums are {}".format(test_people.isnull().value_counts().values[1]))
            print("test file name is {}".format(test_file_list[i]))
        # valid_people = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/valid_new/{valid_file_list[i]}', usecols=["PPF20_VOL", "RFTN20_VOL"])
        valid_people = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_valid/{valid_file_list[i]}')
        valid_people.RFTN20_CP = valid_people.RFTN20_CP.interpolate(method='linear', limit_direction='forward', axis=0)
        if valid_people.isnull().value_counts().values[0] != len(valid_people):
            print("nan nums are {}".format(valid_people.isnull().value_counts().values[1]))
            print("valid file name is {}".format(valid_file_list[i]))
        test_people1 = np.array(test_people.RFTN20_CP)
        for j in range(len(test_people)):
            if np.isnan(test_people1[j]) == True:
                test_people1[j] = test_people1[j-1]
        # ipdb.set_trace()
        test_people["RFTN20_CP"] = test_people1
        nan = np.isnan(test_people1)
        print(True in nan)
        plt.plot(np.arange(len(test_people)), test_people1)
        # plt.plot(np.arange(len(test_people)), test_people[:, 1])
        test_people.to_csv("/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_test/{}".format(test_file_list[i]))
        plt.savefig("/home/user02/TUTMING/ming/adarnn/dataset/data/test_medicine/{}.jpg".format(test_file_list[i]))
        plt.close()
        plt.figure(2 * i + 1)
        valid_people1 = np.array(valid_people.RFTN20_CP)
        for j in range(len(valid_people1)):
            if np.isnan(valid_people1[j]) == True:
                valid_people1[j] = valid_people1[j-1]
        nan = np.isnan(valid_people1)
        valid_people["RFTN20_CP"] = valid_people1
        print(True in nan)
        plt.plot(np.arange(len(valid_people)), valid_people1)
        # plt.plot(np.arange(len(valid_people)), valid_people[:, 1])
        valid_people.to_csv("/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_valid/{}".format(valid_file_list[i]))
        plt.savefig("/home/user02/TUTMING/ming/adarnn/dataset/data/valid_medicine/{}.jpg".format(valid_file_list[i]))
        plt.close()

    file_num = len(train_file_list)
    for i in range(file_num):
        plt.figure(i)
        train_people = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_train/{train_file_list[i]}')
        train_people.RFTN20_CP = train_people.RFTN20_CP.interpolate(method='linear', limit_direction='forward', axis=0)
        if train_people.isnull().value_counts().values[0] != len(train_people):
            print("nan nums are {}".format(train_people.isnull().value_counts().values[1]))
            print("train file name is {}".format(train_file_list[i]))
        train_people1 = np.array(train_people.RFTN20_CP)
        for j in range(len(train_people1)):
            if np.isnan(train_people1[j]) == True:
                train_people1[j] = train_people1[j-1]
        train_people["RFTN20_CP"] = train_people1
        nan = np.isnan(train_people)
        print(True in nan)
        plt.plot(np.arange(len(train_people)), train_people1)
        # plt.plot(np.arange(len(train_people)), train_people[:, 1])
        train_people.to_csv("/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_test/{}".format(train_file_list[i]))
        plt.savefig("/home/user02/TUTMING/ming/adarnn/dataset/data/train_medicine/{}.jpg".format(train_file_list[i]))
        plt.close()

def checknan(traindata):

    if traindata == "train":
        case_id = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_train')
        # case_id = file_name('/home/user02/HYK/bis/database/ce_clean/train')
    elif traindata == "test":
        case_id = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_test')
        # case_id = file_name('/home/user02/HYK/bis/database/ce_clean/test')
    else:
        case_id = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_valid')
        # case_id = file_name('/home/user02/HYK/bis/database/ce_clean/test')

    for i in range(len(case_id)):
        case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

    case_id = list(map(int, case_id))
    case_id.sort()
    print("file_name:", case_id)
    for i in tqdm(range(len(case_id))):
        if traindata == "train":
            data = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_train/{case_id[i]}.csv')
            RFTN20_CP = data.RFTN20_CP
            if RFTN20_CP.isnull().value_counts().values[0] != len(RFTN20_CP):
                print("nan nums are {}".format(RFTN20_CP.isnull().value_counts().values[1]))
                print("train file name is {}".format(case_id[i]))
        elif traindata == 'test':
            data = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_test/{case_id[i]}.csv')
            RFTN20_CP = data.RFTN20_CP
            if RFTN20_CP.isnull().value_counts().values[0] != len(RFTN20_CP):
                print("nan nums are {}".format(RFTN20_CP.isnull().value_counts().values[1]))
                print("test file name is {}".format(case_id[i]))
        else:
            data = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/waiting_valid/{case_id[i]}.csv')
            RFTN20_CP = data.RFTN20_CP
            if RFTN20_CP.isnull().value_counts().values[0] != len(RFTN20_CP):
                print("nan nums are {}".format(RFTN20_CP.isnull().value_counts().values[1]))
                print("valid file name is {}".format(case_id[i]))
#
#
# checknan("train")
# checknan("test")
# checknan("valid")