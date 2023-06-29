import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        return files  # 当


def casefile_clean():
    file_list = file_name('/home/user02/TUTMING/ming/adarnn/dataset/data/medicine_data')
    file_num = len(file_list)
    print(len(file_list), "files was found")
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
    for i in range(file_num):
        adds(file_list[i], clinical)

    return clinical


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


def adds(file, clinical):
    fileid = int(file.split('.csv')[0])
    people = pd.read_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/medicine_data/{file}')
    people = add_feature(people, fileid, clinical)
    people.to_csv(f'/home/user02/TUTMING/ming/adarnn/dataset/data/clean_data/{file}', encoding='utf-8')



