import math

import ipdb
import torch.nn as nn
import torch
import torch.optim as optim
import os
import argparse
import datetime
import numpy as np
import evaluate
from tqdm import tqdm
from utils import utils
# from base.distill_model import Student_AdaRNN, AdaRNN,Distill
from ourmodel.distill_model import AdaRNN, Student_AdaRNN, Distill
from ourmodel.baseline import LstmModel
import dataset.predict_loader as loader
import pretty_errors
# import dataset.mydata_process as data_process
import matplotlib.pyplot as plt
from dataset import database


def pprint(*text):
    # print with UTC+8 time
    time = '[' + str(datetime.datetime.utcnow() +
                     datetime.timedelta(hours=8))[:19] + '] -'
    print(time, *text, flush=True)
    if args.log_file is None:
        return
    with open(args.log_file, 'a') as f:
        print(time, *text, flush=True, file=f)

def get_teacher_model(name='AdaRNN'):
    n_hiddens = [args.hidden_size for i in range(args.num_layers)]
    return AdaRNN(use_bottleneck=True, bottleneck_width=256, n_input=3, n_hiddens=n_hiddens,
                  n_output=args.class_num, dropout=args.dropout, model_type=name, len_seq=args.len_seq,
                  trans_loss=args.loss_type).cuda()


def get_student_model(name='AdaRNN'):
    n_hiddens = [args.hidden_size for i in range(args.num_layers)]
    return Student_AdaRNN(use_bottleneck=True, bottleneck_width=256, n_input=2, n_hiddens=n_hiddens,
                  n_output=args.class_num, dropout=args.dropout, model_type=name, len_seq=args.len_seq,
                  trans_loss=args.loss_type).cuda()


def get_distill_model(name='AdaRNN'):
    n_hiddens = [args.hidden_size for i in range(args.num_layers)]
    return Distill(use_bottleneck=True, bottleneck_width=256, n_input=args.d_feat, n_hiddens=n_hiddens,
                  n_output=args.class_num, dropout=args.dropout, model_type=name, len_seq=args.len_seq,
                  trans_loss=args.loss_type).cuda()

def get_index(num_domain=2):
    index = []
    for i in range(num_domain):
        for j in range(i + 1, num_domain + 1):
            index.append((i, j))
    return index

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_epoch(model, test_loader, prefix='Test'):
    model.eval()
    total_loss = 0
    total_loss_1 = 0
    total_loss_r = 0
    correct = 0
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    for feature, label_reg in tqdm(test_loader, desc=prefix, total=len(test_loader)):
        feature, label_reg = feature.cuda().float(), label_reg.cuda().float()
        with torch.no_grad():
            pred = model.predict(feature[:, :, :2])
        loss = criterion(pred, label_reg)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, label_reg)
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()
    loss = total_loss / len(test_loader)
    loss_1 = total_loss_1 / len(test_loader)
    loss_r = loss_r / len(test_loader)
    return loss, loss_1, loss_r

def test_epoch_inference(model, test_loader, valid_file_list, test_file_list, prefix='Test', valid_infusion_start=None, valid_infusion_stop=None, test_infusion_start=None, test_infusion_stop=None,baseline_model=None):
    model.eval()
    total_loss = 0
    total_loss_1 = 0
    total_loss_r = 0
    total_MDPE = 0
    total_MDAPE = 0
    correct = 0
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    i = 0
    base_predict_list = None
    for feature, label_reg in tqdm(test_loader, desc=prefix, total=len(test_loader)):
        feature, label_reg = feature.cuda().float(), label_reg.cuda().float()
        with torch.no_grad():
            import time
            start = time.time()
            pred = model.predict(feature[:, :, :2])
            end = time.time()
        loss = criterion(pred, label_reg)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, label_reg)
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()
        PE = ((label_reg - pred) / pred).cpu()
        # PE = np.array(PE)
        MDPE = np.median(PE) * 100
        MDAPE = np.median(np.abs(PE)) * 100
        total_MDPE += MDPE
        total_MDAPE += MDAPE
        if i == 0:
            tlabel_list = label_reg
            label_list = label_reg.cpu().numpy()
            predict_list = pred.cpu().numpy()
            tpredict_list = pred
        else:
            # print("tpredict_list shape {}".format(tpredict_list.shape))
            # print("pred shape {}".format(pred.shape))
            tpredict_list = torch.cat((tpredict_list, pred))
            tlabel_list = torch.cat((tlabel_list, label_reg))
            label_list = np.hstack((label_list, label_reg.cpu().numpy()))
            predict_list = np.hstack((predict_list, pred.cpu().numpy()))
        i = i + 1
        # i = i + 1

    return label_list, predict_list
    # return MSE_list, loss_1, RMSE_list, label_list, predict_list, MDPE_list, MDPAE_list


def inference(model, data_loader, epoch, valid_file_list=None, test_file_list=None, valid_infusion_start=None,
              valid_infusion_stop=None, test_infusion_start=None, test_infusion_stop=None, baseline_model=None):
    label_list, predict_list = test_epoch_inference(
        model, data_loader, valid_file_list, test_file_list, prefix='Inference',
        valid_infusion_start=valid_infusion_start,
        valid_infusion_stop=valid_infusion_stop, test_infusion_start=test_infusion_start,
        test_infusion_stop=test_infusion_stop, baseline_model=baseline_model)
    return label_list, predict_list


# 用模型对训练、验证、和测试集的推理 判断模型在哪个集上效果更好
def inference_all(output_path, model, model_path, loaders, valid_file_list, test_file_list, epoch, valid_infusion_start,
                  valid_infusion_stop, test_infusion_start, test_infusion_stop, baseline_model):
    pprint('inference...')
    loss_list = []
    loss_l1_list = []
    loss_r_list = []
    MDPE_list = []
    MDAPE_list = []
    model.load_state_dict(torch.load(model_path))


    i = 0
    list_name = ['train', 'valid', 'test']
    distill_testout = None
    distilllabel_list = None

    for loader in loaders:
        if i == 0:
             label_list, predict_list = inference(
                model, loader, epoch, baseline_model=baseline_model)
        elif i == 1:
            label_list, predict_list = inference(
                model, loader, epoch, valid_file_list, valid_infusion_start=valid_infusion_start,
                valid_infusion_stop=valid_infusion_stop,
                test_infusion_start=test_infusion_start, test_infusion_stop=test_infusion_stop,
                baseline_model=baseline_model)
        elif i == 2:
            label_list, predict_list = inference(
                model, loader, epoch, test_file_list=test_file_list, valid_infusion_start=valid_infusion_start,
                valid_infusion_stop=valid_infusion_stop,
                test_infusion_start=test_infusion_start, test_infusion_stop=test_infusion_stop,
                baseline_model=baseline_model)
            distill_testout = predict_list
            distilllabel_list = label_list

        i = i + 1
    return distill_testout, distilllabel_list



def transform_type(init_weight):
    weight = torch.ones(args.num_layers, args.len_seq).cuda()
    for i in range(args.num_layers):
        for j in range(args.len_seq):
            weight[i, j] = init_weight[i][j].item()
    return weight


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        return files  # 当前路径下所有非目录子文件,列表

def main_transfer(args):
    print(args)
    output_path = args.outdir + '_' + "1_people" + '_' + str(args.n_epochs) +  args.model_name + 'medicine' + \
                  args.loss_type + '_' + str(args.pre_epoch) + \
                  '_' + str(args.dw) + '_' + str(args.lr)
    save_model_name = args.model_name + '_' + args.loss_type + \
                      '_' + str(args.dw) + '_' + str(args.lr) + '.pkl'
    utils.dir_exist(output_path)
    pprint('create loaders...')
    # best
    train_loader_list = loader.TDC_train_data_loader(args.len_seq, args.people_size, args.batch_size)
    valid_loader_list, valid_label = loader.test_data_loader(valid="valid", lensequence=args.len_seq, batch=args.test_size, batch_size=128, timestep=1)
    test_loader_list, test_label = loader.test_data_loader(valid="test", lensequence=args.len_seq, batch=args.test_size, batch_size=128, timestep=1)
    # ni
    # train_loader_list = loader.TDC_nitrain_data_loader(args.len_seq, 24, args.batch_size)
    # valid_loader_list, valid_label = loader.testni_data_loader(valid="valid", lensequence=args.len_seq, batch=args.test_size, batch_size=128, timestep=1)
    # test_loader_list, test_label = loader.testni_data_loader(valid="test", lensequence=args.len_seq, batch=args.test_size, batch_size=128, timestep=1)
    valid_file_list, valid_infusion_start, valid_infusion_stop = loader.time_devide(args.test_size, "valid")
    test_file_list, test_infusion_start, test_infusion_stop = loader.time_devide(args.test_size, "test")


    args.log_file = os.path.join(output_path, 'run.log')
    pprint('create model...')
    # Teacher_model = get_teacher_model(args.model_name)
    Student_model = get_student_model(args.model_name)

    modelname = "your best model path "
    baseline_model = "baseline best model"

    # 推理
    distill_out = []
    distill_label_list = []
    for i in range(args.test_size):
        loaders = train_loader_list[0], valid_loader_list[i], test_loader_list[i]
        distill_testout, distilllabel_list = inference_all(output_path, Student_model, modelname, loaders, valid_file_list[i], test_file_list[i],
                                                                                    epoch=i,
                                                                                    valid_infusion_start=valid_infusion_start[i],
                                                                                    valid_infusion_stop=valid_infusion_stop[i],
                                                                                    test_infusion_start=test_infusion_start[i],
                                                                                    test_infusion_stop=test_infusion_stop[i],baseline_model=baseline_model)
        distill_out.append(distill_testout)
        distill_label_list.append(distilllabel_list)

    return distill_out, distill_label_list


def get_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='AdaRNN')
    parser.add_argument('--d_feat', type=int, default=3)        # 2

    parser.add_argument('--hidden_size', type=int, default=64)     # 180
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--class_num', type=int, default=1)
    parser.add_argument('--pre_epoch', type=int, default=20)  # 30, 40, 50

    # training
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--early_stop', type=int, default=40)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--people_size', type=int, default=1)
    parser.add_argument('--dw', type=float, default=0.05)  # 1.0, 0.01, 5.0
    parser.add_argument('--loss_type', type=str, default='cosine')
    parser.add_argument('--data_mode', type=str,
                        default='tdc')
    parser.add_argument('--num_domain', type=int, default=2)
    parser.add_argument('--len_seq', type=int, default=120)       # 180
    parser.add_argument('--test_size', type=int, default=76)    # 76
    parser.add_argument('--train_size', type=int, default=30)

    # other
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--train_path', default="your train data path")
    parser.add_argument('--test_path', default="your test data path")
    parser.add_argument('--valid_path', default="your valid data path")
    parser.add_argument('--outdir', default='your output data path')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log_file', type=str, default='run.log')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--len_win', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    # device = torch.device("cuda:0")
    #
    distill_out, distill_label_list = main_transfer(args)
    d_box = database.Dataloader(
        database_wdir="your train data path",
        time_step=1,
        nums=1,
        tw=180
    )
    ist, isp = d_box.time_devide(case_nums=76, traindata="test")
    access = evaluate.Evalulate(distill_out, distill_label_list, ist, isp, case_num=76)
    print("MDPE    MDAPE    RMSE    MAE\r")
    for i in range(4):
        p = access.loss(i)
        print("{:.2f}     {:.2f}    {:.2f}  {:.2f}".format(p[0], p[1], p[2], p[3]))



