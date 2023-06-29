# for one people
import math
import ipdb
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
import datetime
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from ourmodel.loss_transfer import TransferLoss
from utils import utils
from ourmodel.distill_model import AdaRNN, Student_AdaRNN, Distill
from ourmodel.baseline import LstmModel
import dataset.new_dataloader as loader
import dataset.mydata_clean as data_clean


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
    return AdaRNN(use_bottleneck=True, bottleneck_width=256, n_input=args.d_feat, n_hiddens=n_hiddens,
                  n_output=args.class_num, dropout=args.dropout, model_type=name, len_seq=args.len_seq,
                  trans_loss=args.loss_type).cuda()


def get_student_model(name='AdaRNN'):
    n_hiddens = [args.hidden_size for i in range(args.num_layers)]
    return Student_AdaRNN(use_bottleneck=True, bottleneck_width=256, n_input=args.d_feat - 1, n_hiddens=n_hiddens,
                          n_output=args.class_num, dropout=args.dropout, model_type=name, len_seq=args.len_seq,
                          trans_loss=args.loss_type).cuda()


def get_distill_model(name='AdaRNN'):
    n_hiddens = [args.hidden_size for i in range(args.num_layers)]
    return Distill(use_bottleneck=True, bottleneck_width=256, n_input=args.d_feat, n_hiddens=n_hiddens,
                   n_output=args.class_num, dropout=args.dropout, model_type=name, len_seq=args.len_seq,
                   trans_loss=args.loss_type).cuda()


def train_AdaRNN(args, Teacher_model, Student_model, teacher_optimizer, student_optimizer, train_loader_list, epoch,
                 iter_count, dist_old=None, weight_mat=None, teacher_weight_mat=None, teacher_dist_old=None):
    Student_model.train()
    Teacher_model.train()
    teacher_criterion = nn.MSELoss()
    student_criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    hidden_criteion = TransferLoss(loss_type="adv", input_dim=64)
    bottleneck_criteion = TransferLoss(loss_type="adv", input_dim=256)
    loss_all = []
    loss_1_all = []
    hidden_loss_all = []
    bottleneck_loss_all = []
    dist_mat = torch.zeros(args.num_layers, args.len_seq).cuda()
    teacher_dist_mat = torch.zeros(args.num_layers, args.len_seq).cuda()
    len_loader = np.inf
    for loader in train_loader_list:
        if len(loader) < len_loader:
            len_loader = len(loader)

    for data_all in tqdm(zip(*train_loader_list), total=len_loader):
        teacher_optimizer.zero_grad()
        student_optimizer.zero_grad()
        list_feat = []
        list_label = []
        for data in data_all:
            feature, label_reg = data[0].cuda().float(
            ), data[1].cuda().float(),
            list_feat.append(feature)
            list_label.append(label_reg)

        flag = False
        index = get_index(len(data_all) - 1)
        for temp_index in index:
            s1 = temp_index[0]
            s2 = temp_index[1]
            if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                flag = True
                break
        if flag:
            continue

        total_loss = torch.zeros(1).cuda()
        teacher_total_loss = torch.zeros(1).cuda()
        total_bottleneck_loss = torch.zeros(1).cuda()
        total_hidden_loss = torch.zeros(1).cuda()
        for i in range(len(index)):
            iter_count = iter_count + 1
            feature_s = list_feat[index[i][0]]
            feature_t = list_feat[index[i][1]]

            label_reg_s = list_label[index[i][0]]
            label_reg_t = list_label[index[i][1]]
            feature_all = torch.cat((feature_s, feature_t), 0)
            pred_all, loss_transfer, out_weight_list, fea, fea_bottleneck = Student_model.forward_pre_train(
                feature_all[:, :, :2], len_win=args.len_win)
            teacher_pred_all, teacher_loss_transfer, teacher_out_weight_list, teacher_fea, teacher_fea_bottleneck = Teacher_model.forward_pre_train(
                feature_all, len_win=args.len_win)


            teacher_pred_s = teacher_pred_all[0:feature_s.size(0)]
            teacher_pred_t = teacher_pred_all[feature_s.size(0):]

            teacher_loss_s = teacher_criterion(teacher_pred_s, label_reg_s)
            teacher_loss_t = teacher_criterion(teacher_pred_t, label_reg_t)

            teacher_total_loss = teacher_total_loss + teacher_loss_s + teacher_loss_t + args.dw * teacher_loss_transfer


            pred_s = pred_all[0:feature_s.size(0)]
            pred_t = pred_all[feature_s.size(0):]
            teacher_fea = teacher_fea.detach()
            teacher_fea_bottleneck = teacher_fea_bottleneck.detach()
            hidden_loss = student_criterion(teacher_fea, fea)
            bottleneck_loss = student_criterion(teacher_fea_bottleneck, fea_bottleneck)

            loss_s = student_criterion(pred_s, label_reg_s)
            loss_t = student_criterion(pred_t, label_reg_t)
            loss_l1 = criterion_1(pred_s, label_reg_s)

            iter_count = iter_count + 1


            total_loss = total_loss + loss_s + loss_t + args.dw * loss_transfer + 0.5 * (hidden_loss + bottleneck_loss)

        loss_all.append(
            [total_loss.item(), (loss_s + loss_t).item(), loss_transfer.item()])

        loss_1_all.append(loss_l1.item())
        hidden_loss_all.append(total_hidden_loss.item())
        bottleneck_loss_all.append(total_bottleneck_loss.item())

        teacher_optimizer.zero_grad()
        teacher_total_loss.backward()
        torch.nn.utils.clip_grad_value_(Teacher_model.parameters(), 3.)
        teacher_optimizer.step()

        student_optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_value_(Student_model.parameters(), 3.)

        student_optimizer.step()

    loss = np.array(loss_all).mean(axis=0)
    loss_l1 = np.array(loss_1_all).mean()
    hidden = np.array(hidden_loss_all).mean()
    bottleneck = np.array(bottleneck_loss_all).mean()

    weight_mat = transform_type(out_weight_list)
    return loss, loss_l1, weight_mat, None, teacher_weight_mat, teacher_dist_mat



def train_epoch_transfer_Boosting(Teacher_model, Student_model, teacher_optimizer, student_optimizer, train_loader_list,
                                  epoch, dist_old=None, weight_mat=None):
    Student_model.train()
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    loss_all = []
    loss_1_all = []
    dist_mat = torch.zeros(args.num_layers, args.len_seq).cuda()
    len_loader = np.inf
    for loader in train_loader_list:
        if len(loader) < len_loader:
            len_loader = len(loader)
    for data_all in tqdm(zip(*train_loader_list), total=len_loader):
        student_optimizer.zero_grad()
        list_feat = []
        list_label = []
        for data in data_all:
            feature, label_reg = data[0].cuda().float(
            ), data[1].cuda().float()
            list_feat.append(feature)
            list_label.append(label_reg)
        flag = False
        index = get_index(len(data_all) - 1)
        for temp_index in index:
            s1 = temp_index[0]
            s2 = temp_index[1]
            if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                flag = True
                break
        if flag:
            continue

        total_loss = torch.zeros(1).cuda()
        for i in range(len(index)):
            feature_s = list_feat[index[i][0]]
            feature_t = list_feat[index[i][1]]
            label_reg_s = list_label[index[i][0]]
            label_reg_t = list_label[index[i][1]]
            feature_all = torch.cat((feature_s, feature_t), 0)

            pred_all, loss_transfer, dist, weight_mat, fea, bottelneck_fea = Student_model.forward_Boosting(
                feature_all, weight_mat)
            dist_mat = dist_mat + dist
            pred_s = pred_all[0:feature_s.size(0)]
            pred_t = pred_all[feature_s.size(0):]

            loss_s = criterion(pred_s, label_reg_s)
            loss_t = criterion(pred_t, label_reg_t)
            loss_l1 = criterion_1(pred_s, label_reg_s)

            total_loss = total_loss + loss_s + loss_t + args.dw * loss_transfer

        loss_all.append(
            [total_loss.item(), (loss_s + loss_t).item(), loss_transfer.item()])
        loss_1_all.append(loss_l1.item())
        student_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(Student_model.parameters(), 3.)
        student_optimizer.step()
    loss = np.array(loss_all).mean(axis=0)
    loss_l1 = np.array(loss_1_all).mean()
    if epoch > 0:  # args.pre_epoch:
        weight_mat = Student_model.update_weight_Boosting(
            weight_mat, dist_old, dist_mat)
    return loss, loss_l1, weight_mat, dist_mat


def get_index(num_domain=2):
    index = []
    for i in range(num_domain):
        for j in range(i + 1, num_domain + 1):
            index.append((i, j))
    return index


def train_epoch_transfer(args, model, optimizer, train_loader_list):
    model.train()
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    loss_all = []
    loss_1_all = []
    len_loader = np.inf
    for loader in train_loader_list:
        if len(loader) < len_loader:
            len_loader = len(loader)

    for data_all in tqdm(zip(*train_loader_list), total=len_loader):
        optimizer.zero_grad()
        list_feat = []
        list_label = []
        for data in data_all:
            feature, label_reg = data[0].cuda().float(
            ), data[1].cuda().float()
            list_feat.append(feature)
            list_label.append(label_reg)
        flag = False
        index = get_index(len(data_all) - 1)
        for temp_index in index:
            s1 = temp_index[0]
            s2 = temp_index[1]
            if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                flag = True
                break
        if flag:
            continue

        total_loss = torch.zeros(1).cuda()
        for i in range(len(index)):
            feature_s = list_feat[index[i][0]]
            feature_t = list_feat[index[i][1]]
            label_reg_s = list_label[index[i][0]]
            label_reg_t = list_label[index[i][1]]
            feature_all = torch.cat((feature_s, feature_t), 0)

            pred_all, loss_transfer, out_weight_list = model.forward_pre_train(
                feature_all, len_win=args.len_win)
            pred_s = pred_all[0:feature_s.size(0)]
            pred_t = pred_all[feature_s.size(0):]

            loss_s = criterion(pred_s, label_reg_s)
            loss_t = criterion(pred_t, label_reg_t)
            loss_l1 = criterion_1(pred_s, label_reg_s)

            total_loss = loss_s + loss_t + args.dw * loss_transfer
        loss_all.append(
            [total_loss.item(), (loss_s + loss_t).item(), loss_transfer.item()])
        loss_1_all.append(loss_l1.item())
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()
    loss = np.array(loss_all).mean(axis=0)
    loss_l1 = np.array(loss_1_all).mean()

    return loss, loss_l1, out_weight_list


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


def test_epoch_inference(model, test_loader, valid_file_list, test_file_list, prefix='Test', valid_infusion_start=None,
                         valid_infusion_stop=None, test_infusion_start=None, test_infusion_stop=None,
                         baseline_model=None):
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
            pred = model.predict(feature[:, :, :2])
            if (valid_file_list != None or test_file_list != None) and baseline_model != None:
                base_predict = baseline_model(feature[:, :120, :3], feature[:, -4:, 0]).squeeze()
                if i == 0:
                    base_predict_list = base_predict.cpu().numpy()
                else:
                    base_predict_list = np.hstack((base_predict_list, base_predict.cpu().numpy()))
        loss = criterion(pred, label_reg)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, label_reg)
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()
        PE = ((label_reg - pred) / pred).cpu()
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
            tpredict_list = torch.cat((tpredict_list, pred))
            tlabel_list = torch.cat((tlabel_list, label_reg))
            label_list = np.hstack((label_list, label_reg.cpu().numpy()))
            predict_list = np.hstack((predict_list, pred.cpu().numpy()))
        i = i + 1
    if valid_file_list == None and test_file_list == None:
        loss = total_loss / len(test_loader)
        loss_1 = total_loss_1 / len(test_loader)
        loss_r = total_loss_r / len(test_loader)
        MDPE = total_MDPE / len(test_loader)
        MDAPE = total_MDAPE / len(test_loader)
        MSE_list = [loss, 0, 0, 0]
        RMSE_list = [loss_r, 0, 0, 0]
        MDPE_list = [MDPE, 0, 0, 0]
        MDPAE_list = [MDAPE, 0, 0, 0]
    elif valid_file_list != None:

        induction_end = valid_infusion_start + 600
        loss = total_loss / len(test_loader)

        loss_r = total_loss_r / len(test_loader)
        induction_loss = criterion(tlabel_list[valid_infusion_start:induction_end],
                                   tpredict_list[valid_infusion_start:induction_end])
        maintenance_loss = criterion(tlabel_list[induction_end:valid_infusion_stop],
                                     tpredict_list[induction_end:valid_infusion_stop])
        recovery_loss = criterion(tlabel_list[valid_infusion_stop:], tpredict_list[valid_infusion_stop:])
        induction_loss_r = torch.sqrt(induction_loss)
        maintenance_loss_r = torch.sqrt(maintenance_loss)
        recovery_loss_r = torch.sqrt(recovery_loss)
        MSE_list = [loss, induction_loss, maintenance_loss, recovery_loss]
        RMSE_list = [loss_r, induction_loss_r, maintenance_loss_r, recovery_loss_r]
        PE = ((tlabel_list[valid_infusion_start:] - tpredict_list[valid_infusion_start:]) / tpredict_list[
                                                                                            valid_infusion_start:]).cpu()

        MDPE = np.median(PE) * 100
        MDAPE = np.median(np.abs(PE)) * 100
        induction_PE = ((tlabel_list[valid_infusion_start:induction_end] - tpredict_list[
                                                                           valid_infusion_start:induction_end]) / tpredict_list[
                                                                                                                  valid_infusion_start:induction_end]).cpu()
        mainteance_PE = ((tlabel_list[induction_end:valid_infusion_stop] - tpredict_list[
                                                                           induction_end:valid_infusion_stop]) / tpredict_list[
                                                                                                                 induction_end:valid_infusion_stop]).cpu()
        recovery_PE = ((tlabel_list[valid_infusion_stop:] - tpredict_list[valid_infusion_stop:]) / tpredict_list[
                                                                                                   valid_infusion_stop:]).cpu()
        induction_MDPE = np.median(induction_PE)
        induction_MDAPE = np.median(np.abs(induction_PE)) * 100
        mainteance_MDPE = np.median(mainteance_PE)
        mainteance_MDAPE = np.median(np.abs(mainteance_PE)) * 100
        recovery_MDPE = np.median(recovery_PE)
        recovery_MDAPE = np.median(np.abs(recovery_PE)) * 100
        MDPE_list = [MDPE, induction_MDPE, mainteance_MDPE, recovery_MDPE]
        MDPAE_list = [MDAPE, induction_MDAPE, mainteance_MDAPE, recovery_MDAPE]
        loss_1 = total_loss_1 / len(test_loader)
    elif test_file_list != None:
        induction_end = test_infusion_start + 600

        loss = total_loss / len(test_loader)

        loss_r = total_loss_r / len(test_loader)
        induction_loss = criterion(tlabel_list[test_infusion_start:induction_end],
                                   tpredict_list[test_infusion_start:induction_end])
        maintenance_loss = criterion(tlabel_list[induction_end:test_infusion_stop],
                                     tpredict_list[induction_end:test_infusion_stop])
        recovery_loss = criterion(tlabel_list[test_infusion_stop:], tpredict_list[test_infusion_stop:])
        induction_loss_r = torch.sqrt(induction_loss)
        maintenance_loss_r = torch.sqrt(maintenance_loss)
        recovery_loss_r = torch.sqrt(recovery_loss)
        MSE_list = [loss, induction_loss, maintenance_loss, recovery_loss]
        RMSE_list = [loss_r, induction_loss_r, maintenance_loss_r, recovery_loss_r]
        PE = ((tlabel_list[test_infusion_start:] - tpredict_list[test_infusion_start:]) / tpredict_list[
                                                                                          test_infusion_start:]).cpu()
        # PE = np.array(PE)
        MDPE = np.median(PE) * 100
        MDAPE = np.median(np.abs(PE)) * 100
        induction_PE = ((tlabel_list[test_infusion_start:induction_end] - tpredict_list[
                                                                          test_infusion_start:induction_end]) / tpredict_list[
                                                                                                                test_infusion_start:induction_end]).cpu()
        mainteance_PE = ((tlabel_list[induction_end:test_infusion_stop] - tpredict_list[
                                                                          induction_end:test_infusion_stop]) / tpredict_list[
                                                                                                               induction_end:test_infusion_stop]).cpu()
        recovery_PE = ((tlabel_list[test_infusion_stop:] - tpredict_list[test_infusion_stop:]) / tpredict_list[
                                                                                                 test_infusion_stop:]).cpu()
        induction_MDPE = np.median(induction_PE)
        induction_MDAPE = np.median(np.abs(induction_PE)) * 100
        mainteance_MDPE = np.median(mainteance_PE)
        mainteance_MDAPE = np.median(np.abs(mainteance_PE)) * 100
        recovery_MDPE = np.median(recovery_PE)
        recovery_MDAPE = np.median(np.abs(recovery_PE)) * 100
        loss_1 = total_loss_1 / len(test_loader)
        MDPE_list = [MDPE, induction_MDPE, mainteance_MDPE, recovery_MDPE]
        MDPAE_list = [MDAPE, induction_MDAPE, mainteance_MDAPE, recovery_MDAPE]

    return MSE_list, loss_1, RMSE_list, label_list, predict_list, MDPE_list, MDPAE_list, base_predict_list


def inference(model, data_loader, epoch, valid_file_list=None, test_file_list=None, valid_infusion_start=None,
              valid_infusion_stop=None, test_infusion_start=None, test_infusion_stop=None, baseline_model=None):
    loss, loss_1, loss_r, label_list, predict_list, MDPE, MDAPE, base_predict_list = test_epoch_inference(
        model, data_loader, valid_file_list, test_file_list, prefix='Inference',
        valid_infusion_start=valid_infusion_start,
        valid_infusion_stop=valid_infusion_stop, test_infusion_start=test_infusion_start,
        test_infusion_stop=test_infusion_stop, baseline_model=baseline_model)

    if valid_file_list != None:
        epoch = epoch * 2
        plt.figure(epoch)
        plt.plot(torch.arange(len(label_list)), label_list, label="ground true")
        plt.plot(torch.arange(len(label_list)), predict_list, label="predict")
        plt.xlabel("time(s)")
        plt.ylabel("BIS")
        plt.legend()
        plt.savefig("/home/user02/TUTMING/ming/adarnn/figure/{}.jpg".format(valid_file_list))
        plt.close()
    elif test_file_list != None:
        epoch = epoch * 2 + 1
        plt.figure(epoch)
        plt.plot(torch.arange(len(label_list)), label_list, label="ground true")
        plt.plot(torch.arange(len(label_list)), predict_list, label="predict")
        plt.xlabel("time(s)")
        plt.ylabel("BIS")
        plt.legend()
        plt.savefig("/home/user02/TUTMING/ming/adarnn/figure/{}_.jpg".format(test_file_list))
        plt.close()
    elif valid_file_list == None and test_file_list == None and epoch == 76:
        plt.figure(200)
        plt.plot(torch.arange(len(label_list)), label_list, label="ground true")
        plt.plot(torch.arange(len(label_list)), predict_list, label="predict")
        plt.xlabel("time(s)")
        plt.ylabel("BIS")
        plt.legend()
        plt.savefig("/home/user02/TUTMING/ming/adarnn/figure/train{}.jpg".format(200))
        plt.close()
    return loss, loss_1, loss_r, label_list, predict_list, MDPE, MDAPE


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
    for loader in loaders:
        if i == 0:
            loss, loss_1, loss_r, label_list, predict_list, MDPE, MDAPE = inference(
                model, loader, epoch, baseline_model=baseline_model)
        elif i == 1:
            loss, loss_1, loss_r, label_list, predict_list, MDPE, MDAPE = inference(
                model, loader, epoch, valid_file_list, valid_infusion_start=valid_infusion_start,
                valid_infusion_stop=valid_infusion_stop,
                test_infusion_start=test_infusion_start, test_infusion_stop=test_infusion_stop,
                baseline_model=baseline_model)
        elif i == 2:
            loss, loss_1, loss_r, label_list, predict_list, MDPE, MDAPE = inference(
                model, loader, epoch, test_file_list=test_file_list, valid_infusion_start=valid_infusion_start,
                valid_infusion_stop=valid_infusion_stop,
                test_infusion_start=test_infusion_start, test_infusion_stop=test_infusion_stop,
                baseline_model=baseline_model)
        loss_list.append(loss)
        loss_l1_list.append(loss_1)
        loss_r_list.append(loss_r)
        MDAPE_list.append(MDAPE)
        MDPE_list.append(MDPE)
        i = i + 1
    return loss_list, loss_l1_list, loss_r_list, MDPE_list, MDAPE_list


def transform_type(init_weight):
    weight = torch.ones(args.num_layers, args.len_seq).cuda()
    for i in range(args.num_layers):
        for j in range(args.len_seq):
            weight[i, j] = init_weight[i][j].item()
    return weight


def main_transfer(args):
    print(args)

    output_path = args.outdir + '_' + str(args.num_layers) + '_' + str(args.n_epochs) + args.model_name + 'medicine' + \
                  args.loss_type + '_' + str(args.pre_epoch) + \
                  '_' + str(args.dw) + '_' + str(args.lr) + "_newsplit"
    save_model_name = args.model_name + '_' + args.loss_type + \
                      '_' + str(args.dw) + '_' + str(args.lr) + '.pkl'
    utils.dir_exist(output_path)
    pprint('create loaders...')

    train_loader_list = loader.TDC_nitrain_data_loader(args.len_seq, 24, args.batch_size)

    valid_loader_list, valid_label = loader.testni_data_loader(valid="valid", lensequence=args.len_seq, batch=20, batch_size=128, timestep=1)
    test_loader_list, test_label = loader.testni_data_loader(valid="test", lensequence=args.len_seq, batch=20, batch_size=128, timestep=1)
    valid_file_list, valid_infusion_start, valid_infusion_stop = loader.time_devide(args.test_size, "valid")
    test_file_list, test_infusion_start, test_infusion_stop = loader.time_devide(args.test_size, "test")
    args.log_file = os.path.join(output_path, 'run.log')
    pprint('create model...')
    model_name = data_clean.file_name(output_path)
    Teacher_model = get_teacher_model(args.model_name)
    Student_model = get_student_model(args.model_name)


    num_model = count_parameters(Student_model)
    print('#model params:', num_model)

    teacher_optimizer = optim.Adam(Teacher_model.parameters(), lr=args.lr)
    student_optimizer = optim.Adam(Student_model.parameters(), lr=args.lr)

    best_score = np.inf
    best_epoch, stop_round = 0, 0
    weight_mat, dist_mat, teacher_weight_mat, teacher_dist_mat = None, None, None, None

    iter_count = 0
    for epoch in range(args.n_epochs):
        pprint('Epoch:', epoch)
        pprint('training...')
        if args.model_name in ['Boosting']:
            loss, loss1, weight_mat, dist_mat = train_epoch_transfer_Boosting(
                Teacher_model, Student_model, teacher_optimizer, student_optimizer, train_loader_list, epoch, dist_mat,
                weight_mat)
        elif args.model_name in ['AdaRNN']:
            loss, loss1, weight_mat, dist_mat, teacher_weight_mat, teacher_dist_mat = train_AdaRNN(
                args, Teacher_model, Student_model, teacher_optimizer, student_optimizer, train_loader_list, epoch,
                iter_count, dist_mat, weight_mat, teacher_weight_mat, teacher_dist_mat)
        else:
            print("error in model_name!")
        pprint(loss, loss1)

        pprint("student lr is {}".format(student_optimizer.state_dict()["param_groups"][0]["lr"]))

        pprint('evaluating...')
        train_loss, train_loss_l1, train_loss_r = test_epoch(
            Student_model, train_loader_list[0], prefix='Train')
        val_loss_sum = 0
        val_loss_l1_sum = 0
        val_loss_r_sum = 0
        test_loss_sum = 0
        test_loss_l1_sum = 0
        test_loss_r_sum = 0

        for i in range(args.test_size):
            val_loss, val_loss_l1, val_loss_r = test_epoch(
                Student_model, valid_loader_list[i], prefix='Valid')
            val_loss_sum += val_loss
            val_loss_l1_sum += val_loss_l1
            val_loss_r_sum += val_loss_r
            test_loss, test_loss_l1, test_loss_r = test_epoch(
                Student_model, test_loader_list[i], prefix='Test')
            test_loss_sum += test_loss
            test_loss_l1_sum += test_loss_l1
            test_loss_r_sum += test_loss_r

        val_loss = float(val_loss_sum / args.test_size)
        val_loss_l1 = float(val_loss_l1_sum / args.test_size)
        val_loss_r = float(val_loss_r_sum / args.test_size)
        test_loss = float(test_loss_sum / args.test_size)
        test_loss_l1 = float(test_loss_l1_sum / args.test_size)
        test_loss_r = float(test_loss_r_sum / args.test_size)


        pprint('valid %.6f, test %.6f' %
               (val_loss, test_loss))

        if val_loss < best_score:
            best_score = val_loss
            best_epoch = epoch
            torch.save(Student_model.state_dict(), os.path.join(
                output_path, save_model_name))

        else:
            stop_round += 1
            if stop_round % 10 == 0:
                svae_name = args.model_name + '_' + args.loss_type + \
                            '_' + str(args.dw) + '_' + str(args.lr) + "_" + str(stop_round) + '.pkl'
                torch.save(Student_model.state_dict(), os.path.join(
                    output_path, svae_name))

    pprint('best val score:', best_score, '@', best_epoch)

    baseline_model = None
    train_MDPE = 0
    train_MDAPE = 0
    train_MSE = 0
    valid_MSE = 0
    test_MSE = 0
    test_MDPE = 0
    test_MDAPE = 0
    train_l1 = 0
    test_l1 = 0
    valid_l1 = 0
    valid_MDPE = 0
    valid_MDAPE = 0
    train_RMSE = 0
    valid_RMSE = 0
    test_RMSE = 0
    valid_induction_MDPE = 0
    valid_mainteance_MDPE = 0
    valid_recovery_MDPE = 0
    valid_induction_MDAPE = 0
    valid_mainteance_MDAPE = 0
    valid_recovery_MDAPE = 0
    test_induction_MDPE = 0
    test_mainteance_MDPE = 0
    test_recovery_MDPE = 0
    test_induction_MDAPE = 0
    test_mainteance_MDAPE = 0
    test_recovery_MDAPE = 0
    valid_induction_MSE = 0
    valid_mainteance_MSE = 0
    valid_recovery_MSE = 0
    valid_induction_RMSE = 0
    valid_mainteance_RMSE = 0
    valid_recovery_RMSE = 0
    test_induction_MSE = 0
    test_mainteance_MSE = 0
    test_recovery_MSE = 0
    test_induction_RMSE = 0
    test_mainteance_RMSE = 0
    test_recovery_RMSE = 0

    for i in range(args.test_size):
        loaders = train_loader_list[0], valid_loader_list[i], test_loader_list[i]
        loss_list, loss_l1_list, loss_r_list, MDPE_list, MDAPE_list = inference_all(output_path, Student_model,
                                                                                    os.path.join(
                                                                                        output_path, save_model_name),
                                                                                    loaders, valid_file_list[i],
                                                                                    test_file_list[i],
                                                                                    epoch=i,
                                                                                    valid_infusion_start=
                                                                                    valid_infusion_start[i],
                                                                                    valid_infusion_stop=
                                                                                    valid_infusion_stop[i],
                                                                                    test_infusion_start=
                                                                                    test_infusion_start[i],
                                                                                    test_infusion_stop=
                                                                                    test_infusion_stop[i],
                                                                                    baseline_model=baseline_model)


        train_MSE += loss_list[0][0]
        valid_MSE += loss_list[1][0]
        test_MSE += loss_list[2][0]
        valid_induction_MSE += loss_list[1][1]
        valid_mainteance_MSE += loss_list[1][2]
        valid_recovery_MSE += loss_list[1][3]
        test_induction_MSE += loss_list[2][1]
        test_mainteance_MSE += loss_list[2][2]
        test_recovery_MSE += loss_list[2][3]

        train_l1 += loss_l1_list[0]
        valid_l1 += loss_l1_list[1]
        test_l1 += loss_l1_list[2]

        train_RMSE += loss_r_list[0][0]
        valid_RMSE += loss_r_list[1][0]
        valid_induction_RMSE += loss_r_list[1][1]
        valid_mainteance_RMSE += loss_r_list[1][2]
        valid_recovery_RMSE += loss_r_list[1][3]
        test_RMSE += loss_r_list[2][0]
        test_induction_RMSE += loss_r_list[2][1]
        test_mainteance_RMSE += loss_r_list[2][2]
        test_recovery_RMSE += loss_r_list[2][3]

        train_MDPE += MDPE_list[0][0]
        valid_MDPE += MDPE_list[1][0]
        valid_induction_MDPE += MDPE_list[1][1]
        valid_mainteance_MDPE += MDPE_list[1][2]
        valid_recovery_MDPE += MDPE_list[1][3]
        test_MDPE += MDPE_list[2][0]
        test_induction_MDPE += MDPE_list[2][1]
        test_mainteance_MDPE += MDPE_list[2][2]
        test_recovery_MDPE += MDPE_list[2][3]

        train_MDAPE += MDAPE_list[0][0]
        valid_MDAPE += MDAPE_list[1][0]
        valid_induction_MDAPE += MDAPE_list[1][1]
        valid_mainteance_MDAPE += MDAPE_list[1][2]
        valid_recovery_MDAPE += MDAPE_list[1][3]
        test_MDAPE += MDAPE_list[2][0]
        test_induction_MDAPE += MDAPE_list[2][1]
        test_mainteance_MDAPE += MDAPE_list[2][2]
        test_recovery_MDAPE += MDAPE_list[2][3]

    valid_induction_MSE = float(valid_induction_MSE / float(args.test_size))
    valid_mainteance_MSE = float(valid_mainteance_MSE / float(args.test_size))
    valid_recovery_MSE = float(valid_recovery_MSE / float(args.test_size))
    valid_induction_RMSE = math.sqrt(valid_induction_MSE)
    valid_mainteance_RMSE = math.sqrt(valid_mainteance_MSE)
    valid_recovery_RMSE = math.sqrt(valid_recovery_MSE)
    test_induction_MSE = float(test_induction_MSE / float(args.test_size))
    test_mainteance_MSE = float(test_mainteance_MSE / float(args.test_size))
    test_recovery_MSE = float(test_recovery_MSE / float(args.test_size))
    test_induction_RMSE = math.sqrt(test_induction_MSE)
    test_mainteance_RMSE = math.sqrt(test_mainteance_MSE)
    test_recovery_RMSE = math.sqrt(test_recovery_MSE)

    valid_induction_MDPE = float(valid_induction_MDPE / float(args.test_size))
    valid_mainteance_MDPE = float(valid_mainteance_MDPE / float(args.test_size))
    valid_recovery_MDPE = float(valid_recovery_MDPE / float(args.test_size))
    valid_induction_MDAPE = float(valid_induction_MDAPE / float(args.test_size))
    valid_mainteance_MDAPE = float(valid_mainteance_MDAPE / float(args.test_size))
    valid_recovery_MDAPE = float(valid_recovery_MDAPE / float(args.test_size))
    test_induction_MDPE = float(test_induction_MDPE / float(args.test_size))
    test_mainteance_MDPE = float(test_mainteance_MDPE / float(args.test_size))
    test_recovery_MDPE = float(test_recovery_MDPE / float(args.test_size))
    test_induction_MDAPE = float(test_induction_MDAPE / float(args.test_size))
    test_mainteance_MDAPE = float(test_mainteance_MDAPE / float(args.test_size))
    test_recovery_MDAPE = float(test_recovery_MDAPE / float(args.test_size))

    train_MDPE = float(train_MDPE / float(args.test_size))
    train_MDAPE = float(train_MDAPE / float(args.test_size))
    valid_MDPE = float(valid_MDPE / float(args.test_size))
    valid_MDAPE = float(valid_MDAPE / float(args.test_size))
    test_MDPE = float(test_MDPE / float(args.test_size))
    test_MDAPE = float(test_MDAPE / float(args.test_size))
    train_MSE = float(train_MSE / float(args.test_size))
    valid_MSE = float(valid_MSE / float(args.test_size))
    test_MSE = float(test_MSE / float(args.test_size))
    train_l1 = float(train_l1 / float(args.test_size))
    valid_l1 = float(valid_l1 / float(args.test_size))
    test_l1 = float(test_l1 / float(args.test_size))
    train_RMSE = math.sqrt(train_MSE)
    valid_RMSE = math.sqrt(valid_MSE)
    test_RMSE = math.sqrt(test_MSE)
    pprint('MSE: train %.6f, valid %.6f, test %.6f ' %
           (train_MSE, valid_MSE, test_MSE))
    pprint('L1:  train %.6f, valid %.6f, test %.6f' %
           (train_l1, valid_l1, test_l1))
    pprint('RMSE: train %.6f, valid %.6f, test %.6f ' %
           (train_RMSE, valid_RMSE, test_RMSE))
    pprint('MDPE: train %.6f, valid %.6f, test %.6f ' %
           (train_MDPE, valid_MDPE, test_MDPE))
    pprint('MDAPE: train %.6f, valid %.6f, test %.6f ' %
           (train_MDAPE, valid_MDAPE, test_MDAPE))
    pprint('InductionRMSE: train %.6f, valid %.6f, test %.6f ' %
           (0.0, valid_induction_RMSE, test_induction_RMSE))
    pprint('InductionMDPE: train %.6f, valid %.6f, test %.6f ' %
           (0.0, valid_induction_MDPE, test_induction_MDPE))
    pprint('InductionMDAPE: train %.6f, valid %.6f, test %.6f ' %
           (0.0, valid_induction_MDAPE, test_induction_MDAPE))
    pprint('MainteanceRMSE: train %.6f, valid %.6f, test %.6f ' %
           (0.0, valid_mainteance_RMSE, test_mainteance_RMSE))
    pprint('MainteanceMDPE: train %.6f, valid %.6f, test %.6f ' %
           (0.0, valid_mainteance_MDPE, test_mainteance_MDPE))
    pprint('MainteanceMDAPE: train %.6f, valid %.6f, test %.6f ' %
           (0.0, valid_mainteance_MDAPE, test_mainteance_MDAPE))
    pprint('RecoveryRMSE: train %.6f, valid %.6f, test %.6f ' %
           (0.0, valid_recovery_RMSE, test_recovery_RMSE))
    pprint('RecoveryMDPE: train %.6f, valid %.6f, test %.6f ' %
           (0.0, valid_recovery_MDPE, test_recovery_MDPE))
    pprint('RecoveryMDAPE: train %.6f, valid %.6f, test %.6f ' %
           (0.0, valid_recovery_MDAPE, test_recovery_MDAPE))

    pprint('Finished.')


def get_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='AdaRNN')
    parser.add_argument('--d_feat', type=int, default=3)  # 2

    parser.add_argument('--hidden_size', type=int, default=64)  # 64
    parser.add_argument('--num_layers', type=int, default=5)  # 2
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--class_num', type=int, default=1)
    parser.add_argument('--pre_epoch', type=int, default=10)  # 30, 40, 50

    # training
    parser.add_argument('--n_epochs', type=int, default=21)
    parser.add_argument('--people_size', type=int, default=180)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dw', type=float, default=0.05)  # 1.0, 0.01, 5.0
    parser.add_argument('--loss_type', type=str, default='cosine')
    parser.add_argument('--data_mode', type=str,
                        default='tdc')
    parser.add_argument('--num_domain', type=int, default=3)
    parser.add_argument('--len_seq', type=int, default=120)
    parser.add_argument('--test_size', type=int, default=20)
    parser.add_argument('--train_size', type=int, default=60)

    # other
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--train_path', default="your train data path")
    parser.add_argument('--test_path', default="your test data path")
    parser.add_argument('--valid_path', default="your valid data path")
    parser.add_argument('--outdir', default='your output data path')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log_file', type=str, default='run.log')
    parser.add_argument('--gpu_id', type=int, default=3)
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


    iter_count = 0
    output_path = args.outdir + '_' + str(args.num_layers) + '_' + str(args.n_epochs) + args.model_name + 'medicine' + \
                  args.loss_type + '_' + str(args.pre_epoch) + \
                  '_' + str(args.dw) + '_' + str(args.lr) + "_newsplit"

    main_transfer(args)

