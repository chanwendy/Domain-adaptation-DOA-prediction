
import ipdb
import torch
import torch.nn as nn
from ourmodel.loss_transfer import TransferLoss
import torch.nn.functional as F


class AdaRNN(nn.Module):
    """
    model_type:  'Boosting', 'AdaRNN'
    """

    def __init__(self, use_bottleneck=False, bottleneck_width=256, n_input=128, n_hiddens=[64, 64], n_output=6, dropout=0.0, len_seq=9, model_type='AdaRNN', trans_loss='mmd'):
        super(AdaRNN, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.n_input = n_input
        self.num_layers = len(n_hiddens)
        self.hiddens = n_hiddens
        self.n_output = n_output
        self.model_type = model_type
        self.trans_loss = trans_loss
        self.len_seq = len_seq
        # 输入个数
        in_size = self.n_input
        # 第一个sequential
        # 相当于先创建GRU，然后一个个加入modulist中，最后将modulist中的GRU再一个个放入sequential中。
        features = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout
            )
            features.append(rnn)
            in_size = hidden
            # 1相当于在sequential中先加入两个GRU
        self.features = nn.Sequential(*features)
        # 这里创建第二个sequential
        if use_bottleneck == True:  # finance
            # 2接着在GRU后面接2个feedforward
            self.bottleneck = nn.Sequential(
                nn.Linear(n_hiddens[-1] + 4, bottleneck_width),
                # nn.Linear(n_hiddens[-1], bottleneck_width),
                nn.ReLU(),
                nn.Linear(bottleneck_width, bottleneck_width),
                # nn.BatchNorm1d(bottleneck_width),
                nn.ReLU(),
                # nn.Sigmoid(),
                nn.Dropout(),
            )
            # 对两个线性层的权重初始化
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            self.bottleneck[2].weight.data.normal_(0, 0.005)
            self.bottleneck[2].bias.data.fill_(0.1)
            self.fc = nn.Linear(bottleneck_width, n_output)
            torch.nn.init.xavier_normal_(self.fc.weight)
        else:
            # 输出
            self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)

        if self.model_type == 'AdaRNN':
            gate = nn.ModuleList()
            # gwl = nn.ModuleList()
            for i in range(len(n_hiddens)):
                gate_weight = nn.Linear(
                    len_seq * self.hiddens[i] * 2, len_seq)
                # gwl_weight = nn.Linear(len_seq, len_seq)
                gate.append(gate_weight)
                # gwl.append(gwl_weight)
            self.gate = gate
            # self.gwl = gwl

            # bnlst = nn.ModuleList()
            # for i in range(len(n_hiddens)):
            #     bnlst.append(nn.BatchNorm1d(len_seq))
            # self.bn_lst = bnlst
            self.softmax = torch.nn.Softmax(dim=0)
            self.init_layers()

    def init_layers(self):
        for i in range(len(self.hiddens)):
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)
            # self.gwl[i].weight.data.normal_(0, 0.05)
            # self.gwl[i].bias.data.fill_(0.0)






    def forward_pre_train(self, x, len_win=0):
        batch = x.shape[0]
        # print("x shape {}".format(x.shape))
        # print(x.shape)        torch.Size([128, 180, 2])     (batch_size*2 (两个域的数据), seqlenght,feature)
        # 获取到gru的输出结果和权重
        out = self.gru_features(x)
        # 获取经过GRU后的结果
        fea = out[0]       # fea shape is torch.Size([128, 180, 180]) #  (batch_size*2 (两个域的数据), seqlength, hiddensize)
        # print("fea shape is {}".format(fea.shape))
        # fea = fea.reshape(-1, 180)
        # print("fea[:, -1, :] shape is {} ".format(fea[:, -1, :].shape))     # (batch_size*2, hiddensize)
        # 将GRU获得的结果经过几个feedfoward获取到FC输出
        output = fea[:, -1, :]
        body = x[:, -4:, 0].reshape(batch, 4)
        output = torch.cat((output, body), 1)
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(output)            # fea[:, -1, :] is torch.Size([128, 180])
            # print("fea_bottleneck shape is {}".format(fea_bottleneck.shape))
            fc_out = self.fc(fea_bottleneck).squeeze()
            # print("fc_out.shape is {}".format(fc_out.shape))
        else:
            fc_out = self.fc_out(fea).squeeze()

        out_list_all, out_weight_list = out[1], out[2]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).cuda()
        # 将GRU获得的每一个输出，对半分，获取到源数据特征和目标数据特征
        for i in range(len(out_list_s)):
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2])
            h_start = 0
            #？
            for j in range(h_start, self.len_seq, 1):
                i_start = j - len_win if j - len_win >= 0 else 0
                i_end = j + len_win if j + len_win < self.len_seq else self.len_seq - 1
                for k in range(i_start, i_end + 1):
                    weight = out_weight_list[i][j] if self.model_type == 'AdaRNN' else 1 / (
                        self.len_seq - h_start) * (2 * len_win + 1)
                    loss_transfer = loss_transfer + weight * criterion_transder.compute(
                        out_list_s[i][:, j, :], out_list_t[i][:, k, :])
        return fc_out, loss_transfer, out_weight_list, fea, fea_bottleneck
        # return fc_out, loss_transfer, out_weight_list, fea[:, -1, :], fea_bottleneck
    # 获取到gru的输出结果和权重
    def gru_features(self, x, predict=False):
        # ipdb.set_trace()
        batch = x.shape[0]
        x_input = x[:, :self.len_seq, :]
        # x_input = x[:, :180, 0].reshape(batch, 180, 1)
        out = None
        out_lis = []
        out_weight_list = [] if (
             self.model_type == 'AdaRNN') else None
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            # print(_[0].shape)
            # z = out[:, -1, :]
            # y = _[0]
            # if torch.equal(z, y):
            #     print('okokokokokok')
            # print("out shape is {}".format(out.shape))
            # print("_ shape is {}".format(_.shape))  # _ (1, 128, 64)
            x_input = out
            out_lis.append(out)
            if self.model_type == 'AdaRNN' and predict == False:
                out_gate = self.process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
                # return3个只用1个接受，则返回的是元祖形式，元素分别为三个变量
        return out, out_lis, out_weight_list

    def process_gate_weight(self, out, index):
        x_s = out[0: int(out.shape[0]//2)]

        x_t = out[out.shape[0]//2: out.shape[0]]
        x_all = torch.cat((x_s, x_t), 2)
        # reshape成2维
        x_all = x_all.view(x_all.shape[0], -1)
        # 进行batchnorm
        # weight = torch.sigmoid(self.bn_lst[index](
        #     self.gate[index](x_all.float())))
        weight = torch.sigmoid(
            self.gate[index](x_all.float()))    # (batch_size, len_seq)
        # 取平均
        # weight = torch.sigmoid(self.gwl[index](weight.float()))
        # weight = weight_1 + weight
        weight = torch.mean(weight, dim=0)      # (len_seq)
        # 取平均
        # weight = torch.mean(weight, dim=0)
        res = self.softmax(weight).squeeze()
        return res

    # 提取源数据特征和目标数据特征
    def get_features(self, output_list):
        fea_list_src, fea_list_tar = [], []
        for fea in output_list:
            fea_list_src.append(fea[0: fea.size(0) // 2])
            fea_list_tar.append(fea[fea.size(0) // 2:])
        return fea_list_src, fea_list_tar

    # For Boosting-based
    def forward_Boosting(self, x, weight_mat=None):
        out = self.gru_features(x)
        fea = out[0]
        batch = x.shape[0]
        output = fea[:, -1, :]
        body = x[:, -4:, 0].reshape(batch, 4)
        output = torch.cat((output, body), 1)
        # fea = fea.reshape(-1, 180)
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(output)
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea).squeeze()

        out_list_all = out[1]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).cuda()
        if weight_mat is None:
            weight = (1.0 / self.len_seq *
                      torch.ones(self.num_layers, self.len_seq)).cuda()
        else:
            weight = weight_mat
        dist_mat = torch.zeros(self.num_layers, self.len_seq).cuda()
        # 计算loss和分布距离差异大小
        for i in range(len(out_list_s)):
            # 计算源域loss
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2])
            # # 计算区间分布差异
            for j in range(self.len_seq):
                loss_trans = criterion_transder.compute(
                    out_list_s[i][:, j, :], out_list_t[i][:, j, :])
                loss_transfer = loss_transfer + weight[i, j] * loss_trans
                dist_mat[i, j] = loss_trans
        return fc_out, loss_transfer, dist_mat, weight, fea[:, -1, :], fea_bottleneck

    # For Boosting-based
    # 根据boosting更新权重（α）
    # For Boosting-based
    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        epsilon = 1e-12
        # detach() 会返回一个新的Tensor对象，不会在反向传播中出现，是相当于复制了一个变量，将它原本requires_grad=True变为了requires_grad=False
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * \
            (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, self.len_seq)
        return weight_mat

    #     # 获取预测输出
    def predict(self, x):

        out = self.gru_features(x, predict=True)
        fea = out[0]
        batch = x.shape[0]
        output = fea[:, -1, :]
        body = x[:, -4:, 0].reshape(batch, 4)
        output = torch.cat((output, body), 1)
        # fea = fea.reshape(-1, 180)
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(output)
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea).squeeze()
        return fc_out



class Student_AdaRNN(nn.Module):
    """
    model_type:  'Boosting', 'AdaRNN'
    """

    def __init__(self, use_bottleneck=False, bottleneck_width=256, n_input=128, n_hiddens=[64, 64], n_output=6, dropout=0.0, len_seq=9, model_type='AdaRNN', trans_loss='mmd'):
        super(Student_AdaRNN, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.n_input = n_input
        self.num_layers = len(n_hiddens)
        self.hiddens = n_hiddens
        self.n_output = n_output
        self.model_type = model_type
        self.trans_loss = trans_loss
        self.len_seq = len_seq
        # 输入个数
        in_size = self.n_input
        # 第一个sequential
        # 相当于先创建GRU，然后一个个加入modulist中，最后将modulist中的GRU再一个个放入sequential中。
        features = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout
            )
            features.append(rnn)
            in_size = hidden
            # 1相当于在sequential中先加入两个GRU
        self.features = nn.Sequential(*features)
        self.relu = nn.ReLU()
        # 这里创建第二个sequential
        if use_bottleneck == True:  # finance
            # 2接着在GRU后面接2个feedforward
            self.bottleneck = nn.Sequential(
                nn.Linear(n_hiddens[-1] + 4, bottleneck_width),
                # nn.Linear(n_hiddens[-1], bottleneck_width),
                nn.ReLU(),
                nn.Linear(bottleneck_width, bottleneck_width),
                # nn.BatchNorm1d(bottleneck_width),
                nn.ReLU(),
                # nn.Sigmoid(),
                nn.Dropout(),
            )
            # 对两个线性层的权重初始化
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            self.bottleneck[2].weight.data.normal_(0, 0.005)
            self.bottleneck[2].bias.data.fill_(0.1)
            self.fc = nn.Linear(bottleneck_width, n_output)
            torch.nn.init.xavier_normal_(self.fc.weight)
        else:
            # 输出
            self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)

        if self.model_type == 'AdaRNN':
            gate = nn.ModuleList()
            # gwl = nn.ModuleList()
            for i in range(len(n_hiddens)):
                gate_weight = nn.Linear(
                    len_seq * self.hiddens[i] * 2, len_seq)
                # gwl_weight = nn.Linear(len_seq, len_seq)
                gate.append(gate_weight)
                # gwl.append(gwl_weight)
            self.gate = gate
            # self.gwl = gwl

            # bnlst = nn.ModuleList()
            # for i in range(len(n_hiddens)):
            #     bnlst.append(nn.BatchNorm1d(len_seq))
            # self.bn_lst = bnlst
            self.softmax = torch.nn.Softmax(dim=0)
            self.init_layers()

    def init_layers(self):
        for i in range(len(self.hiddens)):
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)
            # self.gwl[i].weight.data.normal_(0, 0.05)
            # self.gwl[i].bias.data.fill_(0.0)

    def forward_pre_train(self, x, len_win=0):
        batch = x.shape[0]
        # print("x shape {}".format(x.shape))
        # print(x.shape)        torch.Size([128, 180, 2])     (batch_size*2 (两个域的数据), seqlenght,feature)
        # 获取到gru的输出结果和权重
        out = self.gru_features(x)
        # 获取经过GRU后的结果
        fea = out[0]       # fea shape is torch.Size([128, 180, 180]) #  (batch_size*2 (两个域的数据), seqlength, hiddensize)
        # print("fea shape is {}".format(fea.shape))
        # fea = fea.reshape(-1, 180)
        # print("fea[:, -1, :] shape is {} ".format(fea[:, -1, :].shape))     # (batch_size*2, hiddensize)
        # 将GRU获得的结果经过几个feedfoward获取到FC输出
        output = fea[:, -1, :]
        body = x[:, -4:, 0].reshape(batch, 4)
        output = torch.cat((output, body), 1)
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(output)            # fea[:, -1, :] is torch.Size([128, 180])
            # print("fea_bottleneck shape is {}".format(fea_bottleneck.shape))
            fc_out = self.fc(fea_bottleneck).squeeze()
            # print("fc_out.shape is {}".format(fc_out.shape))
        else:
            fc_out = self.fc_out(fea).squeeze()

        out_list_all, out_weight_list = out[1], out[2]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).cuda()
        # 将GRU获得的每一个输出，对半分，获取到源数据特征和目标数据特征
        for i in range(len(out_list_s)):
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2])
            h_start = 0
            #？
            for j in range(h_start, self.len_seq, 1):
                i_start = j - len_win if j - len_win >= 0 else 0
                i_end = j + len_win if j + len_win < self.len_seq else self.len_seq - 1
                for k in range(i_start, i_end + 1):
                    weight = out_weight_list[i][j] if self.model_type == 'AdaRNN' else 1 / (
                        self.len_seq - h_start) * (2 * len_win + 1)
                    loss_transfer = loss_transfer + weight * criterion_transder.compute(
                        out_list_s[i][:, j, :], out_list_t[i][:, k, :])
        # return fc_out, loss_transfer, out_weight_list, fea[:, -1, :], fea_bottleneck
        return fc_out, loss_transfer, out_weight_list, fea, fea_bottleneck
    # 获取到gru的输出结果和权重
    def gru_features(self, x, predict=False):
        # ipdb.set_trace()
        batch = x.shape[0]
        x_input = x[:, :self.len_seq, :]
        # x_input = x[:, :180, 0].reshape(batch, 180, 1)
        out = None
        out_lis = []
        out_weight_list = [] if (
             self.model_type == 'AdaRNN') else None
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            # print(_[0].shape)
            # z = out[:, -1, :]
            # y = _[0]
            # if torch.equal(z, y):
            #     print('okokokokokok')
            # print("out shape is {}".format(out.shape))
            # print("_ shape is {}".format(_.shape))  # _ (1, 128, 64)
            x_input = out
            out_lis.append(out)
            if self.model_type == 'AdaRNN' and predict == False:
                out_gate = self.process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
                # return3个只用1个接受，则返回的是元祖形式，元素分别为三个变量
        return out, out_lis, out_weight_list

    def process_gate_weight(self, out, index):
        x_s = out[0: int(out.shape[0]//2)]

        x_t = out[out.shape[0]//2: out.shape[0]]
        x_all = torch.cat((x_s, x_t), 2)
        # reshape成2维
        x_all = x_all.view(x_all.shape[0], -1)
        # 进行batchnorm
        # weight = torch.sigmoid(self.bn_lst[index](
        #     self.gate[index](x_all.float())))
        weight = torch.sigmoid(
            self.gate[index](x_all.float()))    # (batch_size, len_seq)
        # 取平均
        # weight = torch.sigmoid(self.gwl[index](weight.float()))
        # weight = weight_1 + weight
        weight = torch.mean(weight, dim=0)      # (len_seq)
        # 取平均
        # weight = torch.mean(weight, dim=0)
        res = self.softmax(weight).squeeze()
        return res

    # 提取源数据特征和目标数据特征
    def get_features(self, output_list):
        fea_list_src, fea_list_tar = [], []
        for fea in output_list:
            fea_list_src.append(fea[0: fea.size(0) // 2])
            fea_list_tar.append(fea[fea.size(0) // 2:])
        return fea_list_src, fea_list_tar

    # For Boosting-based
    def forward_Boosting(self, x, weight_mat=None):
        out = self.gru_features(x)
        fea = out[0]
        batch = x.shape[0]
        output = fea[:, -1, :]
        body = x[:, -4:, 0].reshape(batch, 4)
        output = torch.cat((output, body), 1)
        # fea = fea.reshape(-1, 180)
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(output)
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea).squeeze()

        out_list_all = out[1]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).cuda()
        if weight_mat is None:
            weight = (1.0 / self.len_seq *
                      torch.ones(self.num_layers, self.len_seq)).cuda()
        else:
            weight = weight_mat
        dist_mat = torch.zeros(self.num_layers, self.len_seq).cuda()
        # 计算loss和分布距离差异大小
        for i in range(len(out_list_s)):
            # 计算源域loss
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2])
            # # 计算区间分布差异
            for j in range(self.len_seq):
                loss_trans = criterion_transder.compute(
                    out_list_s[i][:, j, :], out_list_t[i][:, j, :])
                loss_transfer = loss_transfer + weight[i, j] * loss_trans
                dist_mat[i, j] = loss_trans
        return fc_out, loss_transfer, dist_mat, weight, fea[:, -1, :], fea_bottleneck

    # For Boosting-based
    # 根据boosting更新权重（α）
    # For Boosting-based

    #     # 获取预测输出
    def predict(self, x):

        out = self.gru_features(x, predict=True)
        fea = out[0]
        batch = x.shape[0]
        output = fea[:, -1, :]
        body = x[:, -4:, 0].reshape(batch, 4)
        output = torch.cat((output, body), 1)
        # fea = fea.reshape(-1, 180)
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(output)
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea).squeeze()
        return fc_out

#
class Distill(nn.Module):

    def __init__(self, use_bottleneck=True, bottleneck_width=256, n_input=3, n_hiddens=[64, 64], n_output=1, dropout=0.0, len_seq=120, model_type='AdaRNN', trans_loss='mmd'):
        super(Distill, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.n_input = n_input
        self.num_layers = len(n_hiddens)
        self.hiddens = n_hiddens
        self.n_output = n_output
        self.model_type = model_type
        self.trans_loss = trans_loss
        self.len_seq = len_seq
        self.teacher = AdaRNN(self.use_bottleneck, bottleneck_width, n_input, n_hiddens, n_output, dropout, len_seq, model_type, trans_loss)
        self.student = Student_AdaRNN(self.use_bottleneck, bottleneck_width, 2, n_hiddens, n_output, dropout, len_seq,model_type, trans_loss)
        # self.student = Student_AdaRNN(self.use_bottleneck, bottleneck_width, 2, [64], n_output, dropout, len_seq, model_type, trans_loss)



    def forward(self, x, flag):

        if flag == "Teacher":
            pred_all, loss_transfer, out_weight_list, fea, fea_bottleneck = self.teacher.forward_pre_train(x)
        elif flag == "Student":
            pred_all, loss_transfer, out_weight_list, fea, fea_bottleneck = self.student.forward_pre_train(x)

        return pred_all, loss_transfer, out_weight_list, fea, fea_bottleneck

    def backward(self, total_loss, optimizer, flag):
        if flag == "Teacher":
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.teacher.parameters(), 3.)
            optimizer.step()
        elif flag == "Student":
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.student.parameters(), 3.)
            optimizer.step()

    def predict(self, x):
        output = self.student.predict(x)
        return output






