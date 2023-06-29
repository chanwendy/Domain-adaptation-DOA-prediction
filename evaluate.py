import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class Evalulate:
    def __init__(self, x, y, istart, istop, case_num):
        # x:预测结果 y:label len:样本长度
        self.len = case_num
        self.x = x[:self.len]
        self.y = y[:self.len]
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
        self.istrat = istart
        self.istop = istop

        for i in range(case_num):
            self.y[i] = torch.tensor(self.y[i])
            if len(x[i]) < len(y[i]):
                # print("ok")
                self.y[i] = self.y[i][:len(x[i])]
            else:
                self.x[i] = self.x[i][:len(y[i])]

    def rateplot(self, i):
        r = self.x[i] / self.y[i]
        plt.plot(r)

    def ratelist(self):
        r = [0] * self.len
        for i in range(self.len):
            r[i] = self.x[i] / self.y[i]
            plt.plot(r[i])
        plt.show()

    def loss(self, period=0):
        # 0, 1, 2, 3 全阶段,引导期，维持期，复苏期
        t1, t2 = 0, 0

        PE = [0] * self.len
        MSE = [0] * self.len
        MAE = [0] * self.len


        MDPE, MDAPE, RMSE = [0] * self.len, [0] * self.len, [0] * self.len
        for i in range(self.len):
            if period == 0:
                t1 = self.istrat[i]
                t2 = -1
            elif period == 1:
                t1 = self.istrat[i]
                t2 = self.istrat[i] + 600
            elif period == 2:
                t1 = self.istrat[i] + 600
                t2 = self.istop[i]
            elif period == 3:
                t1 = self.istop[i]
                t2 = -1
            PE[i] = ((self.x[i][t1:t2] - self.y[i][t1:t2]) / self.x[i][t1:t2])
            MSE[i] = self.MSE(self.x[i][t1:t2].unsqueeze(-1), self.y[i][t1:t2].unsqueeze(-1))
            MAE[i] = self.MAE(self.x[i][t1:t2].unsqueeze(-1), self.y[i][t1:t2].unsqueeze(-1))
            MDPE[i], MDAPE[i], RMSE[i] = self.estimate(PE=PE[i], MSE=MSE[i])

        return np.mean(MDPE), np.mean(MDAPE), np.mean(RMSE), np.mean(MAE), MDPE, MDAPE, RMSE, MAE, [np.std(MDPE), np.std(MDAPE), np.std(RMSE), np.std(MAE)]

    @staticmethod
    def estimate(PE, MSE):
        """
        :param PE: 每个样本的bis误差（预测bis-真实bis），输入格式：list([样本误差])
        :param MSE: 每个样本的loss， 输入格式：list([样本loss])
        :return: MDPE:误差中位数， MDAPE:绝对误差中位数， RMSE:均方差
        """
        MDPE = np.median(PE) * 100
        MDAPE = np.median(np.abs(PE)) * 100
        RMSE = np.sqrt(MSE)
        return MDPE, MDAPE, RMSE


if __name__ == "__main__":
    x1 = [torch.ones(3000)*10, torch.ones(3000)*110, torch.ones(3000)*56]
    y1 = [torch.ones(3001), torch.ones(3001), torch.ones(3001)]
    e = Evalulate(x1, y1)
    MDPE, MDAPE, RMSE = e.loss()
    e.ratelist()


    # test_MSE = 0
    # test_MDPE = 0
    # test_MDAPE = 0
    # train_RMSE = 0
    # test_RMSE = 0
    # test_induction_MDPE = 0
    # test_mainteance_MDPE = 0
    # test_recovery_MDPE = 0
    # test_induction_MDAPE = 0
    # test_mainteance_MDAPE = 0
    # test_recovery_MDAPE = 0
    # test_induction_MSE = 0
    # test_mainteance_MSE = 0
    # test_recovery_MSE = 0
    # test_induction_RMSE = 0
    # test_mainteance_RMSE = 0
    # test_recovery_RMSE = 0
    # criterion = nn.MSELoss()
    # for i in range(len(test_out)):
    #     test_infusion_start = ist[i]
    #     induction_end = ist[i] + 600
    #     test_infusion_stop = isp[i]
    #     x = min(len(test_out[i]), len(test_label[i]))
    #     loss = criterion(torch.tensor(test_out[i][test_infusion_start:x]), torch.tensor(test_label[i][test_infusion_start:x]))
    #     test_MSE += loss
    #     loss_r = torch.sqrt(loss)
    #     test_RMSE += loss_r
    #     induction_loss = criterion(torch.tensor(test_out[i][test_infusion_start:induction_end]),torch.tensor(test_label[i][test_infusion_start:induction_end]))
    #     test_induction_MSE += induction_loss
    #     maintenance_loss = criterion(torch.tensor(test_out[i][induction_end:test_infusion_stop]),torch.tensor(test_label[i][induction_end:test_infusion_stop]))
    #     test_mainteance_MSE +=maintenance_loss
    #     recovery_loss = criterion(torch.tensor(test_out[i][test_infusion_stop:x]), torch.tensor(test_label[i][test_infusion_stop:x]))
    #     test_recovery_MSE += recovery_loss
    #     induction_loss_r = torch.sqrt(induction_loss)
    #     maintenance_loss_r = torch.sqrt(maintenance_loss)
    #     recovery_loss_r = torch.sqrt(recovery_loss)
    #     test_induction_RMSE += induction_loss_r
    #     test_mainteance_RMSE += maintenance_loss_r
    #     test_recovery_RMSE += recovery_loss_r
    #
    # test_induction_MSE = float(test_induction_MSE / float(len(test_out)))
    # test_mainteance_MSE = float(test_mainteance_MSE / float(len(test_out)))
    # test_recovery_MSE = float(test_recovery_MSE / float(len(test_out)))
    # test_induction_RMSE = math.sqrt(test_induction_MSE)
    # test_mainteance_RMSE = math.sqrt(test_mainteance_MSE)
    # test_recovery_RMSE = math.sqrt(test_recovery_MSE)



