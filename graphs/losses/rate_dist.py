import logging
import json
import os
from statistics import mean

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F




class TrainRDLoss(nn.Module):
    def __init__(self, lambda_):
        super(TrainRDLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.lambda_ = lambda_
        self.rate = 0
        self.rate1 = 0
        self.rate2 = 0
        self.mse = 0
        self.loss = 0

    def forward(self, x, x_hat, rate):
        self.mse = self.mse_loss(x, x_hat)
        self.rate = torch.sum(rate) / torch.numel(x) * 3
        # self.loss = self.mse + self.lambda_ * self.rate
        self.loss = self.rate + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate

    def forward2(self, x, x_hat, rate1, rate2):
        self.mse = self.mse_loss(x, x_hat)
        self.rate1 = torch.sum(rate1) / torch.numel(x) * 3
        self.rate2 = torch.sum(rate2) / torch.numel(x) * 3
        # self.loss = self.mse + self.lambda_ * self.rate
        self.loss = self.rate1 + self.rate2 + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate1, self.rate2

    def forward3(self, x, x_hat, rate1, rate2list):
        self.mse = self.mse_loss(x, x_hat)
        self.rate1 = torch.sum(rate1) / torch.numel(x) * 3
        self.rate2 = 0
        for i in range(len(rate2list)):
            self.rate2 += torch.sum(rate2list[i]) / torch.numel(x) * 3
        self.loss = self.rate1 + self.rate2 + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate1, self.rate2

    def forward4(self, x, x_hat, rate1list, rate2list):
        self.mse = self.mse_loss(x, x_hat)
        self.rate1 = 0
        for i in range(len(rate1list)):
            self.rate1 += torch.sum(rate1list[i]) / torch.numel(x) * 3
        self.rate2 = 0
        for i in range(len(rate2list)):
            self.rate2 += torch.sum(rate2list[i]) / torch.numel(x) * 3
        self.loss = self.rate1 + self.rate2 + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate1, self.rate2


class TrainRLoss(nn.Module):
    def __init__(self):
        super(TrainRLoss, self).__init__()
        self.rate1 = 0
        self.rate2 = 0
        self.loss = 0

    def forward(self, numel_x, rate):
        self.rate1 = torch.sum(rate) / numel_x * 3
        return self.rate1

    def forward2(self, numel_x, sinfos1, sinfos2):
        self.rate1 = torch.sum(sinfos1) / numel_x * 3
        self.rate2 = torch.sum(sinfos2) / numel_x * 3
        self.loss = self.rate1 + self.rate2
        return self.loss, self.rate1, self.rate2


class TrainRLossList(nn.Module):
    def __init__(self):
        super(TrainRLossList, self).__init__()
        self.rate1 = 0
        self.rate1list = []
        self.rate2 = 0
        self.rate2list = []
        self.loss = 0

    # def forward(self, numel_x, sinfoslist):
    #     self.rate1 = 0
    #     self.rate1list = []
    #     for i in range(len(sinfoslist)):
    #         rate1 = torch.sum(sinfoslist[i]) / numel_x * 3
    #         self.rate1 += rate1
    #         # self.rate1list.append(rate1)
    #         self.rate1list.append(rate1.item())  # must do item() here, ow all history/grads is stored in list
    #     return self.rate1, self.rate1list
    def forward(self, numel_x, sinfoslist):
        self.rate1 = 0
        self.rate1list = []
        for i in range(len(sinfoslist)):
            rate1s_for_each_band_clrch = torch.sum(sinfoslist[i], dim=(0, 2, 3)) / numel_x * 3
            self.rate1list.append([r1.item() for r1 in rate1s_for_each_band_clrch])  # must do item() here, ow all history/grads is stored in list
            self.rate1 += torch.sum(rate1s_for_each_band_clrch)
        return self.rate1, self.rate1list

    def forward2(self, numel_x, sinfoslist1, sinfoslist2):
        self.rate1 = 0
        self.rate1list = []
        for i in range(len(sinfoslist1)):
            rate1 = torch.sum(sinfoslist1[i]) / numel_x * 3
            self.rate1 += rate1  # * (i+1)  # CAUTION ! giving more weight to high pass subbands entropy
            # self.rate1list.append(rate1)
            self.rate1list.append(rate1.item())  # must do item() here, ow all history/grads is stored in list
        self.rate2 = 0
        self.rate2list = []
        for i in range(len(sinfoslist2)):
            rate2 = torch.sum(sinfoslist2[i]) / numel_x * 3
            self.rate2 += rate2  # * (i+1)  # CAUTION !
            # self.rate2list.append(rate2)
            self.rate2list.append(rate2.item())  # must do item() here, ow all history/grads is stored in list
        self.loss = self.rate1 + self.rate2
        return self.loss, self.rate1, self.rate2, self.rate1list, self.rate2list  #lists go to logger, other will be used to backprop


class CompressionRLossList(nn.Module):
    def __init__(self):
        super(CompressionRLossList, self).__init__()
        self.rate1list = []

    def forward(self, numel_x, bytestream_list):
        self.rate1list = []
        for i in range(len(bytestream_list)):
            rate1s_for_each_band_clrch = [len(bytestream_b_clr)*8 / numel_x * 3 for bytestream_b_clr in bytestream_list[i]]
            self.rate1list.append(rate1s_for_each_band_clrch)
        return self.rate1list
