# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/20 20:50
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SmaAt-UNet
@File    : encoder_cell.py
@Language: Python3
'''

import torch
from torch import nn

class CBAM(nn.Module):

    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.in_channel = in_channel
        self.CA = self._ChannelAttention(self.in_channel)
        self.SA = self._SpatialAttention()

    def forward(self, inputs):
        CA = self.CA(inputs)
        CA_inputs = CA * inputs
        SA = self.SA(CA_inputs)
        SCA_inputs = SA * CA_inputs
        return SCA_inputs

    class _SpatialAttention(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, 7, padding=3)
            self.sigmoid = nn.Sigmoid()

        def forward(self, inputs):
            max, _ = torch.max(inputs, dim=1, keepdim=True)
            mean = torch.mean(inputs, dim=1, keepdim=True)
            combination = torch.cat((max, mean), 1)
            out = self.conv(combination)
            spatial_weights = self.sigmoid(out)
            return spatial_weights

    class _ChannelAttention(nn.Module):

        def __init__(self, in_channel, ratio=16):
            super().__init__()
            self.in_channel = in_channel
            self.ratio = ratio
            self.max = nn.AdaptiveMaxPool2d(1)
            self.mean = nn.AdaptiveAvgPool2d(1)
            self.mlp = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel // self.ratio, 1, bias=False),
                nn.LeakyReLU(0.1),
                nn.Conv2d(self.in_channel // self.ratio, self.in_channel, 1, bias=False)
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, inputs):
            max = self.max(inputs)
            mean = self.mean(inputs)
            max_out = self.mlp(max)
            mean_out = self.mlp(mean)
            channel_weights = self.sigmoid(max_out + mean_out)
            return channel_weights

class EncoderBasicBlock(nn.Module):

    def __init__(self, in_channel, feature_num):
        super(EncoderBasicBlock, self).__init__()
        self.DSC = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, feature_num, 3, padding=1, bias=False),
            nn.GroupNorm(feature_num // 32, feature_num),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feature_num, feature_num, 3, padding=1, groups=feature_num, bias=False),
            nn.Conv2d(feature_num, feature_num, 3, padding=1, bias=False),
            nn.GroupNorm(feature_num // 32, feature_num),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.CBAM = CBAM(feature_num)

    def forward(self, inputs):
        mid = self.DSC(inputs)
        next_out = mid
        short_cut = self.CBAM(mid)
        return next_out, short_cut