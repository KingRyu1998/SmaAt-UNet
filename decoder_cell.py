# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/29 17:48
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SmaAt-UNet
@File    : decoder_cell.py
@Language: Python3
'''

import torch
from torch import nn

class DecoderBasicBlock(nn.Module):
    
    def __init__(self, in_channel, feature_num):
        super(DecoderBasicBlock, self).__init__()
        self.DSC = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, feature_num, 3, padding=1, bias=False),
            nn.GroupNorm(feature_num // 32, feature_num),
            nn.ReLU(),
            nn.Conv2d(feature_num, feature_num, 3, padding=1, groups=feature_num, bias=False),
            nn.Conv2d(feature_num, feature_num, 3, padding=1, bias=False),
            nn.GroupNorm(feature_num // 32, feature_num),
            nn.ReLU()
        )

    def forward(self, inputs, short_cut):
        combination = torch.cat((inputs, short_cut), dim=1)
        outputs = self.DSC(combination)
        return outputs