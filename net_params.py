# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/29 20:06
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SmaAt-UNet
@File    : net_params.py
@Language: Python3
'''

from torch import nn
from encoder_cell import EncoderBasicBlock
from decoder_cell import DecoderBasicBlock

encoder_params = [
    [
        nn.MaxPool2d(2, 2, 0),
        nn.MaxPool2d(2, 2, 0),
    ],

    [
        EncoderBasicBlock(5, 64),
        EncoderBasicBlock(64, 128),
        EncoderBasicBlock(128, 128),
    ]
]

decoder_params = [
    [
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Upsample(scale_factor=2, mode='bilinear'),
    ],

    [
        DecoderBasicBlock(256, 64),
        DecoderBasicBlock(128, 64),
        nn.Conv2d(64, 5, 1, 1, 0)
    ]
]