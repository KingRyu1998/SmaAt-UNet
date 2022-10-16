# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/29 21:40
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SmaAt-UNet
@File    : encoder.py
@Language: Python3
'''
import torch
from torch import nn

class Encoder(nn.Module):

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.steps = params[0]
        self.layers = params[1][:-1]
        self.last_layer = params[1][-1]
        self.blocks = len(self.layers)
        for idx, (step, layer) in enumerate(zip(self.steps, self.layers)):
            setattr(self, 'step'+str(idx), step)
            setattr(self, 'layer'+str(idx), layer)
        setattr(self, 'last_layer', self.last_layer)

    def forward_by_step(self, inputs, layer, step):
        outputs, short_cuts = layer(inputs)
        outputs = step(outputs)
        return outputs, short_cuts

    def forward(self, inputs):
        all_short_cuts = []
        for index in range(self.blocks):
            inputs, short_cuts = self.forward_by_step(inputs,
                                                       getattr(self, 'layer'+str(index)),
                                                       getattr(self, 'step'+str(index))
                                                       )
            all_short_cuts.append(short_cuts)
        _, short_cuts = getattr(self, 'last_layer')(inputs)
        all_short_cuts.append(short_cuts)
        return tuple(all_short_cuts)
