# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/29 21:40
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SmaAt-UNet
@File    : decoder.py
@Language: Python3
'''

from torch import nn

class Decoder(nn.Module):

    def __init__(self, params):
        super(Decoder, self).__init__()
        self.steps = params[0]
        self.layers = params[1]
        self.blocks = len(self.steps)
        for idx, (layer, step) in enumerate(zip(self.layers[:-1], self.steps)):
            setattr(self, 'layer' + str(idx), layer)
            setattr(self, 'step' + str(idx), step)
        setattr(self, 'last_layer', self.layers[-1])

    def forward_by_stage(self, inputs, short_cuts, layer, step):
        inputs = step(inputs)
        outputs = layer(inputs, short_cuts)
        return outputs

    def forward(self, short_cuts):
        inputs = self.forward_by_stage(short_cuts[-1],
                                       short_cuts[-2],
                                       getattr(self, 'layer0'),
                                       getattr(self, 'step0'),
                                       )
        for idx in range(1, self.blocks):
            outputs = self.forward_by_stage(inputs,
                                            short_cuts[-idx-2],
                                            getattr(self, 'layer' + str(idx)),
                                            getattr(self, 'step' + str(idx)))
            inputs = outputs
        last_layer = getattr(self, 'last_layer')
        preds = last_layer(inputs)
        return preds




