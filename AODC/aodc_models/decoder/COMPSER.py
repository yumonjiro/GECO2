# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class COMPSER(nn.Module):
    def __init__(self):
        super(COMPSER, self).__init__()

        self.decoder = nn.Sequential(nn.Conv2d(512, 256, (3, 3), padding=(1, 1), bias=True),
                                     nn.ReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(256, 128, (3, 3), padding=(1, 1), bias=True),
                                     nn.ReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True),
                                     nn.ReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(64, 1, 1),
                                     nn.ReLU(),
                                     )

        self.weights_normal_init(self.decoder, dev=0.01)

    def forward(self, x):
        x = self.decoder(x)
        return x

    def weights_normal_init(self, model, dev=0.01):
        if isinstance(model, list):
            for m in model:
                self.weights_normal_init(m, dev)
        else:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)
