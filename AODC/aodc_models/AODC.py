# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .encoder import encoder
from .decoder import COMPSER
import random
import torch.nn.functional as F


class AODC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.zero_shot = config.ZERO_SHOT
        self.few_shot = config.FEW_SHOT
        self.encoder = encoder.encoder(config.ZERO_SHOT, config.FEW_SHOT)
        self.decoder = COMPSER.COMPSER()

    def forward(self, image, boxes, mode, categories):
        b, _, imh, imw = image.shape
        if self.few_shot:
            if mode == 'train':
                scale = random.randint(6, 15) / 10.0
            else:
                b_size = torch.stack((boxes[:, 4] - boxes[:, 2], boxes[:, 3] - boxes[:, 1]), dim=-1)
                bs_mean = b_size.view(-1, 3, 2).float().mean(dim=1, keepdim=False)
                scale = torch.clamp(32 - bs_mean.min(), min=0, max=24.0) / 16.0 + 1
            image = F.interpolate(image, size=(int(imh * scale), int(imw * scale)), mode='bilinear', align_corners=True)
            boxes[:, 1: 5] = boxes[:, 1: 5] * scale
        fea = self.encoder(image, boxes, categories)

        denmap = self.decoder(fea)
        if self.few_shot:
            original_sum = denmap.sum(dim=(1, 2, 3), keepdim=True)
            denmap = F.interpolate(denmap, size=(imh, imw), mode='bilinear', align_corners=True)
            denmap = denmap / (denmap.sum(dim=(1, 2, 3), keepdim=True) + 1e-5) * original_sum
        
        return denmap
    