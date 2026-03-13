# -*- coding: utf-8 -*-

from .AODC import AODC
from .loss.genloss import GeneralizedLoss
from .loss.mseloss import MSELoss
        

def build_model(config):
    model = AODC(config)
    # return model, MSELoss(config.FACTOR)
    return model, GeneralizedLoss(config.FACTOR)
