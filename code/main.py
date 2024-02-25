from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from code.gcpds.DataSet.convRFF.train import get_compile_parameters
from keras.applications.vgg16 import VGG16
from gcpds.Metrics import (dice,
                           jaccard,
                           sensitivity,
                           specificity)
from keras.layers import Lambda
from keras.models import Model
from keras import regularizers
from functools import partial
import keras.layers as layers
import keras.backend as K
from code.gcpds.DataSet.convRFF import data
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import wandb
import os


def init_wandb():
    sweep_config = {
        'method': 'grid'
    }

    parameters_dict = {
        'model': {
            'values': ['f_b', 'r_b', 'u_b', 'f_b_s_m3', 'r_b_s_m3', 'u_b_s_m3', 'f_b_s', 'f_r_s',
                       'f_r_s_m1', 'r_b_s', 'r_r_s', 'r_r_s_m1', 'u_r_s_m1', 'u_r_s', 'u_b_s', 'u_r_b']
        },
        'dataset': {
            'values': ['infrared_thermal_feet_80x60_NoAug16']
        }
    }
    sweep_config['parameters'] = parameters_dict
    # 12fe25998ca888a5d462ea747e3d711af79fc285
    os.environ["WANDB_API_KEY"] = "12fe25998ca888a5d462ea747e3d711af79fc285"
    sweep_id = wandb.sweep(sweep_config, project='MamitasLowResolution')


def resizen():
    def resizeAndlowresolution(shape=(256, 256)):
        def func(img, mask):
            return tf.image.resize(tf.image.resize(img, (80, 60)), shape, method='area'), tf.image.resize(mask, (224, 224))
        return func

    data.resize = resizeAndlowresolution


if __name__ == "__main__":
    resize = False
    if resize:
        resizen()
    pass
    # init_wandb()
