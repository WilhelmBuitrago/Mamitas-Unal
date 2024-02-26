import keras

from gcpds.loss.dice import DiceCoefficient
from gcpds.Metrics.dice import DiceCoefficientMetric
import gcpds.Metrics.jaccard as Jaccard
import gcpds.Metrics.sensitivity as Sensitivity
import gcpds.Metrics.specificity as Specificity

from gcpds.models import baseline_fcn as fcn_baseline
from gcpds.models import baseline_unet as unet_baseline
from gcpds.models import baseline_res_unet as res_unet_baseline
from convRFF.models.fcn import b_skips as fcn_b_skips
from convRFF.models.fcn import rff_skips as fcn_rff_skips
from convRFF.models.res_unet import b_skips as res_unet_b_skips
from convRFF.models.res_unet import rff_skips as res_unet_rff_skips
from convRFF.models.unet import b_skips as unet_b_skips
from convRFF.models.unet import rff_skips as unet_rff_skips


def get_model(model, input_shape, out_channels, kernel_regularizer):
    MODELS = {'u_b': unet_baseline(input_shape=input_shape, out_channels=out_channels),
              'u_b_s': unet_b_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer),
              'u_r_s': unet_rff_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=3),
              'u_r_s_m1': unet_rff_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=1),
              'u_b_s_m3': unet_b_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=3),
              'f_b': fcn_baseline(input_shape=input_shape, out_channels=out_channels),
              'r_b': res_unet_baseline(input_shape=input_shape, out_channels=out_channels),
              'f_b_s': fcn_b_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer),
              'f_r_s': fcn_rff_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer),
              'f_r_s_m1': fcn_rff_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=1),
              'f_b_s_m3': fcn_b_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=3),
              'r_b_s': res_unet_b_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer),
              'r_b_s_m3': res_unet_b_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=3),
              'r_r_s': res_unet_rff_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer),
              'r_r_s_m1': res_unet_rff_skips(input_shape=input_shape, out_channels=out_channels, kernel_regularizer=kernel_regularizer, mul_dim=1)
              }
    return MODELS[model]
