import keras
from gcpds.models.baseline_fcn import fcn_baseline
from gcpds.models.baseline_unet import unet_baseline
from gcpds.models.baseline_res_unet import res_unet_baseline
from convRFF.models.fcn.b_skips import get_model as fcn_b_skips
from convRFF.models.fcn.rff_skips import get_model as fcn_rff_skips
from convRFF.models.res_unet.b_skips import get_model as res_unet_b_skips
from convRFF.models.res_unet.rff_skips import get_model as res_unet_rff_skips
from convRFF.models.unet.b_skips import get_model as unet_b_skips
from convRFF.models.unet.rff_skips import get_model as unet_rff_skips


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


if __name__ == "__main__":
    model = 'u_b'
    input_shape = 224, 224, 1
    out_channels = 1
    kernel_regularizer = None
    get_model(model=model, input_shape=input_shape,
              out_channels=out_channels, kernel_regularizer=kernel_regularizer)
