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


class load_models():
    def compile_parameters():
        return {'loss': DiceCoefficient(),
                'optimizer': keras.optimizers.Adam(learning_rate=1e-3),
                'metrics': [Jaccard(),
                            Jaccard(name='Jaccard_0', target_class=0),
                            Jaccard(name='Jaccard_1', target_class=1),
                            Jaccard(name='Jaccard_2', target_class=2),
                            Sensitivity(),
                            Sensitivity(name='Sensitivity_0', target_class=0),
                            Sensitivity(name='Sensitivity_1', target_class=1),
                            Sensitivity(name='Sensitivity_2', target_class=2),
                            Specificity(),
                            Specificity(name='Specificity_0', target_class=0),
                            Specificity(name='Specificity_1', target_class=1),
                            Specificity(name='Specificity_2', target_class=2),
                            DiceCoefficientMetric(),
                            DiceCoefficientMetric(
                                name='DiceCoeficienteMetric_0', target_class=0),
                            DiceCoefficientMetric(
                                name='DiceCoeficienteMetric_1', target_class=1),
                            DiceCoefficientMetric(
                                name='DiceCoeficienteMetric_2', target_class=2),
                            ]
                }

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
