from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from gcpds.DataSet.infrared_thermal_feet import ModifiedInfraredThermalFeet
from gcpds.DataSet.convRFFds.train import train
from utils import init_wandb, resizen
from load_models import get_model
import wandb
import os


def train_model(sweep_id, kwargs_data_augmentation, config=None):
    kernel_regularizer = None
    input_shape = 224, 224, 1
    out_channels = 1
    with wandb.init(config=config) as run:
        config = dict(run.config)
        model = config['model']
        name_dataset = config['dataset']

        if name_dataset == 'infrared_thermal_feet_80x60_NoAug16':
            data_augmentation = False
            kwargs_data_augmentation["repeat"] = 1

        dataset_class = ModifiedInfraredThermalFeet
        model = get_model(model, input_shape, out_channels, kernel_regularizer)
        train(model, dataset_class, run,
              data_augmentation=data_augmentation, **kwargs_data_augmentation)
    os.environ["WANDB_API_KEY"] = "12fe25998ca888a5d462ea747e3d711af79fc285"
    SWEEP_ID = sweep_id
    wandb.agent(SWEEP_ID, train_model, count=1000,
                project='MamitasLowResolution')


if __name__ == "__main__":
    resize = False
    if resize:
        resizen()
    sweep_id = init_wandb()
    kwargs_data_augmentation = dict(repeat=1,
                                    batch_size=2,
                                    shape=224,
                                    split=[0.4, 0.3]
                                    )
    """    
    kwargs_data_augmentation = dict(repeat=7,
                                    flip_left_right=True,
                                    flip_up_down=False,
                                    range_rotate=(-15, 15),
                                    translation_h_w=(0.10, 0.10),
                                    zoom_h_w=(0.15, 0.15),
                                    batch_size=8,
                                    shape=224,
                                    split=[0.6, 0.4]
                                    )
    """
    train(sweep_id, **kwargs_data_augmentation)
