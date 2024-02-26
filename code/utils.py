import gcpds.DataSet.convRFFds.data as data
import tensorflow as tf
import numpy as np
import wandb
import os


def resizen():
    def resizeAndlowresolution(shape=(256, 256)):
        def func(img, mask):
            return tf.image.resize(tf.image.resize(img, (80, 60)), shape, method='area'), tf.image.resize(mask, (224, 224))
        return func

    data.resize = resizeAndlowresolution


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
    return sweep_id


def get_episodenk(train_dataset, nway, kshot):
    support = np.zeros([nway, kshot, 224,  224, 1], dtype=np.float32)
    smasks = np.zeros([nway, kshot, 56, 56, 1], dtype=np.float32)
    query = np.zeros([nway,         224,  224, 1], dtype=np.float32)
    qmask = np.zeros([nway,         224,  224, 1], dtype=np.float32)
    for idx in range(nway):
        for idy1 in range(kshot):
            onedata = train_dataset.batch(1).shuffle(100)
            data = onedata.take(1)
            for img, mask, id_img in data:
                img = img[0, ...]
                img = img[0, ...].numpy()
                mask = mask[0, ...]
                mask = mask[0, ...]
                mask = tf.image.resize(mask, (56, 56)).numpy()
                simg = tf.convert_to_tensor(
                    img.reshape(1, 1, 224, 224, 1), dtype=float)
                smask = tf.convert_to_tensor(mask.reshape(
                    1, 1, 56, 56, 1), dtype=float)
                support[idx, idy1] = simg
                smasks[idx, idy1] = smask
        for idy2 in range(1):
            onedata = train_dataset.batch(1).shuffle(100)
            data = onedata.take(1)
            for img, mask, id_img in data:
                img = img[0, ...]
                img = img[0, ...].numpy()
                mask = mask[0, ...]
                mask = mask[0, ...].numpy()
                qimg = tf.convert_to_tensor(
                    img.reshape(1, 224, 224, 1), dtype=float)
                qmasks = tf.convert_to_tensor(
                    mask.reshape(1, 224, 224, 1), dtype=float)
                query[idx] = qimg
                qmask[idx] = qmasks
    return support, smasks, query, qmask
