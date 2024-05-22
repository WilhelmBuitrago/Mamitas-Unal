"""
https://github.com/cralji/RFF-Nerve-UTP/blob/main/FCN_Nerve-UTP.ipynb
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import keras.layers as layers
from keras.models import Model
from keras.applications.vgg16 import VGG16
import numpy as np

from functools import partial

import tensorflow as tf
from keras import Model, layers, regularizers


DefaultConv2D = partial(layers.Conv2D,
                        kernel_size=3, activation='relu', padding="same")

DefaultPooling = partial(layers.MaxPool2D,
                         pool_size=2)

DefaultTranspConv = partial(layers.Conv2DTranspose,
                            kernel_size=3, strides=2,
                            padding='same',
                            use_bias=False, activation='relu')


def kernel_initializer(seed):
    return tf.keras.initializers.GlorotUniform(seed=seed)


def FSLmodel(encoder='FCN', input_size=(256, 256, 1), k_shot=1, learning_rate=1e-4, out_channels=1, out_ActFunction='sigmoid'):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Se encontraron GPUs. Usando GPU para el entrenamiento.")
        # Configurar una estrategia de distribución para usar una única GPU
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        print("No se encontraron GPUs. Usando CPU para el entrenamiento.")
        # Configurar una estrategia de distribución para usar la CPU
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    with strategy.scope():
        def fcn_encoder(input_size=(256, 256, 3)):
            input_ = layers.Input(shape=input_size)

            x = layers.BatchNormalization(name='Batch00')(input_)

            x = DefaultConv2D(
                32, kernel_initializer=kernel_initializer(34), name='Conv10')(x)
            x = DefaultConv2D(
                32, kernel_initializer=kernel_initializer(4), name='Conv11')(x)
            x = layers.BatchNormalization(name='Batch10')(x)
            x = DefaultPooling(name='Pool10')(x)  # 128x128 -> 64x64

            x = DefaultConv2D(
                32, kernel_initializer=kernel_initializer(56), name='Conv20')(x)
            x = DefaultConv2D(
                32, kernel_initializer=kernel_initializer(28), name='Conv21')(x)
            x = layers.BatchNormalization(name='Batch20')(x)
            x = DefaultPooling(name='Pool20')(x)  # 64x64 -> 32x32

            x = DefaultConv2D(
                64, kernel_initializer=kernel_initializer(332), name='Conv30')(x)
            x = DefaultConv2D(
                64, kernel_initializer=kernel_initializer(2), name='Conv31')(x)
            x = layers.BatchNormalization(name='Batch30')(x)
            x = level_1 = DefaultPooling(name='Pool30')(x)  # 32x32 -> 16x16

            x = DefaultConv2D(
                128, kernel_initializer=kernel_initializer(67), name='Conv40')(x)
            x = DefaultConv2D(
                128, kernel_initializer=kernel_initializer(89), name='Conv41')(x)
            x = layers.BatchNormalization(name='Batch40')(x)
            x = level_2 = DefaultPooling(name='Pool40')(x)  # 16x16 -> 8x8

            x = DefaultConv2D(
                256, kernel_initializer=kernel_initializer(7), name='Conv50')(x)
            x = DefaultConv2D(
                256, kernel_initializer=kernel_initializer(23), name='Conv51')(x)
            x = layers.BatchNormalization(name='Batch50')(x)
            x = DefaultPooling(name='Pool50')(x)  # 8x8 -> 4x4

            model = Model(input_, x, name='FCN_Model')
            return model

        def fcn_decoder(Bi_rep, level_1, level_2, input, out_channels=out_channels, out_ActFunction=out_ActFunction):
            S_input2 = input[0]
            S_mask2 = input[1]
            Q_input2 = input[2]
            # Decode to query segment

        def common_representation(x1, x2):
            x = layers.concatenate([x1, x2], axis=3)
            x = layers.Conv2D(128, 3, padding='same',
                              kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization(axis=3)(x)
            x = layers.Activation('relu')(x)
            return x

        encoder = fcn_encoder(input_size=input_size)

        ###################################### K-shot learning #####################################
        # K shot
        S_input2 = layers.Input(
            (k_shot, input_size[0], input_size[1], input_size[2]))
        Q_input2 = layers.Input(input_size)
        S_mask2 = layers.Input(
            (k_shot, int(input_size[0]/32), int(input_size[1]/32), 1))

        kshot_encoder = tf.keras.models.Sequential()
        kshot_encoder.add(layers.TimeDistributed(encoder, input_shape=(
            k_shot, input_size[0], input_size[1], input_size[2])))

        s_encoded = kshot_encoder(S_input2)
        q_encoded = encoder(Q_input2)
        s_encoded = layers.TimeDistributed(layers.Conv2D(
            128, (3, 3), activation='relu', padding='same'))(s_encoded)
        q_encoded = layers.Conv2D(
            128, (3, 3), activation='relu', padding='same')(q_encoded)

        repc = int(s_encoded.shape[4])
        m = tf.keras.backend.repeat_elements(S_mask2, repc, axis=4)
        x = layers.multiply([s_encoded, m])
        repx = int(x.shape[2])
        repy = int(x.shape[3])
        x = (tf.keras.backend.sum(x, axis=[
            1, 2, 3], keepdims=True) / (tf.keras.backend.sum(m, axis=[1, 2, 3], keepdims=True)))
        x = tf.keras.layers.Reshape(target_shape=(
            np.int32(x.shape[2]), np.int32(x.shape[3]), np.int32(x.shape[4])))(x)
        x = tf.keras.backend.repeat_elements(x, repx, axis=1)
        s_encoded = tf.keras.backend.repeat_elements(x, repy, axis=2)

        # Common Representation of Support and Query sample
        Bi_rep = common_representation(s_encoded, q_encoded)

        x = layers.Conv2D(128, 3, padding='same',
                          kernel_initializer='he_normal')(Bi_rep)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(128, 3, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(128, 3, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(128, 3, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(128, 3, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(128, 3, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, 3,  activation='relu',
                          padding='same', kernel_initializer='he_normal')(x)
        x = layers.Conv2D(2, 3,   activation='relu',
                          padding='same', kernel_initializer='he_normal')(x)
        final = layers.Conv2D(1, 1,   activation='sigmoid')(x)

        seg_model = Model(
            inputs=[S_input2, S_mask2, Q_input2], outputs=final)
        seg_model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        device = strategy.extended._device

        if 'CPU' in device:
            print("El modelo se encuentra alojado en la CPU")
        else:
            print("El modelo se encuentra alojado en la GPU")
        return seg_model


if __name__ == "__main__":
    model = FSLmodel(input_size=(224, 224, 1))
    model.summary()
