"""
https://github.com/cralji/RFF-Nerve-UTP/blob/main/FCN_Nerve-UTP.ipynb
"""

from functools import partial

import tensorflow as tf
from keras import Model, layers, regularizers
import numpy as np


class FCNModel():
    def __inti__(self):
        super.__init__(self, FCNModel)

    def kernel_initializer(self, seed):
        return tf.keras.initializers.GlorotUniform(seed=seed)

    def common_representation(self, x1, x2):
        x = layers.concatenate([x1, x2], axis=3)
        x = layers.Conv2D(128, 3, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)
        return x

    def fcn_encoder(self, initial_layer):
        # Encoder
        self.intial = initial_layer
        x = layers.BatchNormalization(name='Batch00')(initial_layer)

        x = layers.Conv2D(
            32, kernel_size=3, activation='relu', padding="same", kernel_initializer=self.kernel_initializer(34), name='Conv10')(x)
        x = layers.Conv2D(
            32, kernel_size=3, activation='relu', padding="same", kernel_initializer=self.kernel_initializer(4), name='Conv11')(x)
        x = layers.BatchNormalization(name='Batch10')(x)
        x = layers.MaxPool2D(pool_size=2, name='Pool10')(x)  # 128x128 -> 64x64

        x = layers.Conv2D(
            32, kernel_size=3, activation='relu', padding="same", kernel_initializer=self.kernel_initializer(56), name='Conv20')(x)
        x = layers.Conv2D(
            32, kernel_size=3, activation='relu', padding="same", kernel_initializer=self.kernel_initializer(28), name='Conv21')(x)
        x = layers.BatchNormalization(name='Batch20')(x)
        x = layers.MaxPool2D(pool_size=2, name='Pool20')(x)  # 64x64 -> 32x32

        x = layers.Conv2D(
            64, kernel_size=3, activation='relu', padding="same", kernel_initializer=self.kernel_initializer(332), name='Conv30')(x)
        x = layers.Conv2D(
            64, kernel_size=3, activation='relu', padding="same", kernel_initializer=self.kernel_initializer(2), name='Conv31')(x)
        x = layers.BatchNormalization(name='Batch30')(x)
        x = level_1 = layers.MaxPool2D(
            pool_size=2, name='Pool30')(x)  # 32x32 -> 16x16

        x = layers.Conv2D(
            128, kernel_size=3, activation='relu', padding="same", kernel_initializer=self.kernel_initializer(67), name='Conv40')(x)
        x = layers.Conv2D(
            128, kernel_size=3, activation='relu', padding="same", kernel_initializer=self.kernel_initializer(89), name='Conv41')(x)
        x = layers.BatchNormalization(name='Batch40')(x)
        x = level_2 = layers.MaxPool2D(
            pool_size=2, name='Pool40')(x)  # 16x16 -> 8x8

        x = layers.Conv2D(
            256, kernel_size=3, activation='relu', padding="same", kernel_initializer=self.kernel_initializer(7), name='Conv50')(x)
        x = layers.Conv2D(
            256, kernel_size=3, activation='relu', padding="same", kernel_initializer=self.kernel_initializer(23), name='Conv51')(x)
        x = layers.BatchNormalization(name='Batch50')(x)
        x = layers.MaxPool2D(pool_size=2, name='Pool50')(x)  # 8x8 -> 4x4

        return Model(inputs=initial_layer, outputs=[level_1, level_2, x], name="FCN_Model")

    def fcn_decoder(self, encoder_output, name='FCN', out_channels=1, out_ActFunction='sigmoid'):
        # Decoder
        level_1, level_2, x = encoder_output
        x = level_3 = layers.Conv2DTranspose(out_channels, kernel_size=4, strides=2,
                                             padding='same',
                                             activation='relu',
                                             use_bias=False,
                                             kernel_initializer=self.kernel_initializer(
                                                 98),
                                             name='Trans60')(x)
        x = layers.Conv2D(out_channels, kernel_size=1,
                          activation=None, kernel_initializer=self.kernel_initializer(75),
                          name='Conv60')(level_2)

        x = layers.Add(name='Add10')([x, level_3])

        x = level_4 = layers.Conv2DTranspose(out_channels, kernel_size=4, strides=2,
                                             padding='same',
                                             activation='relu',
                                             use_bias=False, kernel_initializer=self.kernel_initializer(87),
                                             name='Trans70')(x)

        x = layers.Conv2D(out_channels, kernel_size=1, activation=None,
                          kernel_initializer=self.kernel_initializer(54),
                          name='Conv70')(level_1)

        x = layers.Add(name='Add20')([x, level_4])

        x = layers.Conv2DTranspose(out_channels, kernel_size=16, strides=8,
                                   activation=out_ActFunction, use_bias=True,
                                   kernel_initializer=self.kernel_initializer(
                                       32),
                                   name='Trans80')(x)
        return x

    def CreateModel(self, input_size=(256, 256, 1), k_shot=1):
        initial_layer = layers.Input(shape=input_size)
        encoder = self.fcn_encoder(initial_layer=initial_layer)
        S_input2 = layers.Input(
            (k_shot, input_size[0], input_size[1], input_size[2]))
        Q_input2 = layers.Input(input_size)
        S_mask2 = layers.Input(
            (k_shot, int(input_size[0]/32), int(input_size[1]/32), 1))

        kshot_encoder = tf.keras.models.Sequential()
        kshot_encoder.add(layers.TimeDistributed(encoder, input_shape=(
            k_shot, input_size[0], input_size[1], input_size[2])))  # Arreglar salida modelo encoder
        s_encoded = kshot_encoder(S_input2)
        level_1, level_2, q_encoded = encoder(Q_input2)

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
        Bi_rep = self.common_representation(s_encoded, q_encoded)

        final = self.fcn_decoder(encoder_output=[level_1, level_2, Bi_rep])

        return Model(inputs=Q_input2, outputs=final)


if __name__ == "__main__":
    model = FCNModel().CreateModel()
    model.summary()
