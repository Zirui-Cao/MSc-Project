from typing import Optional, Sequence, Tuple, Union
import numpy as np


import tensorflow as tf
import tensorflow.keras.layers as nn

class resnet(nn.Layer):
    def __init__(self, feature_size, kernel_size):
        super(resnet, self).__init__()

        self.conv1 = nn.Conv2D(feature_size, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')
        self.bn1 = nn.BatchNormalization()
        self.relu = nn.Activation('relu')
        self.conv2 = nn.Conv2D(feature_size, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')
        self.bn2 = nn.BatchNormalization()

        # if stride != 1:
        #     self.downsample = nn.Conv2D(feature_size, 1, padding='same', kernel_initializer='he_normal')
        # else:
        #     self.downsample = lambda x: x

        self.downsample = nn.Conv2D(feature_size, 1, padding='same', kernel_initializer='he_normal')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.bn2(x)

        down = self.downsample(inputs)
        # print(out.shape)
        # print(down.shape)
        output = nn.add([out, down])
        output = tf.nn.relu(output)
        return output


class UnetrBasicBlock(tf.keras.Model):
    def __init__(self, feature_size, kernel_size):
        super(UnetrBasicBlock, self).__init__()

        self.res = resnet(feature_size, kernel_size)

    def call(self, inputs, training=True):
        return self.res(inputs)

class UnetrPrUpBlock(tf.keras.Model):
    def __init__(self, feature_size, kernel_size, upsample_kernel_size, num_layer):
        super(UnetrPrUpBlock, self).__init__()

        self.num_layer = num_layer
        # self.stem = nn.Conv2D(feature_size, kernel_size=upsample_kernel_size, padding='same', kernel_initializer='he_normal')(nn.UpSampling2D(size=(upsample_kernel_size,upsample_kernel_size)))
        self.stem = nn.Conv2DTranspose(feature_size, kernel_size=upsample_kernel_size, padding='same', kernel_initializer='he_normal')
        self.blocks = tf.keras.Sequential([
            # nn.Conv2D(feature_size, kernel_size=upsample_kernel_size, padding='same', kernel_initializer='he_normal')(nn.UpSampling2D(size=(upsample_kernel_size,upsample_kernel_size))),
            nn.UpSampling2D(size=(2,2), data_format=None),
            nn.Conv2D(feature_size, kernel_size=upsample_kernel_size, padding='same', kernel_initializer='he_normal'),
            # nn.Conv2DTranspose(feature_size, kernel_size=upsample_kernel_size, padding='same',kernel_initializer='he_normal'),
            resnet(feature_size, kernel_size)
        ])

    def call(self, inputs, training=True):
        x = self.stem(inputs)
        for i in range(self.num_layer):
            print(i)
            x = self.blocks(x)
        return x

class UnetrUpBlock(tf.keras.Model):
    def __init__(self, feature_size, kernel_size, upsample_kernel_size):
        super(UnetrUpBlock, self).__init__()

        # self.stem = nn.Conv2D(feature_size, kernel_size=upsample_kernel_size, padding='same', kernel_initializer='he_normal')(nn.UpSampling2D(size=(upsample_kernel_size,upsample_kernel_size)))
        self.stem = tf.keras.Sequential([
            nn.UpSampling2D(size=(2,2), data_format=None),
            nn.Conv2D(feature_size, kernel_size=upsample_kernel_size, padding='same',kernel_initializer='he_normal'),
            # nn.Conv2DTranspose(feature_size, kernel_size=upsample_kernel_size, padding='same',kernel_initializer='he_normal')
            ])
        # self.blocks = resnet(feature_size + feature_size, kernel_size)
        self.blocks = resnet(feature_size, kernel_size)

    def call(self, inputs, skip, training=True):
        x = self.stem(inputs)
        x = nn.concatenate([x, skip], axis=3)
        x = self.blocks(x)
        return x





