from tensorflow.keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


class InceptionV1Module(layers.Layer):

    def __init__(self,
                 conv1x1_filters=64, conv3x3_reduce_filters=96, conv3x3_filters=128, conv5x5_reduce_filters=16, conv5x5_filters=32, pooling_conv_filters=32,
                 **kwargs):
        super(InceptionV1Module, self).__init__(**kwargs)

        self.conv1x1_filters = conv1x1_filters
        self.conv3x3_reduce_filters = conv3x3_reduce_filters
        self.conv3x3_filters = conv3x3_filters
        self.conv5x5_reduce_filters = conv5x5_reduce_filters
        self.conv5x5_filters = conv5x5_filters
        self.pooling_conv_filters = pooling_conv_filters

        self.conv_1x1 = layers.Conv2D(
            conv1x1_filters, (1, 1), padding='same', activation='relu')

        self.conv_3x3_1 = layers.Conv2D(
            conv3x3_reduce_filters, (1, 1), padding='same', activation='relu')
        self.conv_3x3_2 = layers.Conv2D(
            conv3x3_filters, (3, 3), padding='same', activation='relu')

        self.conv_5x5_1 = layers.Conv2D(
            conv5x5_reduce_filters, (1, 1), padding='same', activation='relu')
        self.conv_5x5_2 = layers.Conv2D(
            conv5x5_filters, (5, 5), padding='same', activation='relu')

        self.pooling_1 = layers.MaxPooling2D(
            (3, 3), strides=(1, 1), padding='same')
        self.pooling_2 = layers.Conv2D(
            pooling_conv_filters, (1, 1), padding='same', activation='relu')

    def call(self, inputs):

        res_conv_1x1 = self.conv_1x1(inputs)

        res_conv_3x3 = self.conv_3x3_1(inputs)
        res_conv_3x3 = self.conv_3x3_2(res_conv_3x3)

        res_conv_5x5 = self.conv_5x5_1(inputs)
        res_conv_5x5 = self.conv_5x5_2(res_conv_5x5)

        res_pooling = self.pooling_1(inputs)
        res_pooling = self.pooling_2(res_pooling)

        result = layers.concatenate(
            [res_conv_1x1, res_conv_3x3, res_conv_5x5, res_pooling])

        return result

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1x1_filters': self.conv1x1_filters,
            'conv3x3_reduce_filters': self.conv3x3_reduce_filters,
            'conv3x3_filters': self.conv3x3_filters,
            'conv5x5_reduce_filters': self.conv5x5_reduce_filters,
            'conv5x5_filters': self.conv5x5_filters,
            'pooling_conv_filters': self.pooling_conv_filters
        })
        return config

    def from_config(self, cls, config):
        # raise ValueError("From config")
        return cls(**config)


class InceptionV1ModuleBN(layers.Layer):

    def __init__(self,
                 conv1x1_filters=64, conv3x3_reduce_filters=96, conv3x3_filters=128, conv5x5_reduce_filters=16, conv5x5_filters=32, pooling_conv_filters=32,
                 **kwargs):
        super(InceptionV1ModuleBN, self).__init__(**kwargs)

        self.conv1x1_filters = conv1x1_filters
        self.conv3x3_reduce_filters = conv3x3_reduce_filters
        self.conv3x3_filters = conv3x3_filters
        self.conv5x5_reduce_filters = conv5x5_reduce_filters
        self.conv5x5_filters = conv5x5_filters
        self.pooling_conv_filters = pooling_conv_filters

        self.conv_1x1 = layers.Conv2D(
            conv1x1_filters, (1, 1), padding='same', activation='relu')
        self.bn_1x1 = layers.BatchNormalization()

        self.conv_3x3_1 = layers.Conv2D(
            conv3x3_reduce_filters, (1, 1), padding='same', activation='relu')
        self.conv_3x3_2 = layers.Conv2D(
            conv3x3_filters, (3, 3), padding='same', activation='relu')
        self.bn_3x3_1 = layers.BatchNormalization()
        self.bn_3x3_2 = layers.BatchNormalization()

        self.conv_5x5_1 = layers.Conv2D(
            conv5x5_reduce_filters, (1, 1), padding='same', activation='relu')
        self.conv_5x5_2 = layers.Conv2D(
            conv5x5_filters, (5, 5), padding='same', activation='relu')

        self.bn_5x5_1 = layers.BatchNormalization()
        self.bn_5x5_2 = layers.BatchNormalization()

        self.pooling_1 = layers.MaxPooling2D(
            (3, 3), strides=(1, 1), padding='same')
        self.pooling_2 = layers.Conv2D(
            pooling_conv_filters, (1, 1), padding='same', activation='relu')

    def call(self, inputs):

        res_conv_1x1 = self.conv_1x1(inputs)
        res_conv_1x1 = self.bn_1x1(res_conv_1x1)

        res_conv_3x3 = self.conv_3x3_1(inputs)
        res_conv_3x3 = self.bn_3x3_1(res_conv_3x3)
        res_conv_3x3 = self.conv_3x3_2(res_conv_3x3)
        res_conv_3x3 = self.bn_3x3_2(res_conv_3x3)

        res_conv_5x5 = self.conv_5x5_1(inputs)
        res_conv_5x5 = self.bn_5x5_1(res_conv_5x5)
        res_conv_5x5 = self.conv_5x5_2(res_conv_5x5)
        res_conv_5x5 = self.bn_5x5_2(res_conv_5x5)

        res_pooling = self.pooling_1(inputs)
        res_pooling = self.pooling_2(res_pooling)

        result = layers.concatenate(
            [res_conv_1x1, res_conv_3x3, res_conv_5x5, res_pooling])

        return result

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1x1_filters': self.conv1x1_filters,
            'conv3x3_reduce_filters': self.conv3x3_reduce_filters,
            'conv3x3_filters': self.conv3x3_filters,
            'conv5x5_reduce_filters': self.conv5x5_reduce_filters,
            'conv5x5_filters': self.conv5x5_filters,
            'pooling_conv_filters': self.pooling_conv_filters
        })
        return config

    def from_config(self, cls, config):
        # raise ValueError("From config")
        return cls(**config)
