import tensorflow as tf
from scipy.io import loadmat
import numpy as np


class PrototypeNet(object):
    def __init__(self):
        pass

    def feature_extractor(self, inputs, is_training, reuse=False):
        with tf.variable_scope('extractor', reuse=reuse):
            x = self.conv_block(inputs, 64, is_training, 'conv_block_1')
            x = self.conv_block(x, 64, is_training, 'conv_block_2')
            x = self.conv_block(x, 64, is_training, 'conv_block_3')
            x = self.conv_block(x, 64, is_training, 'conv_block_4')
            x = tf.contrib.layers.flatten(x)
            return x

    def conv_block(self, inputs, out_channels, is_training, name):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(inputs, out_channels, kernel_size=(3, 3), padding='same')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu(x)
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
            return x

    def get_prototype(self, feature, n_shot):
        '''
        Args
            feature: [n_way * n_shot, 256] (e.g. [5*5, 256])
        
        Return
            Prototype: [n_way, 256] (e.g. [5, 256])
        '''
        feature = tf.reshape(feature, shape=[-1, n_shot, tf.shape(feature)[-1]])
        prototype = tf.reduce_mean(feature, axis=1)
        return prototype

    def compute_distance(self, prototype, query_feature):
        '''
        Args
            prototype: [n_way, 256] (e.g. [5, 256])
            query_feature: [n_way * n_query, 256] (e.g. [5*15, 256])

        Return
            dist: [n_way * n_query, n_way] (e.g. [5*15, 5])
        '''

        n_way, dim_feature = tf.shape(prototype)[0], tf.shape(prototype)[1]
        M = tf.shape(query_feature)[0]
        prototype = tf.tile(tf.expand_dims(prototype, axis=0), (M, 1, 1))
        query_feature = tf.tile(tf.expand_dims(query_feature, axis=1), (1, n_way, 1))
        dist = tf.reduce_mean(tf.square(query_feature - prototype), axis=-1)
        return dist
