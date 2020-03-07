import tensorflow as tf
from scipy.io import loadmat
import numpy as np


class PrototypeNet(object):
    def __init__(self):
        pass
        
    def feature_extractor(self, inputs, is_training, reuse=False):
        with tf.variable_scope('extractor', reuse=reuse):
            x = self._convolution_layer(inputs, (3, 3, 64), (1, 1, 1, 1), 'conv_block_1', is_bn=True, is_training=is_training)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            
            x = self._convolution_layer(x, (3, 3, 64), (1, 1, 1, 1), 'conv_block_2', is_bn=True, is_training=is_training)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            
            x = self._convolution_layer(x, (3, 3, 64), (1, 1, 1, 1), 'conv_block_3', is_bn=True, is_training=is_training)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            
            x = self._convolution_layer(x, (3, 3, 64), (1, 1, 1, 1), 'conv_block_4', is_bn=True, is_training=is_training)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            x = tf.reshape(x, [-1, int(np.prod(x.get_shape()[1:]))], name="flatout")
        return x

    def _convolution_layer(self, inputs, kernel_shape, stride, name,
                           flatten=False,
                           padding='SAME',
                           initializer=tf.contrib.layers.xavier_initializer(),
                           activat_fn=tf.nn.relu,
                           reg=None,
                           is_bn=False,
                           is_training=True):

        pre_shape = inputs.get_shape()[-1]
        rkernel_shape = [kernel_shape[0], kernel_shape[1], pre_shape, kernel_shape[2]]

        with tf.variable_scope(name):
            weight = tf.get_variable(
                "weights", rkernel_shape, tf.float32, initializer=initializer, regularizer=reg)
            bias = tf.get_variable(
                "bias", kernel_shape[2], tf.float32, initializer=tf.zeros_initializer())

            net = tf.nn.conv2d(inputs, weight, stride, padding=padding)
            net = tf.add(net, bias)

            if is_bn:
                net = self._batchnorm_conv(net, is_training=is_training)

            if not activat_fn == None:
                net = activat_fn(net, name=name+"_out")

            if flatten == True:
                net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))], name=name+"_flatout")
        return net

    def _batchnorm_conv(self, input, is_training):
        with tf.variable_scope("batchnorm"):
            input = tf.identity(input)
            channels = input.get_shape()[3]

            beta = tf.get_variable(
                "beta", [channels], tf.float32, initializer=tf.zeros_initializer())
            gamma = tf.get_variable(
                "gamma", [channels], tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))

            pop_mean = tf.get_variable(
                "pop_mean", [channels], tf.float32, initializer=tf.zeros_initializer(), trainable=False)
            pop_variance = tf.get_variable(
                "pop_variance", [channels], tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02), trainable=False)

            epsilon = 1e-3
            def batchnorm_train():
                batch_mean, batch_variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)

                decay = 0.99
                train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
                train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))

                with tf.control_dependencies([train_mean, train_variance]):
                    return tf.nn.batch_normalization(input, batch_mean, batch_variance, beta, gamma, epsilon)

            def batchnorm_infer():
                return tf.nn.batch_normalization(input, pop_mean, pop_variance, beta, gamma, epsilon)
            
            batch_normalized_output = tf.cond(is_training, batchnorm_train, batchnorm_infer)
            return batch_normalized_output

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
