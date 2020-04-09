import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np


## CNN
def residual_simple_block(inputs, out_dim, block, is_half=False, is_training=True):
    with tf.variable_scope('block{}'.format(block)):
        if is_half:
            net = convolution_layer(inputs, [3, 3, out_dim], stride=[1, 2, 2, 1], is_bias=False,
                                    name='conv1', is_bn=True, activat_fn=tf.nn.relu, is_training=is_training)
            short_cut = convolution_layer(inputs, [1, 1, out_dim], stride=[1, 2, 2, 1], is_bias=False,
                                          name='short_cut', is_bn=True, activat_fn=None, is_training=is_training)
        else:
            net = convolution_layer(inputs, [3, 3, out_dim], stride=[1, 1, 1, 1], is_bias=False,
                                    name='conv1', is_bn=True, activat_fn=tf.nn.relu, is_training=is_training)
            short_cut = inputs
        net = convolution_layer(net, [3, 3, out_dim], stride=[1, 1, 1, 1], is_bias=False,
                                name='conv2', is_bn=True, activat_fn=None, is_training=is_training)
        net = tf.add(net, short_cut)
        net = tf.nn.relu(net, name="out")
    return net

# def residual_bottle_block(inputs):
#     return net

def convolution_layer(inputs,
                      kernel_shape,
                      stride,
                      name,
                      padding='SAME',
                      is_bias=True,
                      initializer=tf.contrib.layers.xavier_initializer(),
                      is_bn=False,
                      activat_fn=tf.nn.relu,
                      flatten=False,
                      reg=False,
                      is_training=True):

    pre_shape = inputs.get_shape()[-1]
    rkernel_shape = [kernel_shape[0], kernel_shape[1], pre_shape, kernel_shape[2]]

    with tf.variable_scope(name):
        weight = tf.get_variable(
            "weights", rkernel_shape, tf.float32, initializer=initializer, regularizer=reg)
        net = tf.nn.conv2d(inputs, weight, stride, padding=padding)

        if is_bias:
            bias = tf.get_variable(
                "bias", kernel_shape[2], tf.float32, initializer=tf.zeros_initializer())
            net = tf.add(net, bias)

        if is_bn:
            net = batchnorm_conv(net, name=name, is_training=is_training)

        if activat_fn is not None:
            net = activat_fn(net, name=name+"_out")

        if flatten == True:
            net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))], name=name+"_flatout")
    return net

def batchnorm_conv(inputs, name, is_training=tf.cast(True, tf.bool)):
    with tf.variable_scope(name+"_bn", reuse=tf.AUTO_REUSE):
        inputs = tf.identity(inputs)
        channels = inputs.get_shape()[3]

        beta = tf.get_variable(
            "beta", [channels], tf.float32, initializer=tf.zeros_initializer())
        gamma = tf.get_variable(
            "gamma", [channels], tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))

        pop_mean = tf.get_variable(
            "pop_mean", [channels], tf.float32, initializer=tf.zeros_initializer(), trainable=False)
        pop_variance = tf.get_variable(
            "pop_variance", [channels], tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02), trainable=False)

        epsilon = 1e-5
        def batchnorm_train():
            batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=False)

            decay = 0.9 # wei
            train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
            train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))

            with tf.control_dependencies([train_mean, train_variance]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_variance, beta, gamma, epsilon)

        def batchnorm_infer():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_variance, beta, gamma, epsilon)
        
        batch_normalized_output = tf.cond(is_training, batchnorm_train, batchnorm_infer)
        return batch_normalized_output


def max_pool(inputs, kernel_size, strides, padding='VALID', name=None):
    '''
    Args
        kernel_size: e.g. [1, 2, 2, 1]
        strides: e.g. [1, 2, 2, 1]
    '''
    net = tf.nn.max_pool(inputs, ksize=kernel_size, strides=strides, padding=padding, name=name)
    return net

# need to correct
def fc_layer(inputs, 
             output_shape,
             name,
             initializer=tf.contrib.layers.xavier_initializer(),
             activat_fn=tf.nn.relu,
             reg=None):
    '''
    Args

    '''
    with tf.variable_scope(name):
        shape = inputs.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        net = tf.reshape(inputs, [-1, dim])

        weight = tf.get_variable(
            "weights", [dim, output_shape], tf.float32, initializer=initializer, regularizer=reg)
        bias = tf.get_variable(
            "bias", [output_shape], tf.float32, initializer=tf.zeros_initializer())

        # Note that the '+' operation automatically broadcasts the bias.
        net = tf.nn.bias_add(tf.matmul(net, weight), bias)
        if activat_fn is not None:
            net = activat_fn(net, name=name+"_out")
        return net


def fc_layer_test(inputs, 
                  output_shape,
                  name,
                  is_bias=True,
                  initializer=tf.contrib.layers.xavier_initializer(),
                  activat_fn=tf.nn.relu,
                  reg=None):
    '''
    Args

    '''
    with tf.variable_scope(name):
        shape = inputs.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        net = tf.reshape(inputs, [-1, dim])

        weight = tf.get_variable(
            "weights", [dim, output_shape], tf.float32, initializer=initializer, regularizer=reg)
        net = tf.matmul(net, weight)
        
        if is_bias:
            bias = tf.get_variable(
                "bias", [output_shape], tf.float32, initializer=tf.zeros_initializer())
        # Note that the '+' operation automatically broadcasts the bias.
        net = tf.nn.bias_add(net, bias)

        if activat_fn is not None:
            net = activat_fn(net, name=name+"_out")
        return net


## RNN
# def lstm(inputs, name, reuse):
#     with tf.variable_scope(name, reuse=reuse):
#         fw_lstm_cells_encoder = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
#                                     for i in range(len(self.layer_sizes))]
#         bw_lstm_cells_encoder = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
#                                     for i in range(len(self.layer_sizes))]

#         outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_rnn(
#             fw_lstm_cells_encoder,
#             bw_lstm_cells_encoder,
#             inputs,
#             dtype=tf.float32
#         )

#     self.reuse = True
#     self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
#     return outputs

def convolution_layer_meta(inputs, weight, bias, strides, name, is_training=True, is_bn=False, padding='SAME', activat_fn=tf.nn.relu):

    x = tf.nn.conv2d(inputs, weight, strides, padding, name=name + '_conv2d') + bias
    
    if is_bn == True:
        x = batchnorm_conv(x, name=name, is_training=is_training)

    if activat_fn is not None:
        x = activat_fn(x, name=name + "_out")
    
    return x

def fc_layer_meta(inputs, weight, bias, name, activat_fn=tf.nn.relu):

    shape = inputs.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    x = tf.reshape(inputs, [-1, dim])

    x = tf.nn.bias_add(tf.matmul(x, weight), bias)

    if activat_fn is not None:
        x = activat_fn(x, name=name + "_out")
    
    return x    