import tensorflow as tf
import numpy as np

from src.model_utils import residual_simple_block, convolution_layer, max_pool, batchnorm_conv, fc_layer


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


class RelationNet(object):
    def __init__(self, n_way, n_shot, n_query, backbone, learning_rate, is_training):
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.backbone = backbone
        self.lr = learning_rate
        self.is_training = is_training

    def cnn4_encoder(self, inputs, is_training=True, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            net = convolution_layer(inputs, [3, 3, 64], [1, 1, 1, 1], is_bn=True, padding='VALID',
                                    activat_fn=tf.nn.relu, name='conv_block1', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1])
            net = convolution_layer(net, [3, 3, 64], [1, 1, 1, 1], is_bn=True, padding='VALID',
                                    activat_fn=tf.nn.relu, name='conv_block2', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1])
            net = convolution_layer(net, [3, 3, 64], [1, 1, 1, 1], is_bn=True, padding='VALID',
                                    activat_fn=tf.nn.relu, name='conv_block3', is_training=is_training)
            net = convolution_layer(net, [3, 3, 64], [1, 1, 1, 1], is_bn=True, padding='VALID',
                                    activat_fn=tf.nn.relu, name='conv_block4', is_training=is_training)
        return net
    
    def resnet10_encoder(self, inputs, is_training=True, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            # conv 1
            net = convolution_layer(inputs, [7, 7, 64], [1, 2, 2, 1], is_bn=True, padding='VALID',
                                    activat_fn=tf.nn.relu, name='conv_1', is_training=is_training)
            net = max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1])
            
            # conv 2
            net = residual_simple_block(net, 64, block=1, is_half=False, is_training=is_training)
            # conv 3
            net = residual_simple_block(net, 128, block=2, is_half=True, is_training=is_training)
            # conv 4
            net = residual_simple_block(net, 256, block=3, is_half=True, is_training=is_training)
            # conv 5
            net = residual_simple_block(net, 512, block=4, is_half=True, is_training=is_training)
        return net
    
    def relation_module(self, inputs, is_training=True, reuse=False):
        with tf.variable_scope('relation_mod', reuse=reuse):
            net = convolution_layer(inputs, [3, 3, 64], [1, 1, 1, 1], is_bn=True, activat_fn=tf.nn.relu,
                                    name='conv_block1', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1])
            net = convolution_layer(net, [3, 3, 64], [1, 1, 1, 1], is_bn=True, activat_fn=tf.nn.relu,
                                    name='conv_block2', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1])
            
            net = fc_layer(net, 8, name='fc1', activat_fn=tf.nn.relu)
            net = fc_layer(net, 1, name='fc2', activat_fn=tf.nn.sigmoid)
        return net

    def mse(self, y_pred, y_true):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def train(self, support, query):
        '''
        Args
            support: input placeholder with shape [n_way * n_shot, None, None, 3]
            query: input placeholder with shape [n_way * n_query, None, None, 3]
        Return
            optimizer
            loss
            accuracy
        '''
        # support
        if self.backbone == 'conv4':
            support_encode = self.cnn4_encoder(support, is_training=self.is_training) 
        else:
            support_encode = self.resnet10_encoder(support, is_training=self.is_training)
        h, w, c = support_encode.get_shape().as_list()[1:]
        support_encode = tf.reduce_sum(tf.reshape(support_encode, [self.n_way, self.n_shot, h, w, c]), axis=1)
        support_encode = tf.tile(tf.expand_dims(support_encode, axis=0), [self.n_query * self.n_way, 1, 1, 1, 1]) 
        
        # query
        if self.backbone == 'conv4':
            query_encode = self.cnn4_encoder(query, is_training=self.is_training, reuse=True)
        else:
            query_encode = self.resnet10_encoder(query, is_training=self.is_training, reuse=True)
        query_encode = tf.tile(tf.expand_dims(query_encode, axis=0), [self.n_way, 1, 1, 1, 1])
        query_encode = tf.transpose(query_encode, perm=[1, 0, 2, 3, 4])

        relation_pairs = tf.concat([support_encode, query_encode], -1)
        relation_pairs = tf.reshape(relation_pairs, shape=[-1, h, w, c*2])
        relations = self.relation_module(relation_pairs, is_training=self.is_training)  # [75*5, 1]
        relations = tf.reshape(relations, [-1, self.n_way])  # [75, 5]

        labels = np.repeat(np.arange(self.n_way), repeats=self.n_query).astype(np.uint8)  # [75, 1]
        one_hot_labels = tf.one_hot(labels, depth=self.n_way)  # [75, 5]
        
        # loss and accuracy
        self.train_loss = self.mse(y_pred=relations, y_true=one_hot_labels)
        self.train_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations, axis=-1), labels)))

        # optimize
        global_step = tf.Variable(0, trainable=False, name='global_step')
        rate = tf.train.exponential_decay(self.lr, global_step, 2000, 0.5, staircase=True)
        optimizer = tf.train.AdamOptimizer(rate)
        self.train_op = optimizer.minimize(self.train_loss, global_step=global_step)
        
        return self.train_op, self.train_loss, self.train_acc, global_step
    
    # need to correct
    def test(self, support, query):
        # support
        if self.backbone == 'conv4':
            support_encode = self.cnn4_encoder(support, is_training=self.is_training, reuse=True)
        else:
            support_encode = self.resnet10_encoder(support, is_training=self.is_training, reuse=True)
        h, w, c = support_encode.get_shape().as_list()[1:]
        support_encode = tf.reduce_sum(tf.reshape(support_encode, [self.n_way, self.n_shot, h, w, c]), axis=1)
        support_encode = tf.tile(tf.expand_dims(support_encode, axis=0), [self.n_query * self.n_way, 1, 1, 1, 1]) 
        
        # query
        if self.backbone == 'conv4':
            query_encode = self.cnn4_encoder(query, is_training=self.is_training, reuse=True)
        else:
            query_encode = self.resnet10_encoder(query, is_training=self.is_training, reuse=True)
        query_encode = tf.tile(tf.expand_dims(query_encode, axis=0), [self.n_way, 1, 1, 1, 1])
        query_encode = tf.transpose(query_encode, perm=[1, 0, 2, 3, 4])

        relation_pairs = tf.concat([support_encode, query_encode], -1)
        relation_pairs = tf.reshape(relation_pairs, shape=[-1, h, w, c*2])
        relations = self.relation_module(relation_pairs, is_training=self.is_training, reuse=True)  # [75*5, 1]
        relations = tf.reshape(relations, [-1, self.n_way])  # [75, 5]

        labels = np.repeat(np.arange(self.n_way), repeats=self.n_query).astype(np.uint8)  # [75, 1]
        one_hot_labels = tf.one_hot(labels, depth=self.n_way)  # [75, 5]

        # loss and accuracy
        self.test_loss = self.mse(y_pred=relations, y_true=one_hot_labels)
        self.test_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations, axis=-1), labels)))

        return self.test_loss, self.test_acc
