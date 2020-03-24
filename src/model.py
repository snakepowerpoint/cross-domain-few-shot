import tensorflow as tf
import numpy as np

from src.model_utils import residual_simple_block, convolution_layer, max_pool, batchnorm_conv, fc_layer


class PrototypeNet(object):
    def __init__(self, n_way, n_shot, n_query, backbone, learning_rate, is_training):
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.backbone = backbone
        self.lr = learning_rate
        self.is_training = is_training
    
    def cnn4_encoder(self, inputs, is_training=True, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            net = convolution_layer(inputs, [3, 3, 64], [1, 1, 1, 1], is_bn=True,
                                    activat_fn=tf.nn.relu, name='conv_block1', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], name='max_1')
            net = convolution_layer(net, [3, 3, 64], [1, 1, 1, 1], is_bn=True,
                                    activat_fn=tf.nn.relu, name='conv_block2', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], name='max_2')
            net = convolution_layer(net, [3, 3, 64], [1, 1, 1, 1], is_bn=True,
                                    activat_fn=tf.nn.relu, name='conv_block3', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], name='max_3')
            net = convolution_layer(net, [3, 3, 64], [1, 1, 1, 1], is_bn=True,
                                    activat_fn=tf.nn.relu, name='conv_block4', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], name='max_4')
            net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))], name="flatout")
        return net

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
    
    def compute_acc(self, prediction, one_hot_labels):
        labels = tf.argmax(one_hot_labels, axis=1)
        acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(prediction, axis=-1), labels)))
        return acc
    
    def train(self, support, query):
        support_a_feature = self.cnn4_encoder(support, is_training=self.is_training)
        query_b_feature = self.cnn4_encoder(query, is_training=self.is_training, reuse=True)

        # get prototype
        prototype_a = self.get_prototype(support_a_feature, n_shot=self.n_shot)

        # metric function (few-shot classification)
        dists = self.compute_distance(prototype_a, query_b_feature)
        log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [self.n_way, self.n_query, -1])

        # classification loss and accuracy
        query_b_y = np.repeat(np.arange(self.n_way), repeats=self.n_query).astype(np.uint8) 
        query_b_y = np.reshape(query_b_y, [self.n_way, self.n_query])  # [5, 15]
        query_b_y_one_hot = tf.one_hot(query_b_y, depth=self.n_way)

        self.train_loss = -tf.reduce_mean(
            tf.reshape(tf.reduce_sum(tf.multiply(query_b_y_one_hot, log_p_y), axis=-1), [-1]))
        self.train_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), query_b_y)))

        # optimizer
        global_step = tf.Variable(0, trainable=False, name='global_step')
        rate = tf.train.exponential_decay(self.lr, global_step, 2000, 0.5, staircase=True)
        optimizer = tf.train.AdamOptimizer(rate)
        self.train_op = optimizer.minimize(self.train_loss, global_step=global_step)

        return self.train_op, self.train_loss, self.train_acc, global_step

    def test(self, support, query):
        support_a_feature = self.cnn4_encoder(support, is_training=self.is_training, reuse=True)
        query_b_feature = self.cnn4_encoder(query, is_training=self.is_training, reuse=True)

        # get prototype
        prototype_a = self.get_prototype(support_a_feature, n_shot=self.n_shot)

        # metric function (few-shot classification)
        dists = self.compute_distance(prototype_a, query_b_feature)
        log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [self.n_way, self.n_query, -1])

        # classification loss and accuracy
        query_b_y = np.repeat(np.arange(self.n_way), repeats=self.n_query).astype(np.uint8)
        query_b_y = np.reshape(query_b_y, [self.n_way, self.n_query])  # [5, 15]
        query_b_y_one_hot = tf.one_hot(query_b_y, depth=self.n_way)

        self.train_loss = -tf.reduce_mean(
            tf.reshape(tf.reduce_sum(tf.multiply(query_b_y_one_hot, log_p_y), axis=-1), [-1]))
        self.train_acc = tf.reduce_mean(tf.to_float(
            tf.equal(tf.argmax(log_p_y, axis=-1), query_b_y)))

        return self.train_loss, self.train_acc


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
            net = convolution_layer(inputs, [3, 3, 64], [1, 1, 1, 1], is_bn=True,
                                    activat_fn=tf.nn.relu, name='conv_block1', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], name='max_1')
            net = convolution_layer(net, [3, 3, 64], [1, 1, 1, 1], is_bn=True,
                                    activat_fn=tf.nn.relu, name='conv_block2', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], name='max_2')
            net = convolution_layer(net, [3, 3, 64], [1, 1, 1, 1], is_bn=True,
                                    activat_fn=tf.nn.relu, name='conv_block3', is_training=is_training)
            net = convolution_layer(net, [3, 3, 64], [1, 1, 1, 1], is_bn=True,
                                    activat_fn=tf.nn.relu, name='conv_block4', is_training=is_training)
        return net
    
    def resnet10_encoder(self, inputs, is_training=True, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            # conv 1
            net = convolution_layer(inputs, [7, 7, 64], [1, 2, 2, 1], bias=False, is_bn=True,
                                    activat_fn=tf.nn.relu, name='conv_1', is_training=is_training)
            # net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], name='max_1', padding='SAME')
            
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
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], name='max_1', padding='VALID')
            net = convolution_layer(net, [3, 3, 64], [1, 1, 1, 1], is_bn=True, activat_fn=tf.nn.relu,
                                    name='conv_block2', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], name='max_2', padding='VALID')
            
            net = fc_layer(net, 8, name='fc1', activat_fn=tf.nn.relu)
            net = fc_layer(net, 1, name='fc2', activat_fn=tf.nn.sigmoid)
        return net

    def mse(self, y_pred, y_true):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def ce_loss(self, y_pred, y_true):
        ce_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        return tf.reduce_mean(ce_loss)

    def train(self, support, query, regularized=True):
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
        self.train_loss = self.ce_loss(y_pred=relations, y_true=one_hot_labels)
        self.train_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations, axis=-1), labels)))

        if regularized:
            train_vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if 'weight' in v.name]) * 1e-8
            self.train_loss = tf.add(self.train_loss, l2_loss)

        # optimize
        global_step = tf.Variable(0, trainable=False, name='global_step')
        rate = self.lr
        # rate = tf.train.exponential_decay(self.lr, global_step, 5000, 0.5, staircase=True)
        optimizer = tf.train.AdamOptimizer(rate)
        self.train_op = optimizer.minimize(self.train_loss, global_step=global_step)
        
        return self.train_op, self.train_loss, self.train_acc, global_step
    
    # need to correct
    def test(self, support, query, regularized=True):
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

        if regularized:
            train_vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if 'weight' in v.name]) * 1e-8
            self.test_loss = tf.add(self.test_loss, l2_loss)
        
        return self.test_loss, self.test_acc
