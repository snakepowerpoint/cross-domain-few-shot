import tensorflow as tf
import numpy as np

from src.model_utils import residual_simple_block, convolution_layer, max_pool, batchnorm_conv, fc_layer, convolution_layer_meta, fc_layer_meta
from src.model_utils import convolution_layer_meta, fc_layer_meta

class PrototypeNet(object):
    def __init__(self, n_way, n_shot, n_query, backbone, learning_rate, is_training):
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.backbone = backbone
        self.gamma = learning_rate
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
        rate = tf.train.exponential_decay(self.gamma, global_step, 2000, 0.5, staircase=True)
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
    def __init__(self,
                 n_way,
                 n_shot,
                 n_query,
                 n_query_test=15,
                 alpha=1e-3,
                 beta=1.0,
                 gamma=5e-3,
                 decay=0.96, 
                 backbone='resnet',
                 is_training=True):
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_query_test = n_query_test
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.decay = decay
        self.backbone = backbone
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
            net = convolution_layer(inputs, [7, 7, 64], [1, 2, 2, 1], is_bias=False, is_bn=True,
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
    
    def resnet10_encoder_meta(self, inputs, weights, is_training=True):

        # conv 1
        x =         convolution_layer_meta(inputs, weights['conv1'], weights['b1'], [1, 2, 2, 1], name='conv1', is_training=is_training, is_bn=True, padding="SAME")
        x =         max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], name='max_1', padding='SAME')

        # conv 2
        sc_2 = x
        x =         convolution_layer_meta(x, weights['conv2_1'], weights['b2_1'], [1, 1, 1, 1], name='conv2_1', is_training=is_training, is_bn=True, padding="SAME")
        x =         convolution_layer_meta(x, weights['conv2_2'], weights['b2_2'], [1, 1, 1, 1], name='conv2_2', is_training=is_training, is_bn=True, padding="SAME", activat_fn=None)
        x =         tf.add(x, sc_2)
        x =         tf.nn.relu(x, name="conv2_2"+"_out")

        # conv 3
        sc_3 =      convolution_layer_meta(x, weights['conv3_sc'], weights['b3_sc'], [1, 2, 2, 1], name='conv3_sc', is_training=is_training, is_bn=True, padding="SAME", activat_fn=None)
        x =         convolution_layer_meta(x, weights['conv3_1'], weights['b3_1'], [1, 2, 2, 1], name='conv3_1', is_training=is_training, is_bn=True, padding="SAME")
        x =         convolution_layer_meta(x, weights['conv3_2'], weights['b3_2'], [1, 1, 1, 1], name='conv3_2', is_training=is_training, is_bn=True, padding="SAME", activat_fn=None)
        x =         tf.add(x, sc_3)
        x =         tf.nn.relu(x, name="conv3_2"+"_out")

        # conv 4
        sc_4 =      convolution_layer_meta(x, weights['conv4_sc'], weights['b4_sc'], [1, 2, 2, 1], name='conv4_sc', is_training=is_training, is_bn=True, padding="SAME", activat_fn=None)
        x =         convolution_layer_meta(x, weights['conv4_1'], weights['b4_1'], [1, 2, 2, 1], name='conv4_1', is_training=is_training, is_bn=True, padding="SAME")
        x =         convolution_layer_meta(x, weights['conv4_2'], weights['b4_2'], [1, 1, 1, 1], name='conv4_2', is_training=is_training, is_bn=True, padding="SAME", activat_fn=None)
        x =         tf.add(x, sc_4)
        x =         tf.nn.relu(x, name="conv4_2"+"_out")

        # conv 5
        sc_5 =      convolution_layer_meta(x, weights['conv5_sc'], weights['b5_sc'], [1, 2, 2, 1], name='conv5_sc', is_training=is_training, is_bn=True, padding="SAME", activat_fn=None)
        x =         convolution_layer_meta(x, weights['conv5_1'], weights['b5_1'], [1, 2, 2, 1], name='conv5_1', is_training=is_training, is_bn=True, padding="SAME")
        x =         convolution_layer_meta(x, weights['conv5_2'], weights['b5_2'], [1, 1, 1, 1], name='conv5_2', is_training=is_training, is_bn=True, padding="SAME", activat_fn=None)
        x =         tf.add(x, sc_5)
        x =         tf.nn.relu(x, name="conv5_2"+"_out")

        return x

    def resnet10_encoder_weights(self):        
        weights = {}
        
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=dtype)

        with tf.variable_scope("res10_weights", reuse=tf.AUTO_REUSE):
            # conv1
            weights['conv1']    = tf.get_variable('conv1w', [7, 7, 3, 64],  initializer=conv_initializer, dtype=dtype)
            weights['b1']       = tf.get_variable('conv1b', initializer=tf.zeros([64]))
            #weights['b1']       = tf.zeros([64])

            # conv2 - residual_simple_block 
            weights['conv2_1']    = tf.get_variable('conv2w_1', [3, 3, 64, 64],  initializer=conv_initializer, dtype=dtype)
            weights['b2_1']       = tf.get_variable('conv2b_1', initializer=tf.zeros([64]))
            #weights['b2_1']       = tf.zeros([64])

            weights['conv2_2']    = tf.get_variable('conv2w_2', [3, 3, 64, 64],  initializer=conv_initializer, dtype=dtype)
            weights['b2_2']       = tf.get_variable('conv2b_2', initializer=tf.zeros([64]))             
            #weights['b2_2']       = tf.zeros([64])

            # conv3 - residual_simple_block 
            weights['conv3_1']    = tf.get_variable('conv3w_1', [3, 3, 64, 128],  initializer=conv_initializer, dtype=dtype)
            weights['b3_1']       = tf.get_variable('conv3b_1', initializer=tf.zeros([128]))
            #weights['b3_1']       = tf.zeros([128])

            weights['conv3_2']    = tf.get_variable('conv3w_2', [3, 3, 128, 128],  initializer=conv_initializer, dtype=dtype)
            weights['b3_2']       = tf.get_variable('conv3b_2', initializer=tf.zeros([128]))
            #weights['b3_2']       = tf.zeros([128])

            weights['conv3_sc']    = tf.get_variable('conv3w_sc', [1, 1, 64, 128],  initializer=conv_initializer, dtype=dtype)
            weights['b3_sc']       = tf.get_variable('conv3b_sc', initializer=tf.zeros([128]))
            #weights['b3_sc']       = tf.zeros([128])

            # conv4 - residual_simple_block 
            weights['conv4_1']    = tf.get_variable('conv4w_1', [3, 3, 128, 256],  initializer=conv_initializer, dtype=dtype)
            weights['b4_1']       = tf.get_variable('conv4b_1', initializer=tf.zeros([256]))
            #weights['b4_1']       = tf.zeros([256])

            weights['conv4_2']    = tf.get_variable('conv4w_2', [3, 3, 256, 256],  initializer=conv_initializer, dtype=dtype)
            weights['b4_2']       = tf.get_variable('conv4b_2', initializer=tf.zeros([256]))
            #weights['b4_2']       = tf.zeros([256])

            weights['conv4_sc']    = tf.get_variable('conv4w_sc', [1, 1, 128, 256],  initializer=conv_initializer, dtype=dtype)
            weights['b4_sc']       = tf.get_variable('conv4b_sc', initializer=tf.zeros([256]))
            #weights['b4_sc']       = tf.zeros([256])

            # conv5 - residual_simple_block 
            weights['conv5_1']    = tf.get_variable('conv5w_1', [3, 3, 256, 512],  initializer=conv_initializer, dtype=dtype)
            weights['b5_1']       = tf.get_variable('conv5b_1', initializer=tf.zeros([512]))
            #weights['b5_1']       = tf.zeros([512])

            weights['conv5_2']    = tf.get_variable('conv5w_2', [3, 3, 512, 512],  initializer=conv_initializer, dtype=dtype)
            weights['b5_2']       = tf.get_variable('conv5b_2', initializer=tf.zeros([512]))
            #weights['b5_2']       = tf.zeros([512])

            weights['conv5_sc']    = tf.get_variable('conv5w_sc', [1, 1, 256, 512],  initializer=conv_initializer, dtype=dtype)
            weights['b5_sc']       = tf.get_variable('conv5b_sc', initializer=tf.zeros([512]))
            #weights['b5_sc']       = tf.zeros([512])

            return weights        

    def resnet10_classifier(self, inputs, label_dim=64):
        with tf.variable_scope('res10_cls', reuse=tf.AUTO_REUSE):
            net = fc_layer(inputs, label_dim, name='fc', activat_fn=None) # wei, enlarge the output dimension to 200
        return net

    def relation_module(self, inputs, loss_type='softmax', is_training=True, reuse=False):
        with tf.variable_scope('relation_mod', reuse=reuse):
            net = convolution_layer(inputs, [3, 3, 64], [1, 1, 1, 1], is_bn=True, activat_fn=tf.nn.relu,
                                    name='conv_block1', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], name='max_1', padding='VALID')
            net = convolution_layer(net, [3, 3, 64], [1, 1, 1, 1], is_bn=True, activat_fn=tf.nn.relu,
                                    name='conv_block2', is_training=is_training)
            net = max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], name='max_2', padding='VALID')
            
            net = fc_layer(net, 8, name='fc1', activat_fn=tf.nn.relu)
            if loss_type == 'mse':
                net = fc_layer(net, 1, name='fc2', activat_fn=tf.nn.sigmoid)
            elif loss_type == 'softmax':
                net = fc_layer(net, 1, name='fc2', activat_fn=None)
        return net

    def relation_module_meta(self, inputs, weights, loss_type='softmax', is_training=True):
        
        # conv 1
        x =         convolution_layer_meta(inputs, weights['conv1'], weights['b1'], [1, 1, 1, 1], name='M_conv1', is_training=is_training, is_bn=True, bn_momentum=1, padding="SAME")
        x =         max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], name='max_1', padding='VALID')        
        
        # conv 2
        x =         convolution_layer_meta(x, weights['conv2'], weights['b2'], [1, 1, 1, 1], name='M_conv2', is_training=is_training, is_bn=True, bn_momentum=1, padding="SAME")
        x =         max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], name='max_2', padding='VALID')  

        # fc 3
        x =         fc_layer_meta(x, weights['fc3'], weights['b3'], name='M_fc3', activat_fn=tf.nn.relu)

        # fc 4
        x =         fc_layer_meta(x, weights['fc4'], weights['b4'], name='M_fc4', activat_fn=None)

        return x

    def relation_module_weights(self):
        weights = {}
        
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=dtype)

        with tf.variable_scope('relation_mod_weights', reuse=tf.AUTO_REUSE):
            weights['conv1']    = tf.get_variable('conv1w', [3, 3, 1024, 64],  initializer=conv_initializer, dtype=dtype)
            weights['b1']       = tf.get_variable('conv1b', initializer=tf.zeros([64]))

            weights['conv2']    = tf.get_variable('conv2w', [3, 3, 64, 64],  initializer=conv_initializer, dtype=dtype)
            weights['b2']       = tf.get_variable('conv2b', initializer=tf.zeros([64]))
            
            weights['fc3']      = tf.get_variable('fc3w', [64, 8], initializer=fc_initializer, dtype=dtype)
            weights['b3']       = tf.get_variable('fc3b', initializer=tf.zeros([8]))            

            weights['fc4']      = tf.get_variable('fc4w', [8, 1], initializer=fc_initializer, dtype=dtype)
            weights['b4']       = tf.get_variable('fc4b', initializer=tf.zeros([1]))            

            return weights

    def mse(self, y_pred, y_true):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def ce_loss(self, y_pred, y_true):
        #ce_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        log_sm_vals = tf.nn.log_softmax(y_pred)
        ce_loss = tf.reduce_sum(-1*tf.multiply(y_true, log_sm_vals), axis=1)
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
        support_encode = tf.reduce_mean(tf.reshape(support_encode, [self.n_way, self.n_shot, h, w, c]), axis=1)
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
        rate = self.gamma
        # rate = tf.train.exponential_decay(self.gamma, global_step, 5000, 0.5, staircase=True)
        optimizer = tf.train.AdamOptimizer(rate)
        self.train_op = optimizer.minimize(self.train_loss, global_step=global_step)
        
        return self.train_op, self.train_loss, self.train_acc, global_step

    def build(self, n_way, n_shot, n_query, support_x, query_x, labels, regularized=False):
        
        ### build model
        # create network variables
        self.res10_weights = res10_weights = self.resnet10_encoder_weights()
        self.relation_weights = relation_weights = self.relation_module_weights()

        # create labels
        one_hot_labels = tf.one_hot(labels, depth=n_way)  # [75, 5]

        # build res10 network - for support & query x =================================================================================      
        support_x_encode = self.resnet10_encoder_meta(support_x, res10_weights, is_training=self.is_training)

        h, w, c = support_x_encode.get_shape().as_list()[1:]
        support_x_encode = tf.reduce_mean(tf.reshape(support_x_encode, [n_way, n_shot, h, w, c]), axis=1)
        support_x_encode = tf.tile(tf.expand_dims(support_x_encode, axis=0), [n_query * n_way, 1, 1, 1, 1]) 

        query_x_encode = self.resnet10_encoder_meta(query_x, res10_weights, is_training=self.is_training)
        
        query_x_encode = tf.tile(tf.expand_dims(query_x_encode, axis=0), [n_way, 1, 1, 1, 1])
        query_x_encode = tf.transpose(query_x_encode, perm=[1, 0, 2, 3, 4])

        relation_x_pairs = tf.concat([support_x_encode, query_x_encode], -1)
        relation_x_pairs = tf.reshape(relation_x_pairs, shape=[-1, h, w, c*2])

        # build relation network - for support & query x 
        relations_x = self.relation_module_meta(relation_x_pairs, relation_weights, is_training=self.is_training)  # [75*5, 1]
        relations_x = tf.reshape(relations_x, [-1, n_way])  # [75, 5]
        
        # x loss & acc
        self.x_loss = self.ce_loss(y_pred=relations_x, y_true=one_hot_labels)
        self.x_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations_x, axis=-1), labels)))

        # l2 reg
        if regularized:
            train_vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if 'conv' in v.name]) * 1e-8
            self.x_loss = tf.add(self.x_loss, l2_loss)

        # optimizer
        global_step = tf.Variable(0, trainable=False, name='global_step')
        self.train_op = tf.train.AdamOptimizer(
            self.gamma, name="train_op").minimize(self.x_loss, global_step=global_step)

    def build_meta(self, n_way, n_shot, n_query, support_x, query_x, support_a, query_b, 
                   labels, first_lr, regularized=False):
        
        ### build model
        # create network variables
        self.res10_weights = res10_weights = self.resnet10_encoder_weights()
        self.relation_weights = relation_weights = self.relation_module_weights()

        # create labels
        one_hot_labels = tf.one_hot(labels, depth=n_way)  # [75, 5]

        # build res10 network - for support & query x =================================================================================      
        support_x_encode = self.resnet10_encoder_meta(support_x, res10_weights, is_training=self.is_training)

        h, w, c = support_x_encode.get_shape().as_list()[1:]
        support_x_encode = tf.reduce_mean(tf.reshape(support_x_encode, [n_way, n_shot, h, w, c]), axis=1)
        support_x_encode = tf.tile(tf.expand_dims(support_x_encode, axis=0), [n_query * n_way, 1, 1, 1, 1]) 

        query_x_encode = self.resnet10_encoder_meta(query_x, res10_weights, is_training=self.is_training)
        
        query_x_encode = tf.tile(tf.expand_dims(query_x_encode, axis=0), [n_way, 1, 1, 1, 1])
        query_x_encode = tf.transpose(query_x_encode, perm=[1, 0, 2, 3, 4])

        relation_x_pairs = tf.concat([support_x_encode, query_x_encode], -1)
        relation_x_pairs = tf.reshape(relation_x_pairs, shape=[-1, h, w, c*2])

        # build relation network - for support & query x 
        relations_x = self.relation_module_meta(relation_x_pairs, relation_weights, is_training=self.is_training)  # [75*5, 1]
        relations_x = tf.reshape(relations_x, [-1, n_way])  # [75, 5]
        
        # x loss & acc
        self.x_loss = self.ce_loss(y_pred=relations_x, y_true=one_hot_labels)
        self.x_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations_x, axis=-1), labels)))

        # l2 reg
        if regularized:
            train_vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if 'conv' in v.name]) * 1e-8
            self.x_loss = tf.add(self.x_loss, l2_loss)

        # compute res10 gradients
        res10_grads = tf.gradients(self.x_loss, list(res10_weights.values()))
        res10_gvs = dict(zip(res10_weights.keys(), res10_grads))        

        # compute relation gradients
        relation_grads = tf.gradients(self.x_loss, list(relation_weights.values()))
        relation_gvs = dict(zip(relation_weights.keys(), relation_grads))        

        # theta_pi = theta - alpha * grads
        fast_res10_weights =    dict(zip(res10_weights.keys(),        
                                        [res10_weights[key] - first_lr * res10_gvs[key] for key in res10_weights.keys()]))
        fast_relation_weights = dict(zip(relation_weights.keys(),        
                                        [relation_weights[key] - first_lr * relation_gvs[key] for key in relation_weights.keys()]))

        # use theta_pi to forward meta-test - for support a & query b =================================================================
        support_a_encode = self.resnet10_encoder_meta(support_a, fast_res10_weights, is_training=self.is_training)

        h, w, c = support_a_encode.get_shape().as_list()[1:]
        support_a_encode = tf.reduce_mean(tf.reshape(support_a_encode, [n_way, n_shot, h, w, c]), axis=1)
        support_a_encode = tf.tile(tf.expand_dims(support_a_encode, axis=0), [n_query * n_way, 1, 1, 1, 1]) 

        query_b_encode = self.resnet10_encoder_meta(query_b, fast_res10_weights, is_training=self.is_training)
        
        query_b_encode = tf.tile(tf.expand_dims(query_b_encode, axis=0), [n_way, 1, 1, 1, 1])
        query_b_encode = tf.transpose(query_b_encode, perm=[1, 0, 2, 3, 4])

        relation_ab_pairs = tf.concat([support_a_encode, query_b_encode], -1)
        relation_ab_pairs = tf.reshape(relation_ab_pairs, shape=[-1, h, w, c*2])        

        # build relation network - for support a & query b
        relations_ab = self.relation_module_meta(relation_ab_pairs, fast_relation_weights, is_training=self.is_training)  # [75*5, 1]
        relations_ab = tf.reshape(relations_ab, [-1, n_way])  # [75, 5]

        # ab loss & acc
        self.ab_loss = self.ce_loss(y_pred=relations_ab, y_true=one_hot_labels)
        self.ab_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations_ab, axis=-1), labels)))

        # l2 reg
        if regularized:
            train_vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if 'conv' in v.name]) * 1e-8
            self.ab_loss = tf.add(self.ab_loss, l2_loss)

        # optimizer
        #optimizer = tf.train.AdamOptimizer(self.gamma, name="meta_opt")
        #gvs = optimizer.compute_gradients(self.ab_loss)
        #self.meta_op = optimizer.apply_gradients(gvs)
        self.total_loss = tf.add(self.x_loss, self.beta * self.ab_loss)
        
        global_step = tf.Variable(0, trainable=False, name='global_step')
        if self.decay is not None:
            self.gamma = tf.train.exponential_decay(self.gamma, global_step, 15000, self.decay, staircase=True)
        self.meta_op = tf.train.AdamOptimizer(
            self.gamma, name="meta_opt").minimize(self.total_loss, global_step=global_step)

    def train_baseline(self, inputs, labels, learning_rate, label_dim=64, batch_size=16, regularized=False):
        
        ### build model
        # create network variables
        self.res10_weights = res10_weights = self.resnet10_encoder_weights()

        # build res10 network - for support & query x =================================================================================      
        encode = self.resnet10_encoder_meta(inputs, res10_weights, is_training=self.is_training)
        # h, w, c = encode.get_shape().as_list()[1:]
        # print([h, w])
        avgpool = tf.nn.avg_pool(encode, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
        pred = self.resnet10_classifier(avgpool, label_dim=label_dim)

        # x loss & acc
        self.loss = self.ce_loss(y_pred=pred, y_true=labels)
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(pred, axis=-1), tf.argmax(labels, axis=-1))))

        # optimizer
        global_step = tf.Variable(0, trainable=False, name='global_step')
        self.train_op = tf.train.AdamOptimizer(
            learning_rate, name="train_op").minimize(self.loss, global_step=global_step)

    # need to correct
    def test(self, support, query, regularized=True):
        # support
        if self.backbone == 'conv4':
            support_encode = self.cnn4_encoder(support, is_training=self.is_training, reuse=True)
        else:
            support_encode = self.resnet10_encoder(support, is_training=self.is_training, reuse=True)
        h, w, c = support_encode.get_shape().as_list()[1:]
        support_encode = tf.reduce_mean(tf.reshape(support_encode, [self.n_way, self.n_shot, h, w, c]), axis=1) # wei
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

    def test_var(self, support_x, query_x, regularized=False):
        
        ### build model
        # create network variables
        self.res10_weights = res10_weights = self.resnet10_encoder_weights()
        self.relation_weights = relation_weights = self.relation_module_weights()

        # create labels
        labels = np.repeat(np.arange(self.n_way), repeats=self.n_query_test).astype(np.uint8)  # [75, 1]
        one_hot_labels = tf.one_hot(labels, depth=self.n_way)  # [75, 5]

        # build res10 network - for support & query x =================================================================================      
        support_x_encode = self.resnet10_encoder_meta(support_x, res10_weights, is_training=self.is_training)

        h, w, c = support_x_encode.get_shape().as_list()[1:]
        support_x_encode = tf.reduce_mean(tf.reshape(support_x_encode, [self.n_way, self.n_shot, h, w, c]), axis=1)
        support_x_encode = tf.tile(tf.expand_dims(support_x_encode, axis=0), [self.n_query_test * self.n_way, 1, 1, 1, 1]) 

        query_x_encode = self.resnet10_encoder_meta(query_x, res10_weights, is_training=self.is_training)
        
        query_x_encode = tf.tile(tf.expand_dims(query_x_encode, axis=0), [self.n_way, 1, 1, 1, 1])
        query_x_encode = tf.transpose(query_x_encode, perm=[1, 0, 2, 3, 4])

        relation_x_pairs = tf.concat([support_x_encode, query_x_encode], -1)
        relation_x_pairs = tf.reshape(relation_x_pairs, shape=[-1, h, w, c*2])

        # build relation network - for support & query x 
        relations_x = self.relation_module_meta(relation_x_pairs, relation_weights, is_training=self.is_training)  # [75*5, 1]
        relations_x = tf.reshape(relations_x, [-1, self.n_way])  # [75, 5]
        
        # x loss & acc
        self.test_loss = self.ce_loss(y_pred=relations_x, y_true=one_hot_labels)
        self.test_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations_x, axis=-1), labels)))

        # l2 reg
        if regularized:
            train_vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if 'conv' in v.name]) * 1e-8
            self.test_loss = tf.add(self.test_loss, l2_loss)

    def test_meta(self, support_x, query_x, regularized=False):
        
        ### build model
        # create network variables
        self.res10_weights = res10_weights = self.resnet10_encoder_weights()
        self.relation_weights = relation_weights = self.relation_module_weights()

        # create labels
        labels = np.repeat(np.arange(self.n_way), repeats=self.n_query_test).astype(np.uint8)  # [75, 1]
        one_hot_labels = tf.one_hot(labels, depth=self.n_way)  # [75, 5]

        # build res10 network - for support & query x =================================================================================      
        support_x_encode = self.resnet10_encoder_meta(support_x, res10_weights, is_training=self.is_training)

        h, w, c = support_x_encode.get_shape().as_list()[1:]
        support_x_encode = tf.reduce_mean(tf.reshape(support_x_encode, [self.n_way, self.n_shot, h, w, c]), axis=1)
        support_x_encode = tf.tile(tf.expand_dims(support_x_encode, axis=0), [self.n_query_test * self.n_way, 1, 1, 1, 1]) 

        query_x_encode = self.resnet10_encoder_meta(query_x, res10_weights, is_training=self.is_training)
        
        query_x_encode = tf.tile(tf.expand_dims(query_x_encode, axis=0), [self.n_way, 1, 1, 1, 1])
        query_x_encode = tf.transpose(query_x_encode, perm=[1, 0, 2, 3, 4])

        relation_x_pairs = tf.concat([support_x_encode, query_x_encode], -1)
        relation_x_pairs = tf.reshape(relation_x_pairs, shape=[-1, h, w, c*2])

        # build relation network - for support & query x 
        relations_x = self.relation_module_meta(relation_x_pairs, relation_weights, is_training=self.is_training)  # [75*5, 1]
        relations_x = tf.reshape(relations_x, [-1, self.n_way])  # [75, 5]
        
        # x loss & acc
        self.test_loss = self.ce_loss(y_pred=relations_x, y_true=one_hot_labels)
        self.test_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations_x, axis=-1), labels)))

        # l2 reg
        if regularized:
            train_vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if 'conv' in v.name]) * 1e-8
            self.test_loss = tf.add(self.test_loss, l2_loss)
