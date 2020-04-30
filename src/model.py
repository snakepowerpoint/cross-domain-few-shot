import tensorflow as tf
import numpy as np

from src.model_utils import convolution_layer, max_pool, batchnorm_conv, fc_layer
from src.model_utils import convolution_layer_meta, fc_layer_meta


class RelationNet(object):
    def __init__(self,
                 alpha=1e-3,
                 beta=1.0,
                 gamma=1e-3,
                 decay=0.96, 
                 backbone='resnet',
                 is_training=True):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.decay = decay
        self.backbone = backbone
        self.is_training = is_training

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
        conv_initializer = tf.keras.initializers.he_normal()

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
            net = fc_layer(inputs, label_dim, is_bias=True, name='fc', activat_fn=None) # wei, enlarge the output dimension to 200
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
        conv_initializer = tf.keras.initializers.he_normal()
        fc_initializer = tf.keras.initializers.he_normal()

        with tf.variable_scope('relation_mod_weights', reuse=tf.AUTO_REUSE):
            weights['conv1']    = tf.get_variable('conv1w', [3, 3, 1024, 512],  initializer=conv_initializer, dtype=dtype)
            weights['b1']       = tf.get_variable('conv1b', initializer=tf.zeros([512]))
 
            weights['conv2']    = tf.get_variable('conv2w', [3, 3, 512, 512],  initializer=conv_initializer, dtype=dtype)
            weights['b2']       = tf.get_variable('conv2b', initializer=tf.zeros([512]))
            
            weights['fc3']      = tf.get_variable('fc3w', [512, 8], initializer=fc_initializer, dtype=dtype)
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
                   labels, first_lr, beta, regularized=False):
        
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
        self.total_loss = tf.add(self.x_loss, beta * self.ab_loss)
        
        global_step = tf.Variable(0, trainable=False, name='global_step')
        if self.decay is not None:
            self.gamma = tf.train.exponential_decay(self.gamma, global_step, 15000, self.decay, staircase=True)
        self.meta_op = tf.train.AdamOptimizer(
            self.gamma, name="meta_opt").minimize(self.total_loss, global_step=global_step)

    def build_maml(self, n_way, n_shot, n_query, support_x, query_x, support_a, query_b, labels, first_lr, multi_tasks=False):
        
        img_h, img_w = support_x.get_shape()[2], support_x.get_shape()[3]
        #num_domains = support_a.get_shape()[0]

        support_x = tf.reshape(support_x, [n_way * n_shot, img_h, img_w, 3])
        query_x = tf.reshape(query_x, [n_way * n_query, img_h, img_w, 3])
        support_a = tf.reshape(support_a, [-1, n_way * n_shot, img_h, img_w, 3])
        query_b = tf.reshape(query_b, [-1, n_way * n_query, img_h, img_w, 3])

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
        self.ab_loss, self.ab_acc = [], []   
        if multi_tasks and self.is_training:
            num_domains = 3
        else:
            num_domains = 1
        for i in range(num_domains):
            support_a_encode = self.resnet10_encoder_meta(support_a[i], fast_res10_weights, is_training=self.is_training)

            h, w, c = support_a_encode.get_shape().as_list()[1:]
            support_a_encode = tf.reduce_mean(tf.reshape(support_a_encode, [n_way, n_shot, h, w, c]), axis=1)
            support_a_encode = tf.tile(tf.expand_dims(support_a_encode, axis=0), [n_query * n_way, 1, 1, 1, 1]) 

            query_b_encode = self.resnet10_encoder_meta(query_b[i], fast_res10_weights, is_training=self.is_training)
            
            query_b_encode = tf.tile(tf.expand_dims(query_b_encode, axis=0), [n_way, 1, 1, 1, 1])
            query_b_encode = tf.transpose(query_b_encode, perm=[1, 0, 2, 3, 4])

            relation_ab_pairs = tf.concat([support_a_encode, query_b_encode], -1)
            relation_ab_pairs = tf.reshape(relation_ab_pairs, shape=[-1, h, w, c*2])        

            # build relation network - for support a & query b
            relations_ab = self.relation_module_meta(relation_ab_pairs, fast_relation_weights, is_training=self.is_training)  # [75*5, 1]
            relations_ab = tf.reshape(relations_ab, [-1, n_way])  # [75, 5]

            # ab loss & acc
            self.ab_loss.append(self.ce_loss(y_pred=relations_ab, y_true=one_hot_labels))
            self.ab_acc.append(tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations_ab, axis=-1), labels))))

        self.ab_loss = tf.reduce_mean(self.ab_loss)
        self.ab_acc = tf.reduce_mean(self.ab_acc)
        self.total_loss = self.x_loss + self.beta * self.ab_loss
        
        global_step = tf.Variable(0, trainable=False, name='global_step')
        if self.decay is not None:
            self.gamma = tf.train.exponential_decay(self.gamma, global_step, 15000, self.decay, staircase=True)
        self.meta_op = tf.train.AdamOptimizer(
            self.gamma, name="meta_opt").minimize(self.total_loss, global_step=global_step)

    def build_maml_val(self, n_way, n_shot, n_query, train_support, train_query, test_support, test_query, labels, first_lr):

        img_h, img_w = train_support.get_shape()[2], train_support.get_shape()[3]

        train_support = tf.reshape(train_support, [n_way * n_shot, img_h, img_w, 3])
        train_query = tf.reshape(train_query, [n_way * n_query, img_h, img_w, 3])
        test_support = tf.reshape(test_support, [n_way * n_shot, img_h, img_w, 3])
        test_query = tf.reshape(test_query, [n_way * n_query, img_h, img_w, 3])

        ### build model
        # create network variables
        res10_weights = self.resnet10_encoder_weights()
        relation_weights = self.relation_module_weights()

        # create labels
        one_hot_labels = tf.one_hot(labels, depth=n_way)  # [75, 5]
        
        # Use theta from mini to start =================================================================================     
        support_encode = self.resnet10_encoder_meta(train_support, res10_weights, is_training=self.is_training)

        h, w, c = support_encode.get_shape().as_list()[1:]
        support_encode = tf.reduce_mean(tf.reshape(support_encode, [n_way, n_shot, h, w, c]), axis=1)
        support_encode = tf.tile(tf.expand_dims(support_encode, axis=0), [n_query * n_way, 1, 1, 1, 1]) 

        query_encode = self.resnet10_encoder_meta(train_query, res10_weights, is_training=self.is_training)
        
        query_encode = tf.tile(tf.expand_dims(query_encode, axis=0), [n_way, 1, 1, 1, 1])
        query_encode = tf.transpose(query_encode, perm=[1, 0, 2, 3, 4])

        relation_pairs = tf.concat([support_encode, query_encode], -1)
        relation_pairs = tf.reshape(relation_pairs, shape=[-1, h, w, c*2])

        # build relation network 
        relations = self.relation_module_meta(relation_pairs, relation_weights, is_training=self.is_training)  # [75*5, 1]
        relations = tf.reshape(relations, [-1, n_way])  # [75, 5]
        
        # maml train loss & acc
        self.maml_train_loss = self.ce_loss(y_pred=relations, y_true=one_hot_labels)
        self.maml_train_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations, axis=-1), labels)))

        # compute res10 gradients
        res10_grads = tf.gradients(self.maml_train_loss, list(res10_weights.values()))
        res10_gvs = dict(zip(res10_weights.keys(), res10_grads))        

        # compute relation gradients
        relation_grads = tf.gradients(self.maml_train_loss, list(relation_weights.values()))
        relation_gvs = dict(zip(relation_weights.keys(), relation_grads))        

        # Update theta by training data from test dataset
        # theta_pi = theta - alpha * grads
        fast_res10_weights =    dict(zip(res10_weights.keys(),        
                                        [res10_weights[key] - first_lr * res10_gvs[key] for key in res10_weights.keys()]))
        fast_relation_weights = dict(zip(relation_weights.keys(),        
                                        [relation_weights[key] - first_lr * relation_gvs[key] for key in relation_weights.keys()]))

        # Update theta N times
        for _ in range(4):
            support_encode = self.resnet10_encoder_meta(train_support, fast_res10_weights, is_training=self.is_training)

            h, w, c = support_encode.get_shape().as_list()[1:]
            support_encode = tf.reduce_mean(tf.reshape(support_encode, [n_way, n_shot, h, w, c]), axis=1)
            support_encode = tf.tile(tf.expand_dims(support_encode, axis=0), [n_query * n_way, 1, 1, 1, 1]) 

            query_encode = self.resnet10_encoder_meta(train_query, fast_res10_weights, is_training=self.is_training)
            
            query_encode = tf.tile(tf.expand_dims(query_encode, axis=0), [n_way, 1, 1, 1, 1])
            query_encode = tf.transpose(query_encode, perm=[1, 0, 2, 3, 4])

            relation_pairs = tf.concat([support_encode, query_encode], -1)
            relation_pairs = tf.reshape(relation_pairs, shape=[-1, h, w, c*2])

            # build relation network 
            relations = self.relation_module_meta(relation_pairs, fast_relation_weights, is_training=self.is_training)  # [75*5, 1]
            relations = tf.reshape(relations, [-1, n_way])  # [75, 5]
            
            # maml train loss & acc
            self.maml_train_loss = self.ce_loss(y_pred=relations, y_true=one_hot_labels)
            self.maml_train_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations, axis=-1), labels)))

            # compute res10 gradients
            res10_grads = tf.gradients(self.maml_train_loss, list(fast_res10_weights.values()))
            res10_gvs = dict(zip(fast_res10_weights.keys(), res10_grads))        

            # compute relation gradients
            relation_grads = tf.gradients(self.maml_train_loss, list(fast_relation_weights.values()))
            relation_gvs = dict(zip(fast_relation_weights.keys(), relation_grads))        

            # Update theta by training data from test dataset
            # theta_pi = theta - alpha * grads
            fast_res10_weights =    dict(zip(fast_res10_weights.keys(),        
                                            [fast_res10_weights[key] - first_lr * res10_gvs[key] for key in fast_res10_weights.keys()]))
            fast_relation_weights = dict(zip(fast_relation_weights.keys(),        
                                            [fast_relation_weights[key] - first_lr * relation_gvs[key] for key in fast_relation_weights.keys()]))

        # Evaluate the test acc by test data
        # use theta_pi to forward meta-test
        support_encode = self.resnet10_encoder_meta(test_support, fast_res10_weights, is_training=self.is_training)

        h, w, c = support_encode.get_shape().as_list()[1:]
        support_encode = tf.reduce_mean(tf.reshape(support_encode, [n_way, n_shot, h, w, c]), axis=1)
        support_encode = tf.tile(tf.expand_dims(support_encode, axis=0), [n_query * n_way, 1, 1, 1, 1]) 

        query_encode = self.resnet10_encoder_meta(test_query, fast_res10_weights, is_training=self.is_training)
        
        query_encode = tf.tile(tf.expand_dims(query_encode, axis=0), [n_way, 1, 1, 1, 1])
        query_encode = tf.transpose(query_encode, perm=[1, 0, 2, 3, 4])

        relation_pairs = tf.concat([support_encode, query_encode], -1)
        relation_pairs = tf.reshape(relation_pairs, shape=[-1, h, w, c*2])        

        # build relation network 
        relations = self.relation_module_meta(relation_pairs, fast_relation_weights, is_training=self.is_training)  # [75*5, 1]
        relations = tf.reshape(relations, [-1, n_way])  # [75, 5]

        # maml test loss & acc
        self.maml_test_loss = self.ce_loss(y_pred=relations, y_true=one_hot_labels)
        self.maml_test_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations, axis=-1), labels)))

    def build_maml_test_train(self, n_way, n_shot, n_query, train_support, train_query, labels, first_lr):

        img_h, img_w = train_support.get_shape()[2], train_support.get_shape()[3]

        train_support = tf.reshape(train_support, [n_way * n_shot, img_h, img_w, 3])
        train_query = tf.reshape(train_query, [n_way * n_query, img_h, img_w, 3])

        ### build model
        # create network variables
        res10_weights = self.resnet10_encoder_weights()
        relation_weights = self.relation_module_weights()

        # create labels
        one_hot_labels = tf.one_hot(labels, depth=n_way)  # [75, 5]
        
        # Use theta from mini to start =================================================================================     
        support_encode = self.resnet10_encoder_meta(train_support, res10_weights, is_training=self.is_training)

        h, w, c = support_encode.get_shape().as_list()[1:]
        support_encode = tf.reduce_mean(tf.reshape(support_encode, [n_way, n_shot, h, w, c]), axis=1)
        support_encode = tf.tile(tf.expand_dims(support_encode, axis=0), [n_query * n_way, 1, 1, 1, 1]) 

        query_encode = self.resnet10_encoder_meta(train_query, res10_weights, is_training=self.is_training)
        
        query_encode = tf.tile(tf.expand_dims(query_encode, axis=0), [n_way, 1, 1, 1, 1])
        query_encode = tf.transpose(query_encode, perm=[1, 0, 2, 3, 4])

        relation_pairs = tf.concat([support_encode, query_encode], -1)
        relation_pairs = tf.reshape(relation_pairs, shape=[-1, h, w, c*2])

        # build relation network 
        relations = self.relation_module_meta(relation_pairs, relation_weights, is_training=self.is_training)  # [75*5, 1]
        relations = tf.reshape(relations, [-1, n_way])  # [75, 5]
        
        # maml train loss & acc
        self.maml_train_loss = self.ce_loss(y_pred=relations, y_true=one_hot_labels)
        self.maml_train_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations, axis=-1), labels)))
        
        global_step = tf.Variable(0, trainable=False, name='global_step')
        self.meta_op = tf.train.AdamOptimizer(
            self.alpha, name="meta_opt").minimize(self.maml_train_loss, global_step=global_step)

    def build_maml_test_inference(self, n_way, n_shot, n_query, test_support, test_query, labels):

        img_h, img_w = test_support.get_shape()[2], test_support.get_shape()[3]

        test_support = tf.reshape(test_support, [n_way * n_shot, img_h, img_w, 3])
        test_query = tf.reshape(test_query, [n_way * n_query, img_h, img_w, 3])

        # create network variables
        res10_weights = self.resnet10_encoder_weights()
        relation_weights = self.relation_module_weights()

        # create labels
        one_hot_labels = tf.one_hot(labels, depth=n_way)  # [75, 5]

        # Evaluate the test acc by test data
        # use theta_pi to forward meta-test
        support_encode = self.resnet10_encoder_meta(test_support, res10_weights, is_training=self.is_training)

        h, w, c = support_encode.get_shape().as_list()[1:]
        support_encode = tf.reduce_mean(tf.reshape(support_encode, [n_way, n_shot, h, w, c]), axis=1)
        support_encode = tf.tile(tf.expand_dims(support_encode, axis=0), [n_query * n_way, 1, 1, 1, 1]) 

        query_encode = self.resnet10_encoder_meta(test_query, res10_weights, is_training=self.is_training)
        
        query_encode = tf.tile(tf.expand_dims(query_encode, axis=0), [n_way, 1, 1, 1, 1])
        query_encode = tf.transpose(query_encode, perm=[1, 0, 2, 3, 4])

        relation_pairs = tf.concat([support_encode, query_encode], -1)
        relation_pairs = tf.reshape(relation_pairs, shape=[-1, h, w, c*2])        

        # build relation network 
        relations = self.relation_module_meta(relation_pairs, relation_weights, is_training=self.is_training)  # [75*5, 1]
        relations = tf.reshape(relations, [-1, n_way])  # [75, 5]

        # maml test loss & acc
        self.maml_test_loss = self.ce_loss(y_pred=relations, y_true=one_hot_labels)
        self.maml_test_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(relations, axis=-1), labels)))

    def train_baseline(self, inputs, labels, learning_rate, label_dim=64, regularized=False):
        
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
