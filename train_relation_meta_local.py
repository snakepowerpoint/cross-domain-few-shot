# deep learning framework
import tensorflow as tf
import tensorflow.contrib.slim as slim

# computation
import numpy as np
import random
from scipy.stats import sem, t
from scipy import mean
import timeit

# io
import os
import argparse

# customerized 
from src.load_data import Pacs, Cub, Omniglot, MiniImageNet
from src.model_local import PrototypeNet, RelationNet ### use model_local.py

# miscellaneous
import gc

# multi-thread
import threading
import queue

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', default='baseline_meta', type=str)
parser.add_argument('--test_name', default='test_meta', type=str)
parser.add_argument('--n_way', default=5, type=int)
parser.add_argument('--n_shot', default=5, type=int)
parser.add_argument('--n_query', default=16, type=int)
parser.add_argument('--alpha', default=1e-3, type=float)
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--gamma', default=5e-3, type=float)
parser.add_argument('--decay', default=0.96, type=float)
parser.add_argument('--n_iter', default=40000, type=int)
parser.add_argument('--multi_domain', default=False, type=bool)
parser.add_argument('--pretrain', default=True, type=bool)

class Preprocessor(threading.Thread):
    def __init__(self, id, mini_obj, pacs_obj, mini_queue, pacs_queue, n_way, n_shot, n_query, multi_domain=True):
        # for thread
        self.thread_id = id
        self._stop_event = threading.Event()

        # for obj
        self.mini_obj = mini_obj
        self.pacs_obj = pacs_obj
        self.mini_queue = mini_queue
        self.pacs_queue = pacs_queue

        # for few-shot param.
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        # for pacs
        self.multi_domain = multi_domain
        self.pacs_domains = list(self.pacs_obj.data_dict['train'].keys())
        self.pacs_categories = list(self.pacs_obj.data_dict['train'][self.pacs_domains[0]].keys())

        threading.Thread.__init__(self)

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()        

    def run(self):
        print("\n=== Process {} start!".format(self.thread_id))
        print("n_way = {}".format(self.n_way))
        print("n_shot = {}".format(self.n_shot))
        print("n_query = {}".format(self.n_query))
        print("multi_domain = {}\n".format(self.multi_domain))

        while True:
            # for mini
            mini_sup, mini_qu = self.mini_obj.get_task()
            self.mini_queue.put((mini_sup, mini_qu))
            
            # for pacs
            selected_categories = random.sample(self.pacs_categories, k=self.n_way)
            if self.multi_domain == True:
                pacs_domain_a, pacs_domain_b = random.sample(self.pacs_domains, k=2)
                pacs_sup, _ = self.pacs_obj.get_task(pacs_domain_a, selected_categories, self.n_shot, self.n_query, mode='train', target='support')
                _, pacs_qu = self.pacs_obj.get_task(pacs_domain_b, selected_categories, self.n_shot, self.n_query, mode='train', target='query')
            else:
                pacs_domain = random.sample(self.pacs_domains, k=1)
                pacs_sup, pacs_qu = self.pacs_obj.get_task(pacs_domain[0], selected_categories, self.n_shot, self.n_query, mode='train', target='all')

            self.pacs_queue.put((pacs_sup, pacs_qu))

            if self.is_stopped():
                break

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def main(args):
    img_w = 224
    img_h = 224
    
    n_way = args.n_way
    n_shot = args.n_shot
    n_query = args.n_query
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    decay = args.decay
    multi_domain = args.multi_domain
    pretrain = args.pretrain
       
    ## establish training graph
    # inputs placeholder (support and query randomly sampled from two domain)
    """
    
    Meta-train
    :support_x: for mini-imagenet
    :query_x: for mini-imagenet
    
    Meta-test
    :support_a: for PACS
    :query_b: for PACS
    
    """
    support_x = tf.placeholder(tf.float32, shape=[n_way, n_shot, None, None, 3])  
    query_x = tf.placeholder(tf.float32, shape=[n_way, n_query, None, None, 3])
    support_a = tf.placeholder(tf.float32, shape=[n_way, n_shot, None, None, 3])  
    query_b = tf.placeholder(tf.float32, shape=[n_way, n_query, None, None, 3])
    is_training = tf.placeholder(tf.bool)

    beta_param = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    # reshape
    support_x_reshape = tf.reshape(support_x, [n_way * n_shot, img_h, img_w, 3])
    query_x_reshape = tf.reshape(query_x, [n_way * n_query, img_h, img_w, 3])
    support_a_reshape = tf.reshape(support_a, [n_way * n_shot, img_h, img_w, 3])
    query_b_reshape = tf.reshape(query_b, [n_way * n_query, img_h, img_w, 3])

    # Meta-RelationNet
    print("\n=== Build model...")
    print("First learning rate: ", alpha)
    print("beta: ", beta)
    print("Second learning rate: ", gamma)
    print("Decay second learning rate: ", decay)
    model = RelationNet(n_way, n_shot, n_query, alpha=alpha, gamma=gamma, decay=decay,
                        backbone='resnet', is_training=is_training)
    model.train_meta(   support_x=support_x_reshape, query_x=query_x_reshape, 
                        support_a=support_a_reshape, query_b=query_b_reshape,
                        beta=beta_param, regularized=True)

    model_summary()

    # saver for saving session
    saver = tf.train.Saver()
    log_path = os.path.join('logs', args.log_path, '_'.join(
        (args.test_name, 'alpha'+str(alpha), 'beta'+str(beta), 'gamma'+str(gamma), 'decay'+str(decay))))
    
    checkpoint_file = log_path + "/checkpoint.ckpt"
    lastest_checkpoint = tf.train.latest_checkpoint(log_path)

    pretrain_log_path = os.path.join('logs', 'pretrain_baseline', 'pretrain_baseline_val_lr0.001_decay1.0') 
    pretrain_ckeckpoint = tf.train.latest_checkpoint(pretrain_log_path)

    init = [
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            ]

    def restore_from_pretrain(sess, checkpoint):
        if checkpoint:
            print("Restore pretrained weights from checkpoint: {}".format(checkpoint))
            var = tf.global_variables()
            var_to_restore = [val  for val in var if (('res10_weights' in val.name) or ('bn' in val.name)) and ('meta_op' not in val.name) and ('M_' not in val.name)]
            
            print("Pretrained weights:")
            for _var in var_to_restore:
                print(_var)

            saver = tf.train.Saver(var_to_restore)
            saver.restore(sess, checkpoint)
            return True
        else:
            print("Pretrained checkpoint not found: {}".format(checkpoint))
            return False

    def restore_from_checkpoint(sess, saver, checkpoint):
        if checkpoint:
            print("Restore session from checkpoint: {}".format(checkpoint))
            saver.restore(sess, checkpoint)
            return True
        else:
            print("Checkpoint not found: {}".format(checkpoint))
            return False


    print("\n=== Load data...")
    # make queue
    mini_queue = queue.Queue(maxsize = 10)
    pacs_queue = queue.Queue(maxsize = 10)

    # load mini-imagenet
    mini = MiniImageNet()

    # load PACS
    pacs = Pacs()

    # load CUB data
    cub = Cub(mode='test')

    # pass obj to preprocess thread and start
    pre_thread = Preprocessor(0, mini, pacs, mini_queue, pacs_queue, n_way, n_shot, n_query, multi_domain=multi_domain)
    pre_thread.start()

    ## training
    with tf.Session() as sess:

        # Creates a file writer for the log directory.
        file_writer_train = tf.summary.FileWriter(os.path.join(log_path, "train"), sess.graph)
        file_writer_test = tf.summary.FileWriter(os.path.join(log_path, "test"), sess.graph)

        # store variables
        tf.summary.image("Support x image", support_x_reshape[:4], max_outputs=4, collections=['train', 'test'])
        tf.summary.image("Query x image", query_x_reshape[:4], max_outputs=4, collections=['train', 'test'])
        tf.summary.scalar("x loss", model.x_loss, collections=['train', 'test'])
        tf.summary.scalar("x acc", model.x_acc, collections=['train', 'test'])

        tf.summary.image("Support a image", support_a_reshape[:4], max_outputs=4, collections=['train'])
        tf.summary.image("Query b image", query_b_reshape[:4], max_outputs=4, collections=['train'])
        tf.summary.scalar("ab loss", model.ab_loss, collections=['train'])
        tf.summary.scalar("ab acc", model.ab_acc, collections=['train'])        
        
        train_merged = tf.summary.merge_all('train')
        test_merged = tf.summary.merge_all('test')
        
        sess.run(init)
        print("\n=== Ckeck pretrained model...")
        if lastest_checkpoint:
            restore_from_checkpoint(sess, saver, lastest_checkpoint)
        elif pretrain:
            restore_from_pretrain(sess, pretrain_ckeckpoint)

        print("\n=== Start training...")
        preprocess_time = 0
        train_time = 0
        for i_iter in range(args.n_iter):

            start = timeit.default_timer()
            # mini-imagenet ======================================================================
            curr_support_x, curr_query_x = mini_queue.get()

            # PACS ===============================================================================
            curr_support_a, curr_query_b = pacs_queue.get()
            stop = timeit.default_timer()

            preprocess_time += (stop - start) 

            # pretrain miniimagenet for 5K
#            if i_iter > 5000:
#                beta = 1.0
#                print("=== Start meta-learning: Beta = [{}]", beta)
#            else:
#                beta = 0.0

            start = timeit.default_timer()
            # training                 
            sess.run([model.meta_op], feed_dict={
                support_x: curr_support_x,
                query_x: curr_query_x,
                support_a: curr_support_a,
                query_b: curr_query_b,
                beta_param: beta,
                is_training: True
            })
            stop = timeit.default_timer()

            train_time += (stop - start) 

            # evaluation
            if i_iter % 50 == 0:
                # evaluation
                summary_train, train_x_loss, train_x_acc, train_ab_loss, train_ab_acc = \
                    sess.run([train_merged, model.x_loss, model.x_acc, model.ab_loss, model.ab_acc], 
                        feed_dict={
                            support_x: curr_support_x,
                            query_x: curr_query_x,
                            support_a: curr_support_a,
                            query_b: curr_query_b,                
                            beta_param: beta,
                            is_training: False
                        })

                # get task from unseen domain (CUB)
                #cub_support, cub_query = cub.get_task(n_way, n_shot, n_query)
                cub_support, cub_query = cub.get_task_from_raw(n_way, n_shot, n_query)
                summary_test, cub_loss, cub_acc = \
                    sess.run([test_merged, model.x_loss, model.x_acc], 
                        feed_dict={
                            support_x: cub_support,
                            query_x: cub_query, 
                            support_a: curr_support_a,
                            query_b: curr_query_b,                                        
                            beta_param: beta,
                            is_training: False
                        })

                # log all variables
                file_writer_train.add_summary(summary_train, global_step=i_iter)
                file_writer_test.add_summary(summary_test, global_step=i_iter)

                print('Iteration: %d, train x : [%f, %f], train ab: [%f, %f]' % 
                    (i_iter+1, train_x_loss, train_x_acc, train_ab_loss, train_ab_acc))
                print('==> CUB: [%f, %f]' % (cub_loss, cub_acc))
                print("==> Preporcess time: ", preprocess_time/(i_iter+1))
                print("==> Train time: ", train_time/(i_iter+1))

            # save session every 2000 iteration
            if (i_iter+1) % 2000 == 0:    
                saver.save(sess, checkpoint_file, global_step=(i_iter+1))
                print('Save session at step %d' % (i_iter+1))

        pre_thread.stop()
        pre_thread.join()

        file_writer_train.close()
        file_writer_test.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
