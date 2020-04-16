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
import src.load_data
from src.load_data import Pacs, Cub, Omniglot, MiniImageNet
from src.model import PrototypeNet, RelationNet

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
parser.add_argument('--test_iter', default=200, type=int)
parser.add_argument('--cross_domain_in_task', default=False, type=bool)
parser.add_argument('--test_domain', default='Cub', type=str)
parser.add_argument('--pretrain', default=True, type=bool)
parser.add_argument('--resume_epoch', default=0, type=int)

class Preprocessor(threading.Thread):
    def __init__(self, id, mini_obj, meta_objs, mini_queue, meta_queue, n_way, n_shot, n_query, meta_domains, cross_domain_in_task=False):
        # for thread
        self.thread_id = id
        self._stop_event = threading.Event()

        # for obj
        self.mini_obj = mini_obj
        self.meta_objs = meta_objs
        self.mini_queue = mini_queue
        self.meta_queue = meta_queue

        # for few-shot param.
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        # for pacs
        self.cross_domain_in_task = cross_domain_in_task
        self.meta_domains = meta_domains

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
        print("cross_domain_in_task = {}".format(self.cross_domain_in_task))
        print("meta_domains = {}\n".format(self.meta_domains))

        while True:
            # for mini
            mini_sup, mini_qu = self.mini_obj.get_task()
            self.mini_queue.put((mini_sup, mini_qu))
            
            # for meta
            if self.cross_domain_in_task == True:
                print("TODO") 
            else:
                selected_domain = random.sample(self.meta_domains, k=1)[0]
                meta_sup, meta_qu = self.meta_objs[selected_domain].get_task_from_raw(self.n_way, self.n_shot, self.n_query, mode='train')

            self.meta_queue.put((meta_sup, meta_qu))

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
    cross_domain_in_task = args.cross_domain_in_task
    test_domain = args.test_domain
    pretrain = args.pretrain
    resume_epoch = args.resume_epoch

    meta_domains = ['Cars', 'Cub', 'Places', 'Plantae']
    meta_domains.remove(test_domain)

    ## establish training graph
    # inputs placeholder (support and query randomly sampled from two domain)
    """
    Meta-train
    :support_x: for mini-imagenet
    :query_x: for mini-imagenet
    
    Meta-test
    :support_a: for PACS (a domain)
    :query_b: for PACS (b domain)
    
    Inference
    :support_a: for CUB
    :query_b: for CUB
    """

    support_x = tf.placeholder(tf.float32, shape=[n_way, n_shot, None, None, 3])  
    query_x = tf.placeholder(tf.float32, shape=[n_way, n_query, None, None, 3])
    support_a = tf.placeholder(tf.float32, shape=[n_way, n_shot, None, None, 3])  
    query_b = tf.placeholder(tf.float32, shape=[n_way, n_query, None, None, 3])
    is_training = tf.placeholder(tf.bool)

    # reshape
    support_x_reshape = tf.reshape(support_x, [n_way * n_shot, img_h, img_w, 3])
    query_x_reshape = tf.reshape(query_x, [n_way * n_query, img_h, img_w, 3])
    support_a_reshape = tf.reshape(support_a, [n_way * n_shot, img_h, img_w, 3])
    query_b_reshape = tf.reshape(query_b, [n_way * n_query, img_h, img_w, 3])

    labels_input = tf.placeholder(tf.int64, shape=[None])

    # Meta-RelationNet
    print("\n=== Build model...")
    print("First learning rate: ", alpha)
    print("beta: ", beta)
    print("Second learning rate: ", gamma)
    print("Decay second learning rate: ", decay)
    model = RelationNet(n_way, n_shot, n_query, alpha=alpha, beta=beta, gamma=gamma, decay=decay,
                        backbone='resnet', is_training=is_training)
    model.build_meta(n_way, n_shot, n_query,
                     support_x=support_x_reshape, query_x=query_x_reshape, 
                     support_a=support_a_reshape, query_b=query_b_reshape,
                     labels=labels_input, first_lr=alpha)
    
    model_summary()

    # saver for saving session
    regular_saver = tf.train.Saver()
    best_saver = tf.train.Saver(max_to_keep=None)

    wei_path = '/data/wei/model/cross-domain-few-shot/logs'
    log_path = os.path.join(wei_path, args.log_path, '_'.join(
        (args.test_name, 'alpha'+str(alpha), 'beta'+str(beta), 'gamma'+str(gamma), 'decay'+str(decay))))
    lastest_checkpoint = tf.train.latest_checkpoint(log_path)

    pretrain_log_path = os.path.join(wei_path, 'pretrain_baseline', 'pretrain_baseline_batch64_en200_lr0.001_decay0.96_240k') 
    pretrain_ckeckpoint = tf.train.latest_checkpoint(pretrain_log_path)

    best_checkpoint_path = os.path.join(log_path, 'best_performance')
    checkpoint_file = os.path.join(log_path, "checkpoint.ckpt")

    init = [
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
    ]

    def restore_from_checkpoint(sess, saver, checkpoint):
        if checkpoint:
            print("Restore session from checkpoint: {}".format(checkpoint))
            saver.restore(sess, checkpoint)
            return True
        else:
            print("Checkpoint not found: {}".format(checkpoint))
            return False

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

    print("\n=== Load data...")
    # make queue
    mini_queue = queue.Queue(maxsize = 10)
    meta_queue = queue.Queue(maxsize = 10)

    # Init mini-imagenet
    mini = MiniImageNet()

    # Init meta domain classes
    meta = {}
    for curr_domain in meta_domains:
        fn = getattr(src.load_data, curr_domain)
        meta[curr_domain] = fn()

    # Init test domain class
    fn = getattr(src.load_data, test_domain)
    test_obj = fn()
    
    # pass obj to preprocess thread and start
    pre_thread = Preprocessor(0, mini, meta, mini_queue, meta_queue, n_way, n_shot, n_query, meta_domains=meta_domains, cross_domain_in_task=cross_domain_in_task)
    pre_thread.start()

    ## training
    with tf.Session() as sess:

        labels = np.repeat(np.arange(n_way), repeats=n_query).astype(np.uint8)

        # Creates a file writer for the log directory.
        file_writer_train = tf.summary.FileWriter(os.path.join(log_path, "train"), sess.graph)
        file_writer_val = tf.summary.FileWriter(os.path.join(log_path, "val"), sess.graph)
        file_writer_test = tf.summary.FileWriter(os.path.join(log_path, "test"), sess.graph)

        # store variables
        tf.summary.image("Support x image", support_x_reshape[:4], max_outputs=4, collections=['train', 'val', 'test'])
        tf.summary.image("Query x image", query_x_reshape[:4], max_outputs=4, collections=['train', 'val', 'test'])
        tf.summary.scalar("x loss", model.x_loss, collections=['train', 'val', 'test'])
        tf.summary.scalar("x acc", model.x_acc, collections=['train', 'val', 'test'])

        tf.summary.image("Support a image", support_a_reshape[:4], max_outputs=4, collections=['train', 'val'])
        tf.summary.image("Query b image", query_b_reshape[:4], max_outputs=4, collections=['train', 'val'])
        tf.summary.scalar("ab loss", model.ab_loss, collections=['train', 'val'])
        tf.summary.scalar("ab acc", model.ab_acc, collections=['train', 'val'])        
        
        train_merged = tf.summary.merge_all('train')
        val_merged = tf.summary.merge_all('val')
        test_merged = tf.summary.merge_all('test')
        
        sess.run(init)
        print("\n=== Ckeck pretrained model...")
        if lastest_checkpoint:
            restore_from_checkpoint(sess, regular_saver, lastest_checkpoint)
        elif pretrain:
            restore_from_pretrain(sess, pretrain_ckeckpoint)

        print("\n=== Start training...")
        preprocess_time = 0
        train_time = 0
        for i_iter in range(resume_epoch, args.n_iter):

            start = timeit.default_timer()
            # mini-imagenet ======================================================================
            curr_support_x, curr_query_x = mini_queue.get()

            # meta ===============================================================================
            curr_support_a, curr_query_b = meta_queue.get()
            stop = timeit.default_timer()

            preprocess_time += (stop - start) 

            start = timeit.default_timer()
            # training                 
            sess.run([model.meta_op], feed_dict={
                support_x: curr_support_x,
                query_x: curr_query_x,
                support_a: curr_support_a,
                query_b: curr_query_b,                
                labels_input: labels,
                is_training: True
            })
            stop = timeit.default_timer()

            train_time += (stop - start) 

            # evaluation
            if i_iter % 50 == 0:
                # evaluation
                summary_train, train_x_loss, train_x_acc, train_ab_loss, train_ab_acc = \
                    sess.run([train_merged, model.x_loss, model.x_acc, model.ab_loss, model.ab_acc], feed_dict={
                        support_x: curr_support_x,
                        query_x: curr_query_x,
                        support_a: curr_support_a,
                        query_b: curr_query_b,
                        labels_input: labels,
                        is_training: False
                    })
                
                # monitor validation
                # get task from validation set of mini-ImageNet
                # TODO: cross domain in task
                curr_support_x, curr_query_x = mini.get_task(n_way, n_shot, n_query, aug=False, mode='val')
                selected_domain = random.sample(meta_domains, k=1)[0] 
                curr_support_a, curr_query_b = meta[selected_domain].get_task_from_raw(n_way, n_shot, n_query, aug=False, mode='val')
            
                summary_val, val_x_loss, val_x_acc, val_ab_loss, val_ab_acc = \
                    sess.run([val_merged, model.x_loss, model.x_acc, model.ab_loss, model.ab_acc], feed_dict={
                        support_x: curr_support_x,
                        query_x: curr_query_x,
                        support_a: curr_support_a,
                        query_b: curr_query_b,
                        labels_input: labels,
                        is_training: False
                    })

                # get task from unseen domain (CUB) done
                test_support, test_query = test_obj.get_task_from_raw(n_way, n_shot, n_query, aug=False, mode='test')
                summary_test, test_x_loss, test_x_acc = \
                    sess.run([test_merged, model.x_loss, model.x_acc], feed_dict={
                        support_x: test_support,
                        query_x: test_query,
                        labels_input: labels,
                        is_training: False
                    })

                # log all variables
                file_writer_train.add_summary(summary_train, global_step=i_iter)
                file_writer_val.add_summary(summary_val, global_step=i_iter)
                file_writer_test.add_summary(summary_test, global_step=i_iter)

                print('Iteration: %d' % (i_iter+1))
                print('train x: [%f, %f], train ab: [%f, %f]' % (train_x_loss, train_x_acc, train_ab_loss, train_ab_acc))
                print('val x: [%f, %f], val ab: [%f, %f]' % (val_x_loss, val_x_acc, val_ab_loss, val_ab_acc))
                print('==> CUB: [%f, %f]' % (test_x_loss, test_x_acc))
                print("==> Preporcess time: ", preprocess_time/(i_iter+1))
                print("==> Train time: ", train_time/(i_iter+1))

            # save session every 2000 iteration
            if (i_iter+1) % 2000 == 0:    
                regular_saver.save(sess, checkpoint_file, global_step=(i_iter+1))
                print('Save session at step %d' % (i_iter+1))

        # compute accuracy on 600 randomly generated task from mini-ImageNet, and CUB
        test_iter = args.test_iter
        acc = np.empty((test_iter, 2))
        for i in range(test_iter):
            curr_support_x, curr_query_x = mini.get_task(n_way, n_shot, n_query, aug=False, mode='test')
            test_support, test_query = test_obj.get_task_from_raw(n_way, n_shot, n_query, aug=False, mode='test')

            test_x_acc = sess.run([model.x_acc], feed_dict={
                support_x: curr_support_x,
                query_x: curr_query_x,
                labels_input: labels,
                is_training: False
            })
            
            cub_x_acc = sess.run([model.x_acc], feed_dict={
                support_x: test_support,
                query_x: test_query,
                labels_input: labels,
                is_training: False
            })

            acc[i] = np.concatenate([test_x_acc, cub_x_acc])
            
        mean_acc = mean(acc, axis=0)
        confidence = 0.95
        std_err = sem(acc)
        h = std_err * t.ppf((1 + confidence) / 2, test_iter - 1)
        
        print('Overall mini-ImageNet acc: {}+-{}%'.format(mean_acc[0], h[0]))
        print('Overall CUB acc: {}+-{}%'.format(mean_acc[1], h[1]))

        pre_thread.stop()
        pre_thread.join()
        
        file_writer_train.close()
        file_writer_test.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
