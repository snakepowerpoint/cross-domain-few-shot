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
import errno
import argparse

# customerized 
import src.load_data
from src.load_data import Pacs, Cub, Omniglot, MiniImageNet
from src.model import RelationNet

# miscellaneous
import gc

# multi-thread
import threading
import queue

from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', default='baseline_maml', type=str)
parser.add_argument('--test_name', default='test_maml', type=str)
parser.add_argument('--ckpt_name', default='No', type=str)
parser.add_argument('--n_way', default=5, type=int)
parser.add_argument('--n_shot', default=5, type=int)
parser.add_argument('--n_query', default=15, type=int)
parser.add_argument('--alpha', default=1e-3, type=float)
parser.add_argument('--test_iter', default=200, type=int)
parser.add_argument('--repeat_exp', default=100, type=int)
parser.add_argument('--test_domain', default='Cub', type=str)
parser.add_argument('--load_task', default='No', type=str)

class Preprocessor(threading.Thread):
    def __init__(self, id, test_obj, test_queue, n_way, n_shot, n_query):
        # for thread
        self.thread_id = id
        self._stop_event = threading.Event()

        # for obj
        self.test_obj = test_obj
        self.test_queue = test_queue

        # for few-shot param.
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

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

        while True:
            # for mini
            test_sup, test_qu = self.test_obj.get_task_from_raw(self.n_way, self.n_shot, self.n_query, aug=False, mode='test')
            self.test_queue.put((test_sup, test_qu))

            if self.is_stopped():
                break

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def dump_train_task(pkl_path, ckpt_name, acc, data):
    dump_path = os.path.join(pkl_path, ckpt_name)
    mkdir_p(dump_path)
    print("==> Dump training task: {}".format(os.path.join(dump_path, str(acc) + '.pickle')))
    f = open(os.path.join(dump_path, str(acc) + '.pickle'), 'wb')
    pickle.dump(data, f)
    f.close()

def main(args):
    img_w = 224
    img_h = 224

    n_way = args.n_way
    n_shot = args.n_shot
    n_query = args.n_query
    alpha = args.alpha
    test_domain = args.test_domain
    test_iter = args.test_iter
    repeat_exp = args.repeat_exp
    ckpt_name = None if args.ckpt_name in ['No', 'N', 'n'] else args.ckpt_name
    load_task = None if args.load_task in ['No', 'N', 'n'] else args.load_task

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

    train_support = tf.placeholder(tf.float32, shape=[n_way, n_shot-1, img_h, img_w, 3])  
    train_query = tf.placeholder(tf.float32, shape=[n_way, 1, img_h, img_w, 3])
    test_support = tf.placeholder(tf.float32, shape=[n_way, n_shot, img_h, img_w, 3])  
    test_query = tf.placeholder(tf.float32, shape=[n_way, n_query, img_h, img_w, 3])
    is_training = tf.placeholder(tf.bool)

    labels_input = tf.placeholder(tf.int64, shape=[None])

    # Meta-RelationNet
    print("\n=== Build model...")
    print("First learning rate: ", alpha)

    model = RelationNet(alpha=alpha, beta=0, gamma=0, decay=0, backbone='resnet', is_training=False)
    
    model.build_maml_test_train(n_way, n_shot, 1,
                                train_support=train_support, train_query=train_query, 
                                labels=labels_input, first_lr=alpha)
    model.build_maml_test_inference(n_way, n_shot, n_query,
                                    test_support=test_support, test_query=test_query,
                                    labels=labels_input)
    model_summary()

    # saver for saving session
    regular_saver = tf.train.Saver()

    wei_path = '/data/wei/model/cross-domain-few-shot/logs'
    log_path = os.path.join(wei_path, args.log_path, args.test_name, 'best_performance')
    #log_path = os.path.join(wei_path, args.log_path, args.test_name)
    if ckpt_name == None:
        checkpoint = tf.train.latest_checkpoint(log_path)
        ckpt_name = os.path.split(checkpoint)[-1]
    else:
        checkpoint = os.path.join(log_path, ckpt_name)
    print("Restore session from checkpoint: {}".format(checkpoint))

    pkl_path = os.path.join(wei_path, args.log_path, args.test_name, 'train_pkl')

    init = [
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
    ]

    def restore_from_checkpoint(sess, saver, checkpoint):
        if checkpoint:
            saver.restore(sess, checkpoint)
            return True
        else:
            print("Checkpoint not found: {}".format(checkpoint))
            return False

    print("\n=== Load data...")
    # make queue
    test_queue = queue.Queue(maxsize = 10)

    # Init test domain class
    fn = getattr(src.load_data, test_domain)
    test_obj = fn()
    
    # pass obj to preprocess thread and start
    pre_thread = Preprocessor(0, test_obj, test_queue, n_way, n_shot, n_query)
    pre_thread.start()

    print("\n=== Start testing...")

    ## testing
    with tf.Session() as sess:

        best_acc = 0
        for e in range(repeat_exp):
            print("\n=== Test: [%d] ===" % e)

            preprocess_time = 0
            test_time = 0
            total_loss = []
            total_acc = []

            iter_pbar = tqdm(range(test_iter))
            for i_iter in iter_pbar:

                sess.run(init)
                #print("\n=== Check pretrained model...")
                if checkpoint:
                    restore_from_checkpoint(sess, regular_saver, checkpoint)
                else:
                    print("Checkpoint not found!")
                    return

                # get test data
                if load_task == None:
                    start = timeit.default_timer()
                    curr_support, curr_query = test_queue.get()
                    stop = timeit.default_timer()
                    preprocess_time += (stop - start) 
                else:
                    with open(os.path.join(pkl_path, ckpt_name, load_task), 'rb') as f:
                        curr_support, curr_query = pickle.load(f)
                        f.close()

                # get label data
                train_labels = np.repeat(np.arange(n_way), repeats=1).astype(np.uint8)
                test_labels = np.repeat(np.arange(n_way), repeats=n_query).astype(np.uint8)

                for i in range(100):

                    rdm_idx = [random.randint(0, n_shot-1) for _ in range(n_way)]
                    curr_train_query = np.asarray([curr_support[i, idx, :, :, :] for i, idx in enumerate(rdm_idx)])
                    curr_train_query = np.expand_dims(curr_train_query, axis=1)
                    curr_train_support = np.stack([np.delete(curr_support[i, :, :, :, :], idx, axis=0) for i, idx in enumerate(rdm_idx)])

                    _, maml_train_loss, maml_train_acc = \
                        sess.run([model.meta_op, model.maml_train_loss, model.maml_train_acc], 
                        feed_dict={
                            train_support: curr_train_support,
                            train_query: curr_train_query,          
                            labels_input: train_labels,
                            is_training: True
                        })
                    #print("Iter: [%d] Train: [%f, %f]" % (i, maml_train_loss, maml_train_acc))

                start = timeit.default_timer()
                # testing
                maml_test_loss, maml_test_acc = \
                    sess.run([model.maml_test_loss, model.maml_test_acc], 
                    feed_dict={
                        test_support: curr_support,
                        test_query: curr_query,                        
                        labels_input: test_labels,
                        is_training: False
                    })
                stop = timeit.default_timer()

                test_time += (stop - start) 

                total_loss.append(maml_test_loss)
                total_acc.append(maml_test_acc)

            acc = mean(total_acc)
            loss = mean(total_loss)
            confidence = 0.95
            std_err = sem(total_acc)
            h = std_err * t.ppf((1 + confidence) / 2, test_iter - 1)

            print('Test: [%f, %f +- %f]' % (loss, acc, h))
            print("==> Preporcess time: ", preprocess_time/(i_iter+1))
            print("==> Test time: ", test_time/(i_iter+1))
            
            if best_acc < acc:
                best_acc = acc
                best_h = h
                #if load_task == None:
                #    dump_train_task(pkl_path, ckpt_name, best_acc, (curr_train_support, curr_train_query))
            
        print("Best performance: [%f +- %f]" % (best_acc, best_h))
        pre_thread.stop()
        pre_thread.join()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
