# deep learning framework
import tensorflow as tf
import tensorflow.contrib.slim as slim
import torch

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
from src.load_data import Pacs, Cub, Omniglot, MiniImageNet, MiniImageNetFull
from src.model import RelationNet
from src.model_utils import load_weights

# miscellaneous
import gc

# multi-thread
import threading
import queue

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', default='baseline', type=str)
parser.add_argument('--test_name', default='test', type=str)
parser.add_argument('--n_way', default=5, type=int)
parser.add_argument('--n_shot', default=5, type=int)
parser.add_argument('--n_query', default=16, type=int)
parser.add_argument('--n_query_test', default=15, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--n_iter', default=40000, type=int)
parser.add_argument('--n_episode', default=100, type=int)
parser.add_argument('--resume_epoch', default=0, type=int)
parser.add_argument('--test_iter', default=200, type=int)
parser.add_argument('--multi_domain', default=True, type=bool)
parser.add_argument('--full_size', default=False, type=bool)
parser.add_argument('--fine_tune', default=False, type=bool)
# parser.add_argument('--pretrain_path', default='logs/pretrain_baseline/backup_full_b16nob_init_960k', type=str)
parser.add_argument('--pretrain_path', default='logs/pretrain_baseline/pretrain_baseline_batch_en64_lr0.001_decay0.96', type=str)
parser.add_argument('--source', default='tensorflow', type=str)


class Preprocessor(threading.Thread):
    def __init__(self, id, mini_obj, mini_queue, n_way, n_shot, n_query, is_full_size=False):
        # for thread
        self.thread_id = id
        self._stop_event = threading.Event()

        # for few-shot param.
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        # for obj
        self.is_full_size = is_full_size
        if self.is_full_size:
            self.mini_obj = mini_obj.task_generator_raw(
                n_way=self.n_way, n_shot=self.n_shot, n_query=self.n_query, aug=True, mode='train')
        else:
            self.mini_obj = mini_obj
        self.mini_queue = mini_queue        
        
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
        print("full_size = {}".format(self.is_full_size))

        while True:
            # for mini
            if self.is_full_size:
                # mini_sup, mini_qu = self.mini_obj.get_task_from_raw(n_way=self.n_way, n_shot=self.n_shot, n_query=self.n_query, aug=True, mode='train')
                mini_sup, mini_qu = next(self.mini_obj)
            else:
                mini_sup, mini_qu = self.mini_obj.get_task(n_way=self.n_way, n_shot=self.n_shot, n_query=self.n_query, aug=True, mode='train')
            
            self.mini_queue.put((mini_sup, mini_qu))

            if self.is_stopped():
                break

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

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
        var_to_restore = [val for val in var if (('res10_weights' in val.name) or (
            'bn' in val.name)) and ('meta_op' not in val.name) and ('M_' not in val.name)]

        print("Pretrained weights:")
        for _var in var_to_restore:
            print(_var)

        saver = tf.train.Saver(var_to_restore)
        saver.restore(sess, checkpoint)
        return True
    else:
        print("Pretrained checkpoint not found: {}".format(checkpoint))
        return False

def load_pytorch_weights(sess, weight_path):
    if weight_path:
        print("Load pytorch weights from: {}".format(weight_path))
        tmp = torch.load(weight_path)
        pytorch_weights = {}
        for key, value in tmp['state'].items():
            weights = torch.Tensor.cpu(value)
            pytorch_weights[key] = weights.data.numpy()

        load_weights(sess, pytorch_weights)
        print("===Done===")
    else:
        print('Pytorch weights are not found')

def main(args):
    img_w = 224  # 84 or 224
    img_h = 224
    
    n_way = args.n_way
    n_shot = args.n_shot
    n_query = args.n_query
    n_query_test = args.n_query_test
    test_iter = args.test_iter
    n_episode = args.n_episode
    resume_epoch = args.resume_epoch
    
    full_size = args.full_size    
    # fine_tune = args.fine_tune
    pretrain_path = args.pretrain_path
    source = args.source

    ## establish training graph
    support_a = tf.placeholder(tf.float32, shape=[n_way, n_shot, None, None, 3])  
    query_b = tf.placeholder(tf.float32, shape=[n_way, None, None, None, 3])
    labels_input = tf.placeholder(tf.int64, shape=[None])
    n_query_input = tf.placeholder_with_default(n_query, shape=None)
    is_training = tf.placeholder(tf.bool)

    # reshape
    support_a_reshape = tf.reshape(support_a, [n_way * n_shot, img_h, img_w, 3])
    query_b_reshape = tf.reshape(query_b, [n_way * n_query_input, img_h, img_w, 3])
    
    # feature extractor
    init_lr = args.lr
    train_labels = np.repeat(np.arange(n_way), repeats=n_query).astype(np.uint8)
    test_labels = np.repeat(np.arange(n_way), repeats=n_query_test).astype(np.uint8)

    model = RelationNet(gamma=init_lr, backbone='resnet', is_training=is_training)
    model.build(n_way, n_shot, n_query=n_query_input, support_x=support_a_reshape, 
                query_x=query_b_reshape, labels=labels_input, regularized=False)
    
    model_summary()

    ## saver for saving session
    regular_saver = tf.train.Saver()
    best_saver = tf.train.Saver(max_to_keep=None)

    ## log and checkpoint
    # wei_path = '/data/wei/model/cross-domain-few-shot/logs'
    rahul_path = 'logs'
    log_path = os.path.join(rahul_path, args.log_path, '_'.join((args.test_name, 'lr'+str(args.lr))))
    lastest_checkpoint = tf.train.latest_checkpoint(log_path)
    
    best_checkpoint_path = os.path.join(log_path, 'best_performance')

    ## pre-train weights
    pytorch_weight_path = '/data/rahul/workspace/cross-domain-few-shot/cross-domain-few-shot/model/baseline/399.tar'  # or best_model.tar
    
    ## load data
    # load Mini-ImageNet
    if full_size:
        dataset = MiniImageNetFull()
    else:
        dataset = MiniImageNet()

    # load CUB data
    cub = Cub()

    # make queue
    mini_queue = queue.Queue(maxsize = 10)

    # pass obj to preprocess thread and start
    pre_thread = Preprocessor(0, dataset, mini_queue, n_way, n_shot, n_query, is_full_size=full_size)
    pre_thread.start()

    init = [
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ]
    ## training
    with tf.Session() as sess:
        # Creates a file writer for the log directory.
        file_writer_train = tf.summary.FileWriter(os.path.join(log_path, "train"), sess.graph)
        file_writer_val = tf.summary.FileWriter(os.path.join(log_path, "val"), sess.graph)
        # file_writer_test = tf.summary.FileWriter(os.path.join(log_path, "test"), sess.graph)

        # store variables
        s_imgs = tf.summary.image("Support a image", support_a_reshape[:2], max_outputs=2)
        q_imgs = tf.summary.image("Query b image", query_b_reshape[:2], max_outputs=2) 
        x_loss = tf.summary.scalar("Classification loss", model.x_loss)
        x_acc = tf.summary.scalar("Accuracy", model.x_acc)
        
        merged = tf.summary.merge([s_imgs, q_imgs, x_loss, x_acc])
        
        sess.run(init)
        print("=== Start training...")
        print("=== Ckeck pre-trained backbone model...")
        if source == 'tensorflow':
            pretrain_ckeckpoint = tf.train.latest_checkpoint(pretrain_path)
            if pretrain_ckeckpoint:
                restore_from_pretrain(sess, pretrain_ckeckpoint)
            else:
                print("Can not find the tensorfow pre-trained weights!")
        elif source == 'pytorch':
            if pytorch_weight_path:
                load_pytorch_weights(sess, pytorch_weight_path)
            else:
                print("Can not find the pytorch pre-trained weights!")
        else:
            print("Can not find the pre-trained backbone weights.")
        
        print("=== Ckeck last checkpoint model...")
        if lastest_checkpoint:
            restore_from_checkpoint(sess, regular_saver, lastest_checkpoint)
        else:
            print("Train from {} step".format(resume_epoch))
        
        preprocess_time = 0
        train_time = 0

        best_val_acc = 0
        final_acc = np.empty((n_episode)) # np.empty((test_iter, 2))
        final_loss = np.empty((n_episode))
        for i_iter in range(resume_epoch, args.n_iter):

            start = timeit.default_timer()
            # get support and query
            support, query = mini_queue.get()
            stop = timeit.default_timer()
            
            preprocess_time += (stop - start) 

            start = timeit.default_timer()
            # training                 
            sess.run([model.train_op], feed_dict={
                support_a: support,
                query_b: query,
                labels_input: train_labels,
                n_query_input: n_query,
                is_training: True
            })
            stop = timeit.default_timer()

            train_time += (stop - start) 
            
            # evaluation
            if (i_iter + 1) % 100 == 0:
                summary_train, loss, acc = sess.run([merged, model.x_loss, model.x_acc], feed_dict={
                    support_a: support,
                    query_b: query,
                    labels_input: train_labels, 
                    n_query_input: n_query,
                    is_training: False
                })

                # get task from validation
                if full_size:
                    support, query = dataset.get_task_from_raw(n_way, n_shot, n_query, aug=False, mode='val')
                else:
                    support, query = dataset.get_task(n_way, n_shot, n_query, aug=False, mode='val')
                summary_val, val_loss, val_acc = sess.run([merged, model.x_loss, model.x_acc], feed_dict={
                    support_a: support,
                    query_b: query,
                    labels_input: train_labels, 
                    n_query_input: n_query,
                    is_training: False
                })

                # log all variables
                file_writer_train.add_summary(summary_train, global_step=i_iter)
                file_writer_val.add_summary(summary_val, global_step=i_iter)
                # file_writer_test.add_summary(summary_test, global_step=i_iter)

                print('Iteration: %d' % (i_iter+1))
                print('train cost: %g, train acc: %g' % (loss, acc))
                print('eval cost: %g, eval acc: %g' % (val_loss, val_acc))
                print("==> Preporcess time: ", preprocess_time/(i_iter+1))
                print("==> Train time: ", train_time/(i_iter+1))
                
                # evaluate on n_episode tasks
                iter_pbar = tqdm(range(n_episode))
                for i in iter_pbar:
                    if full_size:
                        support, query = dataset.get_task_from_raw(n_way, n_shot, n_query, aug=False, mode='val')
                    else:
                        support, query = dataset.get_task(n_way, n_shot, n_query, aug=False, mode='val')
                    val_loss, val_acc = sess.run([model.x_loss, model.x_acc], feed_dict={
                        support_a: support,
                        query_b: query,
                        labels_input: train_labels,
                        n_query_input: n_query,
                        is_training: False
                    })

                    final_loss[i] = val_loss
                    final_acc[i] = val_acc
                    
                mean_loss = mean(final_loss, axis=0)
                mean_acc = mean(final_acc, axis=0)
                confidence = 0.95
                std_err = sem(final_acc)
                h = std_err * t.ppf((1 + confidence) / 2, n_episode - 1)
                        
                print('{} mini-ImageNet acc: {}+-{} %'.format(n_episode, mean_acc, h))
                    
                # save best checkpoint
                if mean_acc > best_val_acc:
                    best_val_acc = mean_acc
                    ckpt_name = 'best'
                    best_saver.save(sess, os.path.join(best_checkpoint_path , ckpt_name))
                    print("=== Save best model! ===")
                    print("Best val loss: {}".format(mean_loss))
                    print("Best val acc: {}".format(mean_acc))
                                    
            # save session every several iterations
            if (i_iter+1) % 2500 == 0:    
                best_val_acc = mean_acc
                ckpt_name = "val_acc_" + str(mean_acc) + "_loss_" + str(np.round(mean_loss, 4)) + ".ckpt"
                regular_saver.save(sess, os.path.join(log_path, ckpt_name), global_step=(i_iter+1))
                print('Save session at step %d' % (i_iter+1))
        
        pre_thread.stop()
        pre_thread.join()

        file_writer_train.close()
        file_writer_val.close()
        # file_writer_test.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
