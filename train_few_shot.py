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
from src.load_data import Pacs, Cub, Omniglot, MiniImageNet, MiniImageNetFull
from src.model import PrototypeNet, RelationNet

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
parser.add_argument('--test_iter', default=200, type=int)
parser.add_argument('--multi_domain', default=True, type=bool)
parser.add_argument('--pretrain', default=True, type=bool)
parser.add_argument('--fine_tune', default=False, type=bool)
parser.add_argument('--resume_epoch', default=0, type=int)


class Preprocessor(threading.Thread):
    def __init__(self, id, mini_obj, mini_queue, n_way, n_shot, n_query):
        # for thread
        self.thread_id = id
        self._stop_event = threading.Event()

        # for obj
        self.mini_obj = mini_obj
        self.mini_queue = mini_queue

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
            mini_sup, mini_qu = self.mini_obj.get_task_from_raw(
                n_way=self.n_way, n_shot=self.n_shot, n_query=self.n_query, aug=True, mode='train')
            self.mini_queue.put((mini_sup, mini_qu))

            if self.is_stopped():
                break

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def main(args):
    img_w = 224  # 84 or 224
    img_h = 224
    
    n_way = args.n_way
    n_shot = args.n_shot
    n_query = args.n_query
    n_query_test = args.n_query_test
    
    pretrain = args.pretrain
    resume_epoch = args.resume_epoch
    fine_tune = args.fine_tune
    test_iter = args.test_iter
    final_acc = np.empty((test_iter, 2))

    dataset = MiniImageNetFull()
    
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

    model = RelationNet(n_way, n_shot, n_query, n_query_test, gamma=init_lr, 
                        backbone='resnet', is_training=is_training)
    model.build(n_way, n_shot, n_query=n_query_input, support_x=support_a_reshape, 
                query_x=query_b_reshape, labels=labels_input, regularized=False)
    
    model_summary()

    # saver for saving session
    saver = tf.train.Saver(max_to_keep=10)
    best_saver = tf.train.Saver(max_to_keep=10)
    log_path = os.path.join('logs', args.log_path,
                            '_'.join((args.test_name, 'lr'+str(args.lr))))
    
    if fine_tune:
        checkpoint_file = log_path + "/checkpoint_fine_tune.ckpt"
    else:
        checkpoint_file = log_path + "/checkpoint.ckpt"
    lastest_checkpoint = tf.train.latest_checkpoint(log_path)
    
    pretrain_log_path = os.path.join(
        'logs', 'pretrain_baseline', 'backup_full_b16nob_960k')
    pretrain_ckeckpoint = tf.train.latest_checkpoint(pretrain_log_path)
    #pretrain_ckeckpoint = os.path.join(pretrain_log_path, 'checkpoint.ckpt-24000')

    best_checkpoint_path = os.path.join(log_path, 'best_performance')

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
            var_to_restore = [val for val in var if (('res10_weights' in val.name) or ('bn' in val.name)) and ('meta_op' not in val.name) and ('M_' not in val.name)]

            print("Pretrained weights:")
            for _var in var_to_restore:
                print(_var)

            saver = tf.train.Saver(var_to_restore)
            saver.restore(sess, checkpoint)
            return True
        else:
            print("Pretrained checkpoint not found: {}".format(checkpoint))
            return False

    # load CUB data
    cub = Cub(mode='test')

    # make queue
    mini_queue = queue.Queue(maxsize = 10)

    # pass obj to preprocess thread and start
    pre_thread = Preprocessor(0, dataset, mini_queue, n_way, n_shot, n_query)
    pre_thread.start()

    ## training
    with tf.Session() as sess:
        # Creates a file writer for the log directory.
        file_writer_train = tf.summary.FileWriter(os.path.join(log_path, "train"), sess.graph)
        file_writer_val = tf.summary.FileWriter(os.path.join(log_path, "val"), sess.graph)
        file_writer_test = tf.summary.FileWriter(os.path.join(log_path, "test"), sess.graph)

        # store variables
        s_imgs = tf.summary.image("Support a image", support_a_reshape[:4], max_outputs=4)
        q_imgs = tf.summary.image("Query b image", query_b_reshape[:4], max_outputs=4) 
        x_loss = tf.summary.scalar("Classification loss", model.x_loss)
        x_acc = tf.summary.scalar("Accuracy", model.x_acc)
        
        merged = tf.summary.merge([s_imgs, q_imgs, x_loss, x_acc])
        
        sess.run(init)
        print("=== Start training...")
                
        if lastest_checkpoint:
            restore_from_checkpoint(sess, saver, lastest_checkpoint)
        elif pretrain:
            print("=== Ckeck pretrained model...")
            print("=== pretrained model path: {}".format(pretrain_log_path))
            print(pretrain_ckeckpoint)            
            restore_from_pretrain(sess, pretrain_ckeckpoint)
        
        preprocess_time = 0
        train_time = 0
        best_cub_acc = 0
        for i_iter in range(resume_epoch, args.n_iter):

            start = timeit.default_timer()
            # get support and query
            # support, query = dataset.get_task(n_way, n_shot, n_query, mode='train')
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
            if i_iter % 100 == 0:
                summary_train, loss, acc = sess.run([merged, model.x_loss, model.x_acc], feed_dict={
                    support_a: support,
                    query_b: query,
                    labels_input: train_labels, 
                    n_query_input: n_query,
                    is_training: False
                })

                # get task from validation
                support, query = dataset.get_task_from_raw(n_way, n_shot, n_query, aug=False, mode='val')
                summary_val, val_loss, val_acc = sess.run([merged, model.x_loss, model.x_acc], feed_dict={
                    support_a: support,
                    query_b: query,
                    labels_input: train_labels, 
                    n_query_input: n_query,
                    is_training: False
                })

                # get task from unseen domain (CUB)
                cub_support, cub_query = cub.get_task_from_raw(n_way, n_shot, n_query_test, aug=False)
                summary_test, cub_loss, cub_acc = sess.run([merged, model.x_loss, model.x_acc], feed_dict={
                    support_a: cub_support,
                    query_b: cub_query,
                    labels_input: test_labels,
                    n_query_input: n_query_test,
                    is_training: False
                })

                # log all variables
                file_writer_train.add_summary(summary_train, global_step=i_iter)
                file_writer_val.add_summary(summary_val, global_step=i_iter)
                file_writer_test.add_summary(summary_test, global_step=i_iter)

                print('Iteration: %d, train cost: %g, train acc: %g' % (i_iter+1, loss, acc))
                print('eval cost: %g, eval acc: %g' % (val_loss, val_acc))
                print('CUB cost: %g, CUB acc: %g' % (cub_loss, cub_acc))
                print("==> Preporcess time: ", preprocess_time/(i_iter+1))
                print("==> Train time: ", train_time/(i_iter+1))

                if (i_iter > 1500) and (fine_tune == True):
                    # compute accuracy on several randomly generated task from mini-ImageNet and CUB
                    iter_pbar = tqdm(range(test_iter))
                    for i in iter_pbar:
                        # get task from test
                        support, query = dataset.get_task_from_raw(n_way, n_shot, n_query_test, aug=False, mode='test')
                        val_acc = sess.run([model.x_acc], feed_dict={
                            support_a: support,
                            query_b: query,
                            labels_input: test_labels,
                            n_query_input: n_query_test,
                            is_training: False
                        })

                        cub_support, cub_query = cub.get_task_from_raw(n_way, n_shot, n_query_test)
                        cub_acc = sess.run([model.x_acc], feed_dict={
                            support_a: cub_support,
                            query_b: cub_query,
                            labels_input: test_labels,
                            n_query_input: n_query_test,
                            is_training: False
                        })
                        final_acc[i] = np.concatenate([val_acc, cub_acc])
                    
                    mean_acc = mean(final_acc, axis=0)
                    confidence = 0.95
                    std_err = sem(final_acc)
                    h = std_err * t.ppf((1 + confidence) / 2, test_iter - 1)
                        
                    print('Overall mini-ImageNet acc: {}+-{}%'.format(mean_acc[0], h[0]))
                    print('Overall CUB acc: {}+-{}%'.format(mean_acc[1], h[1]))                

                    if mean_acc[1] >= best_cub_acc:
                        best_cub_acc = mean_acc[1]
                        ckpt_name = "CUB_acc" + str(best_cub_acc) + "_h" + str(h[1]) + "_Mini_acc" + str(mean_acc[0]) + "_h" + str(h[0]) + ".ckpt" 
                        best_saver.save(sess, os.path.join(best_checkpoint_path , ckpt_name), global_step=(i_iter+1))
                        print("=== New record! ===")
            
            # save session every 2000 iteration
            if (i_iter+1) % 2000 == 0:    
                saver.save(sess, checkpoint_file, global_step=(i_iter+1))
                print('Save session at step %d' % (i_iter+1))
        
        pre_thread.stop()
        pre_thread.join()

        file_writer_train.close()
        file_writer_val.close()
        file_writer_test.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
