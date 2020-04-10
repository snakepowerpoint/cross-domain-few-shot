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
from src.model import PrototypeNet, RelationNet ### use model_local.py

# miscellaneous
import gc


parser = argparse.ArgumentParser()
parser.add_argument('--log_path', default='pretrain_baseline', type=str)
parser.add_argument('--test_name', default='pretrain_mini', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--decay', default=0.96, type=float)
parser.add_argument('--n_iter', default=240000, type=int)
parser.add_argument('--num_class', default=200, type=int)
parser.add_argument('--start_iter', default=0, type=int)

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def main(args):
    img_w = 224
    img_h = 224
    
    lr = args.lr
    decay = args.decay
    batch_size = args.batch_size
    num_class = args.num_class
    start_iter = args.start_iter
       
    ## establish training graph
    # inputs placeholder (support and query randomly sampled from two domain)
    inputs = tf.placeholder(tf.float32, shape=[batch_size, img_w, img_h, 3])  
    labels = tf.placeholder(tf.float32, shape=[batch_size, num_class])
    learning_rate = tf.placeholder(tf.float32)  
    is_training = tf.placeholder(tf.bool)

    # Meta-RelationNet
    print("=== Build model...")
    print("learning rate: ", lr)
    print("Decay second learning rate: ", decay)
    model = RelationNet(0, 0, 0, alpha=0, gamma=lr, decay=decay,
                        backbone='resnet', is_training=is_training)
    model.train_baseline(inputs=inputs, labels=labels, label_dim=num_class, learning_rate=learning_rate)

    model_summary()

    # saver for saving session
    saver = tf.train.Saver()
    log_path = os.path.join('logs', args.log_path, '_'.join(
        (args.test_name, 'lr'+str(lr), 'decay'+str(decay))))
    
    checkpoint_file = log_path + "/checkpoint.ckpt"
    lastest_checkpoint = tf.train.latest_checkpoint(log_path)
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


    print("=== Load data...")
    # load mini-imagenet
    mini = MiniImageNetFull()
    
    ## training
    with tf.Session() as sess:

        # Creates a file writer for the log directory.
        file_writer_train = tf.summary.FileWriter(os.path.join(log_path, "train"), sess.graph)
        #file_writer_test = tf.summary.FileWriter(os.path.join(log_path, "test"), sess.graph)

        # store variables
        tf.summary.image("input", inputs[:4], max_outputs=4, collections=['train', 'test'])
        tf.summary.scalar("loss", model.loss, collections=['train', 'test'])
        tf.summary.scalar("acc", model.acc, collections=['train', 'test'])
        
        train_merged = tf.summary.merge_all('train')
        #test_merged = tf.summary.merge_all('test')
        
        print("=== Start training...")
        sess.run(init)
        restore_from_checkpoint(sess, saver, lastest_checkpoint)
        mini_datagen = mini.batch_generator(label_dim=num_class, batch_size=batch_size)
        for i_iter in range(start_iter, args.n_iter):

            # mini-imagenet ======================================================================
            curr_inputs, curr_labels = next(mini_datagen)

            # learning rate decay
            #if i_iter == 40000:
            #    lr = lr * 0.5

            # training                 
            sess.run([model.train_op], feed_dict={
                inputs: curr_inputs,
                labels: curr_labels,
                learning_rate: lr,
                is_training: True
            })

            # evaluation
            if i_iter % 50 == 0:
                # evaluation
                summary_train, train_loss, train_acc = \
                    sess.run([train_merged, model.loss, model.acc], 
                        feed_dict={
                            inputs: curr_inputs,
                            labels: curr_labels,
                            learning_rate: lr,
                            is_training: False
                        })
                """
                # get task from val
                val_inputs, val_labels = mini.get_batch(mode='val', aug=False)
                summary_test, test_loss, test_acc = \
                    sess.run([test_merged, model.loss, model.acc], 
                        feed_dict={
                            inputs: val_inputs,
                            labels: val_labels,
                            learning_rate: lr,
                            is_training: False
                        })
                """
                # log all variables
                file_writer_train.add_summary(summary_train, global_step=i_iter)
                #file_writer_test.add_summary(summary_test, global_step=i_iter)

                print('Iteration: %d, Train [%f, %f]' % (i_iter+1, train_loss, train_acc))
                #print('==> Test: [%f, %f]' % (test_loss, test_acc))

            # save session every 2000 iteration
            if (i_iter+1) % 2000 == 0:    
                saver.save(sess, checkpoint_file, global_step=(i_iter+1))
                print('Save session at step %d' % (i_iter+1))

        file_writer_train.close()
        #file_writer_test.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
