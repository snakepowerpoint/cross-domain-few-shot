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
from src.model import PrototypeNet, RelationNet

# miscellaneous
import gc


parser = argparse.ArgumentParser()
parser.add_argument('--log_path', default='baseline', type=str)
parser.add_argument('--test_name', default='test', type=str)
parser.add_argument('--n_way', default=5, type=int)
parser.add_argument('--n_shot', default=5, type=int)
parser.add_argument('--n_query', default=16, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--n_iter', default=40000, type=int)
parser.add_argument('--multi_domain', default=True, type=bool)


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def main(args):
    img_w = 224  # 84 or 224
    img_h = 224
    
    n_way = args.n_way
    n_shot = args.n_shot
    n_query = args.n_query
    
    dataset = MiniImageNet()
    
    ## establish training graph
    support_a = tf.placeholder(tf.float32, shape=[n_way, n_shot, None, None, 3])  
    query_b = tf.placeholder(tf.float32, shape=[n_way, n_query, None, None, 3])
    is_training = tf.placeholder(tf.bool)

    # reshape
    support_a_reshape = tf.reshape(support_a, [n_way * n_shot, img_h, img_w, 3])
    query_b_reshape = tf.reshape(query_b, [n_way * n_query, img_h, img_w, 3])

    # feature extractor
    init_lr = args.lr
    model = RelationNet(n_way, n_shot, n_query, backbone='resnet',
                        learning_rate=init_lr, is_training=is_training)
    train_op, train_loss, train_acc, global_step = model.train(
        support=support_a_reshape, query=query_b_reshape)
    
    model_summary()

    ## establish test graph
    model_test = RelationNet(n_way, n_shot, n_query, backbone='resnet',
                             learning_rate=init_lr, is_training=is_training)
    test_loss, test_acc = model_test.test(support=support_a_reshape, query=query_b_reshape)

    # saver for saving session
    saver = tf.train.Saver()
    log_path = os.path.join('logs', args.log_path,
                            '_'.join((args.test_name, 'lr'+str(args.lr))))
    
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

    # load CUB data
    cub = Cub(size=(224, 224), mode='test')
    
    ## training
    with tf.Session() as sess:
        # Creates a file writer for the log directory.
        file_writer_train = tf.summary.FileWriter(os.path.join(log_path, "train"), sess.graph)
        file_writer_val = tf.summary.FileWriter(os.path.join(log_path, "val"), sess.graph)
        file_writer_test = tf.summary.FileWriter(os.path.join(log_path, "test"), sess.graph)

        # store variables
        tf.summary.image("Support a image", support_a_reshape[:4], max_outputs=4)
        tf.summary.image("Query b image", query_b_reshape[:4], max_outputs=4)
        tf.summary.scalar("Classification loss", train_loss)
        tf.summary.scalar("Accuracy", train_acc)
        merged = tf.summary.merge_all()
        
        sess.run(init)
        restore_from_checkpoint(sess, saver, lastest_checkpoint)
        preprocess_time = 0
        train_time = 0
        for i_iter in range(args.n_iter):

            start = timeit.default_timer()
            # get support and query
            support, query = dataset.get_task(n_way, n_shot, n_query, mode='train')
            stop = timeit.default_timer()
            
            preprocess_time += (stop - start) 

            start = timeit.default_timer()
            # training                 
            _, step = sess.run([train_op, global_step], feed_dict={
                support_a: support,
                query_b: query,
                is_training: True
            })
            stop = timeit.default_timer()

            train_time += (stop - start) 
            
            # evaluation
            if i_iter % 50 == 0:
                summary_train, loss, acc = sess.run([merged, test_loss, test_acc], feed_dict={
                    support_a: support,
                    query_b: query,
                    is_training: False
                })

                # get task from validation
                support, query = dataset.get_task(n_way, n_shot, n_query, aug=False, mode='test')
                summary_val, val_loss, val_acc = sess.run([merged, test_loss, test_acc], feed_dict={
                    support_a: support,
                    query_b: query,
                    is_training: False
                })

                # get task from unseen domain (CUB)
                cub_support, cub_query = cub.get_task(n_way, n_shot, n_query)
                summary_test, cub_loss, cub_acc = sess.run([merged, test_loss, test_acc], feed_dict={
                    support_a: cub_support,
                    query_b: cub_query,
                    is_training: False
                })

                # log all variables
                file_writer_train.add_summary(summary_train, global_step=step)
                file_writer_val.add_summary(summary_val, global_step=step)
                file_writer_test.add_summary(summary_test, global_step=step)

                print('Iteration: %d, train cost: %g, train acc: %g' % (step, loss, acc))
                print('eval cost: %g, eval acc: %g' % (val_loss, val_acc))
                print('CUB cost: %g, CUB acc: %g' % (cub_loss, cub_acc))
                print("==> Preporcess time: ", preprocess_time/(i_iter+1))
                print("==> Train time: ", train_time/(i_iter+1))

            # save session every 2000 iteration
            if step % 2000 == 0:    
                saver.save(sess, checkpoint_file, global_step=step)
                print('Save session at step %d' % step)

        # compute accuracy on 600 randomly generated task from CUB
        acc = []
        for _ in range(600):
            cub_support, cub_query = cub.get_task(n_way, n_shot, n_query)
            cub_acc = sess.run([test_acc], feed_dict={
                support_a: cub_support,
                query_b: cub_query,
                is_training: False
            })
            acc.append(cub_acc)
        acc = np.array(acc)

        confidence = 0.95
        std_err = sem(acc)
        h = std_err * t.ppf((1 + confidence) / 2, 600 - 1)
        print('Overall acc: {}+-{}%'.format(mean(acc), h))

        file_writer_train.close()
        file_writer_val.close()
        file_writer_test.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
