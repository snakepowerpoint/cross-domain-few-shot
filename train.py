# deep learning framework
import tensorflow as tf
import tensorflow.contrib.slim as slim

# computation
import numpy as np
import random

# io
import os
import argparse

# customerized 
from src.load_data import Pacs, Cub
from src.model import PrototypeNet

# miscellaneous
import gc


parser = argparse.ArgumentParser()
parser.add_argument('--log_path', default='baseline', type=str)
parser.add_argument('--test_name', default='test', type=str)
parser.add_argument('--n_way', default=5, type=int)
parser.add_argument('--n_shot', default=5, type=int)
parser.add_argument('--n_query', default=15, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--n_iter', default=20000, type=int)
parser.add_argument('--multi_domain', default=True, type=bool)


def compute_acc(prediction, one_hot_labels): 
    labels = tf.argmax(one_hot_labels, axis=1)
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(prediction, axis=-1), labels)))
    return acc

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def main(args):
    img_w = 227
    img_h = 227
    
    n_way = args.n_way
    n_shot = args.n_shot
    n_query = args.n_query
    
    dataset = Pacs()

    ## establish training graph
    # inputs placeholder (support and query randomly sampled from two domain)
    support_a = tf.placeholder(tf.float32, shape=[n_way, n_shot, None, None, 3])  
    query_b = tf.placeholder(tf.float32, shape=[n_way, n_query, None, None, 3])
    query_b_y = tf.placeholder(tf.int64, [None, None])  # e.g. [5, 15]
    
    is_training = tf.placeholder(tf.bool)

    # reshape
    support_a_reshape = tf.reshape(support_a, [n_way * n_shot, img_h, img_w, 3])
    query_b_reshape = tf.reshape(query_b, [n_way * n_query, img_h, img_w, 3])

    # feature extractor
    protonet = PrototypeNet()

    support_a_feature = protonet.feature_extractor(support_a_reshape, is_training)
    query_b_feature = protonet.feature_extractor(query_b_reshape, is_training, reuse=True)

    # get prototype
    prototype_a = protonet.get_prototype(support_a_feature, n_shot=n_shot)

    # metric function (few-shot classification)
    dists = protonet.compute_distance(prototype_a, query_b_feature)
    log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [n_way, n_query, -1])

    # classification loss and accuracy
    query_b_y_one_hot = tf.one_hot(query_b_y, depth=n_way)
    ce_loss = -tf.reduce_mean(
        tf.reshape(tf.reduce_sum(tf.multiply(query_b_y_one_hot, log_p_y), axis=-1), [-1]))
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), query_b_y)))

    # optimizer
    global_step = tf.Variable(0, trainable=False, name='global_step')
    rate = tf.train.exponential_decay(args.lr, global_step, 2000, 0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(rate)
    train_op = optimizer.minimize(ce_loss, global_step=global_step)
    
    model_summary()

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
    cub = Cub(img_size=227)
    cub_support, cub_query = cub.get_task()

    ## training
    with tf.Session() as sess:
        # Creates a file writer for the log directory.
        file_writer_train = tf.summary.FileWriter(os.path.join(log_path, "train"), sess.graph)
        file_writer_test = tf.summary.FileWriter(os.path.join(log_path, "test"), sess.graph)

        # store variables
        tf.summary.image("Support a image", support_a_reshape[:4], max_outputs=4)
        tf.summary.image("Query b image", query_b_reshape[:4], max_outputs=4)
        tf.summary.scalar("Classification loss", ce_loss)
        tf.summary.scalar("Accuracy", acc)
        merged = tf.summary.merge_all()

        # get test unseen domains
        domains = list(dataset.data_dict.keys())
        test_domain = domains[2]
        domains.pop(2)

        domain_a = domains[0]
        domain_b = domains[1]
        categories = list(dataset.data_dict[domain_a].keys())

        sess.run(init)
        restore_from_checkpoint(sess, saver, lastest_checkpoint)
        for i_iter in range(args.n_iter):
            # get domain A and B
            if args.multi_domain:
                domain_a, domain_b = random.sample(domains, k=2)

            # get categories
            selected_categories = random.sample(categories, k=n_way)

            # get support and query from domain A and B
            support, _ = dataset.get_task(domain_a, selected_categories, n_shot, n_query)
            _, query = dataset.get_task(domain_b, selected_categories, n_shot, n_query)
            
            labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)

            # training                 
            _, step = sess.run([train_op, global_step], feed_dict={
                support_a: support,
                query_b: query,
                query_b_y: labels,
                is_training: True
            })

            # evaluation
            if i_iter % 100 == 0:
                # get training loss and accuracy
                summary_train, train_loss, train_acc = sess.run([merged, ce_loss, acc], feed_dict={
                    support_a: support,
                    query_b: query,
                    query_b_y: labels,
                    is_training: False
                })

                # get task from unseen domain 
                support, query = dataset.get_task(test_domain, selected_categories, n_shot, n_query)
                
                summary_test, test_loss, test_acc = sess.run([merged, ce_loss, acc], feed_dict={
                    support_a: support,
                    query_b: query,
                    query_b_y: labels,
                    is_training: False
                })

                cub_loss, cub_acc = sess.run([ce_loss, acc], feed_dict={
                    support_a: cub_support,
                    query_b: cub_query,
                    query_b_y: labels,
                    is_training: False
                })

                # log all variables
                file_writer_train.add_summary(summary_train, global_step=i_iter)
                file_writer_test.add_summary(summary_test, global_step=i_iter)

                print('Iteration: %d, train cost: %g, test cost: %g, train acc: %g, test acc: %g' %
                      (step, train_loss, test_loss, train_acc, test_acc))
                print('cub cost: %g, cub acc: %g' % (cub_loss, cub_acc))

            # save session every 2000 iteration
            if step % 2000 == 0:    
                saver.save(sess, checkpoint_file, global_step=step)
                print('Save session at step %d' % step)

        file_writer_train.close()
        file_writer_test.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
