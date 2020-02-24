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
from src.load_data import Pacs
from src.model import PrototypeNet

# miscellaneous
import gc


parser = argparse.ArgumentParser()
parser.add_argument('--log_path', default='logs/test/', type=str)
parser.add_argument('--test_name', default='test', type=str)

parser.add_argument('--n_way', default=5, type=int)
parser.add_argument('--n_shot', default=5, type=int)
parser.add_argument('--n_query', default=15, type=int)

parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--n_iter', default=20000, type=int)


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

    ## establish graph
    # inputs placeholder (get tasks from domain A and B)
    support_a = tf.placeholder(tf.float32, shape=[n_way, n_shot, None, None, 3])  
    query_a = tf.placeholder(tf.float32, shape=[n_way, n_query, None, None, 3])  
    support_b = tf.placeholder(tf.float32, shape=[n_way, n_shot, None, None, 3])
    query_b = tf.placeholder(tf.float32, shape=[n_way, n_query, None, None, 3])
    
    query_b_y = tf.placeholder(tf.int64, [None, None])  # e.g. [5, 15]
    
    # reshape
    support_a_reshape = tf.reshape(support_a, [n_way * n_shot, img_h, img_w, 3])
    query_a_reshape = tf.reshape(query_a, [n_way * n_query, img_h, img_w, 3])
    support_b_reshape = tf.reshape(support_b, [n_way * n_shot, img_h, img_w, 3])
    query_b_reshape = tf.reshape(query_b, [n_way * n_query, img_h, img_w, 3])

    # feature extractor
    protonet = PrototypeNet()

    support_a_feature = protonet.feature_extractor(support_a_reshape)
    query_a_feature = protonet.feature_extractor(query_a_reshape, reuse=True)
    support_b_feature = protonet.feature_extractor(support_b_reshape, reuse=True)
    query_b_feature = protonet.feature_extractor(query_b_reshape, reuse=True)

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
    rate = tf.train.inverse_time_decay(args.lr, global_step, decay_steps=1, decay_rate=5e-5)
    optimizer = tf.train.AdamOptimizer(rate)
    train_op = optimizer.minimize(ce_loss, global_step=global_step)
    
    model_summary()

    ## training
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Creates a file writer for the log directory.
        log_path = os.path.join(args.log_path,
                                '_'.join((args.test_name, 'lr'+str(args.lr))))
        file_writer_train = tf.summary.FileWriter(os.path.join(log_path, "train"), sess.graph)
        file_writer_test = tf.summary.FileWriter(os.path.join(log_path, "test"), sess.graph)

        # store variables
        tf.summary.image("Support a image", support_a_reshape[:4], max_outputs=4)
        tf.summary.image("Query b image", query_b_reshape[:4], max_outputs=4)
        tf.summary.scalar("Classification loss", ce_loss)
        tf.summary.scalar("Accuracy", acc)
        merged = tf.summary.merge_all()

        sess.run(init)
        
        train_domain_index = [0, 1, 3]
        test_domain = dataset.domains[2]
        for i_iter in range(args.n_iter):
            # get domain A and B
            random.shuffle(train_domain_index)
            domain_a = dataset.domains[train_domain_index[0]]
            domain_b = dataset.domains[train_domain_index[1]]
            
            # get categories
            categories = random.sample(dataset.categories, k=n_way)

            # get task from domain A and B, and the task contains support and query
            task_a = dataset.get_task(domain_a, categories, n_shot, n_query)
            task_b = dataset.get_task(domain_b, categories, n_shot, n_query)
            
            labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)

            # training                 
            _, _ = sess.run([train_op, global_step], feed_dict={
                support_a: task_a['support'],
                query_a: task_a['query'],
                support_b: task_b['support'],
                query_b: task_b['query'],
                query_b_y: labels
            })

            if i_iter % 100 == 0:
                
                summary_train, train_loss, train_acc = sess.run([merged, ce_loss, acc], feed_dict={
                    support_a: task_a['support'],
                    query_a: task_a['query'],
                    support_b: task_b['support'],
                    query_b: task_b['query'],
                    query_b_y: labels
                })
                
                # evaluation on classification of target task
                test_task = dataset.get_task(test_domain, categories, n_shot, n_query)
                
                summary_test, test_loss, test_acc = sess.run([merged, ce_loss, acc], feed_dict={
                    support_a: test_task['support'],
                    query_a: test_task['query'],
                    support_b: test_task['support'],
                    query_b: test_task['query'],
                    query_b_y: labels
                })

                # log all variables
                file_writer_train.add_summary(summary_train, global_step=i_iter)
                file_writer_test.add_summary(summary_test, global_step=i_iter)

                print('Iteration: %d, train cost: %g, test cost: %g, train acc: %g, test acc: %g' %
                      (i_iter+1, train_loss, test_loss, train_acc, test_acc))

        file_writer_train.close()
        file_writer_test.close()



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
