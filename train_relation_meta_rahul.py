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
from src.model_rahul import PrototypeNet, RelationNet

# miscellaneous
import gc


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
parser.add_argument('--multi_domain', default=True, type=bool)


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

    # reshape
    support_x_reshape = tf.reshape(support_x, [n_way * n_shot, img_h, img_w, 3])
    query_x_reshape = tf.reshape(query_x, [n_way * n_query, img_h, img_w, 3])
    support_a_reshape = tf.reshape(support_a, [n_way * n_shot, img_h, img_w, 3])
    query_b_reshape = tf.reshape(query_b, [n_way * n_query, img_h, img_w, 3])

    # Meta-RelationNet
    print("=== Build model...")
    print("First learning rate: ", alpha)
    print("beta: ", beta)
    print("Second learning rate: ", gamma)
    print("Decay second learning rate: ", decay)
    model = RelationNet(n_way, n_shot, n_query, alpha=alpha, beta=beta, gamma=gamma, decay=decay,
                        backbone='resnet', is_training=is_training)
    model.train_meta(support_x=support_x_reshape, query_x=query_x_reshape, 
                     support_a=support_a_reshape, query_b=query_b_reshape)

    model_summary()

    # saver for saving session
    saver = tf.train.Saver()
    log_path = os.path.join('logs', args.log_path, '_'.join(
        (args.test_name, 'alpha'+str(alpha), 'beta'+str(beta), 'gamma'+str(gamma), 'decay'+str(decay))))
    
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
    mini = MiniImageNet()

    # load PACS
    pacs = Pacs()

    # load CUB data
    cub = Cub(mode='test')
    
    ## training
    with tf.Session() as sess:

        # Creates a file writer for the log directory.
        file_writer_train = tf.summary.FileWriter(os.path.join(log_path, "train"), sess.graph)
        file_writer_test = tf.summary.FileWriter(os.path.join(log_path, "test"), sess.graph)

        # store variables
        tf.summary.image("Support x image", support_x_reshape[:4], max_outputs=4)
        tf.summary.image("Query x image", query_x_reshape[:4], max_outputs=4)
        tf.summary.scalar("x loss", model.x_loss)
        tf.summary.scalar("x acc", model.x_acc)

        tf.summary.image("Support a image", support_a_reshape[:4], max_outputs=4)
        tf.summary.image("Query b image", query_b_reshape[:4], max_outputs=4)
        tf.summary.scalar("ab loss", model.ab_loss)
        tf.summary.scalar("ab acc", model.ab_acc)        
        
        merged = tf.summary.merge_all()

        # get test unseen domains
        domains = list(pacs.data_dict.keys())
        test_domain = 'sketch'
        domains.remove(test_domain)

        domain_a = domains[0]
        domain_b = domains[1]
        categories = list(pacs.data_dict[domain_a].keys())
        
        print("=== Start training...")
        sess.run(init)
        restore_from_checkpoint(sess, saver, lastest_checkpoint)
        for i_iter in range(args.n_iter):

            # mini-imagenet ======================================================================
            curr_support_x, curr_query_x = mini.get_task(n_way, n_shot, n_query)

            # PACS ===============================================================================
            # get domain A and B
            if args.multi_domain:
                domain_a, domain_b = random.sample(domains, k=2)

            # get categories
            selected_categories = random.sample(categories, k=n_way)

            # get support and query from domain A and B
            curr_support_a, _ = pacs.get_task(domain_a, selected_categories, n_shot, n_query)
            _, curr_query_b = pacs.get_task(domain_b, selected_categories, n_shot, n_query)
            
            # training                 
            sess.run([model.meta_op], feed_dict={
                support_x: curr_support_x,
                query_x: curr_query_x,
                support_a: curr_support_a,
                query_b: curr_query_b,                
                is_training: True
            })

            # evaluation
            if i_iter % 50 == 0:
                curr_support_x, curr_query_x = mini.get_task(n_way, n_shot, n_query, aug=False, mode='val')
                curr_support_a, curr_query_a = pacs.get_task(
                    test_domain, selected_categories, n_shot, n_query, aug=False)
                
                summary_train, train_x_loss, train_x_acc, train_ab_loss, train_ab_acc = \
                    sess.run([merged, model.x_loss, model.x_acc, model.ab_loss, model.ab_acc], 
                        feed_dict={
                            support_x: curr_support_x,
                            query_x: curr_query_x,
                            support_a: curr_support_a,
                            query_b: curr_query_a,                
                            is_training: False
                        })

                # get task from unseen domain (CUB)
                cub_support, cub_query = cub.get_task(n_way, n_shot, n_query)
                summary_test, cub_loss, cub_acc = \
                    sess.run([merged, model.ab_loss, model.ab_acc], 
                        feed_dict={
                            support_x: curr_support_x,
                            query_x: curr_query_x,
                            support_a: cub_support,
                            query_b: cub_query,                
                            is_training: False
                        })

                # log all variables
                file_writer_train.add_summary(summary_train, global_step=i_iter)
                file_writer_test.add_summary(summary_test, global_step=i_iter)

                print('Iteration: %d, train x : [%f, %f], train ab: [%f, %f]' % 
                    (i_iter+1, train_x_loss, train_x_acc, train_ab_loss, train_ab_acc))
                print('==> CUB: [%f, %f]' % (cub_loss, cub_acc))

            # save session every 2000 iteration
            if (i_iter+1) % 2000 == 0:    
                saver.save(sess, checkpoint_file, global_step=(i_iter+1))
                print('Save session at step %d' % (i_iter+1))

        file_writer_train.close()
        file_writer_test.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
