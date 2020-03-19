# deep learning framework
import tensorflow as tf
import tensorflow.contrib.slim as slim

# computation
import numpy as np
import random
from scipy.stats import sem, t
from scipy import mean

# io
import os
import argparse

# customerized 
from src.load_data import Pacs, Cub
from src.model import PrototypeNet, RelationNet

# miscellaneous
import gc


parser = argparse.ArgumentParser()
parser.add_argument('--log_path', default='baseline', type=str)
parser.add_argument('--test_name', default='test', type=str)
parser.add_argument('--n_way', default=5, type=int)
parser.add_argument('--n_shot', default=5, type=int)
parser.add_argument('--n_query', default=15, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--n_iter', default=40000, type=int)
parser.add_argument('--multi_domain', default=True, type=bool)


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def main(args):
    img_w = 84
    img_h = 84
    
    n_way = args.n_way
    n_shot = args.n_shot
    n_query = args.n_query
    
    dataset = Pacs()
    
    ## establish training graph
    # inputs placeholder (support and query randomly sampled from two domain)
    support_a = tf.placeholder(tf.float32, shape=[n_way, n_shot, None, None, 3])  
    query_b = tf.placeholder(tf.float32, shape=[n_way, n_query, None, None, 3])
    is_training = tf.placeholder(tf.bool)

    # reshape
    support_a_reshape = tf.reshape(support_a, [n_way * n_shot, img_h, img_w, 3])
    query_b_reshape = tf.reshape(query_b, [n_way * n_query, img_h, img_w, 3])

    # feature extractor
    init_lr = args.lr
    model = RelationNet(n_way, n_shot, n_query, backbone='conv4', learning_rate=init_lr, is_training=is_training)
    train_op, train_loss, train_acc, global_step = model.train(support=support_a_reshape, query=query_b_reshape)
    
    model_summary()

    ## establish test graph
    model_test = RelationNet(n_way, n_shot, n_query, backbone='conv4', learning_rate=init_lr, is_training=is_training)
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
    cub = Cub()
    
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

        # get test unseen domains
        domains = list(dataset.data_dict.keys())
        test_domain = 'sketch'
        domains.remove(test_domain)  # drop test domain data

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
            
            # training                 
            _, step = sess.run([train_op, global_step], feed_dict={
                support_a: support,
                query_b: query,
                is_training: True
            })

            # evaluation
            if i_iter % 50 == 0:
                summary_train, loss, acc = sess.run([merged, test_loss, test_acc], feed_dict={
                    support_a: support,
                    query_b: query,
                    is_training: False
                })

                # get task from unseen domain in PACS
                support, query = dataset.get_task(test_domain, selected_categories, n_shot, n_query)
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
                print('eval %s cost: %g, eval %s acc: %g' %
                      (test_domain, val_loss, test_domain, val_acc))
                print('cub cost: %g, cub acc: %g' % (cub_loss, cub_acc))

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
