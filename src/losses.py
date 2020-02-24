import tensorflow as tf


def compute_classification_loss(labels, outputs):
    with tf.name_scope('loss'):
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=outputs)
    return cross_entropy

def compute_intra_loss(prototype, fc1_query, query_labels):
    '''
    Compute intra-class loss

    Args
        prototype: [5, 512]
        fc1_query: [75, 512]
        query_labels: [75, 5]
        
    Return
        scalar
    '''
    num_class = prototype.shape[0]

    loss_per_class = []
    for i_class in range(num_class):
        i_class_prototype = prototype[i_class]  # i-th prototype

        i_class_mask = tf.equal(query_labels[:, i_class], 1)
        fc1_i_class = tf.boolean_mask(fc1_query, i_class_mask)  # all queries for i-th class

        i_class_query_norm = tf.norm(fc1_i_class - i_class_prototype, axis=1)
        loss_per_class.append(tf.reduce_sum(i_class_query_norm))

    intra_class_loss = tf.reduce_sum(loss_per_class)
    return intra_class_loss

def compute_inter_loss(prototype, fc1_query, query_labels):
    '''
    Compute inter-class loss

    Args
        prototype: [5, 512]
        fc1_query: [75, 512]
        query_labels: [75, 5]

    Return
        scalar
    '''
    num_class = prototype.shape[0]

    loss_per_class = []
    for i_class in range(num_class):
        i_class_prototype = prototype[i_class]  # i-th prototype

        # select all queries that are not i-class
        i_class_mask = tf.equal(query_labels[:, i_class], 0)
        fc1_not_i_class = tf.boolean_mask(fc1_query, i_class_mask)

        i_class_other_query_norm = tf.norm(fc1_not_i_class - i_class_prototype, axis=1)
        i_class_loss = tf.exp(tf.negative(i_class_other_query_norm))  # exp(-d(prototype, y))
        i_class_loss = tf.log(tf.reduce_sum(i_class_loss))  # log(sum(loss_per_class))
        loss_per_class.append(i_class_loss)

    inter_class_loss = tf.reduce_sum(loss_per_class)
    return inter_class_loss
