from keras import backend as K
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import pdb

# Use this loss function with median frequency coefficients weights
# for class balance
def weighted_loss(num_classes, coefficients, labels, logits):
    """Implements weighted categorical cross entropy objective function
    Arguments 
    =========
        Input:
                num_classes: The number of classes
                coefficients: Vector which contains alpha coefficients
    """    
    coefficients = tf.constant(coefficients)
    num_classes = tf.constant(num_classes)

    def loss(labels, logits):
        with tf.name_scope('loss_1'):
            logits = tf.reshape(logits, (-1, num_classes))
            #print(tf.shape(logits))
            epsilon = tf.constant(value=1e-10)

            logits = logits + epsilon
            # consturct one-hot label array
            #pdb.set_trace()
            labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))
            softmax = tf.nn.softmax(logits)
            #pdb.set_trace()
            log_inter = tf.log(softmax+epsilon)
            mul1 = labels*log_inter
            term1 = tf.multiply(mul1, coefficients)
            cross_entropy = -tf.reduce_sum(term1, reduction_indices=[1]) # error here
            #cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), coefficients), reduction_indices=[1])
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

            tf.add_to_collection('losses', cross_entropy_mean)
            loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            #loss = cross_entropy_mean
        return loss
    return loss(labels, logits)
