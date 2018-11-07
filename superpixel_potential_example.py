# sp tem
import tensorflow as tf
import pdb

ses = tf.InteractiveSession()

sp_map = tf.constant([[1,1,1,2,2],
                      [1,1,1,2,2],
                      [3,3,4,4,5],
                      [3,3,4,4,5]])

nb_classes = 3

# replicate the sp_map m times and have the shape of [rows,cols,m), where m in the number of labels
extended_sp_map = tf.stack([sp_map] * nb_classes)

q_vals = tf.random_uniform(shape=[nb_classes,4,5])

# This will put True where the max prob label, False otherwise:
cond_max_label = tf.equal(q_vals, tf.reduce_max(q_vals,axis=0))

print(ses.run(q_vals))

# These would be the learned parameters:
w_low = tf.constant(0.1)
#w_low_m = tf.constant([0.11,0.10,0.09])
w_low_m = tf.constant([[0.11,0.,0.],
                       [0., 0.10, 0.],
                       [0., 0., 0.09]])
w_high = tf.constant(0.9)

#sp_indx = 4
sp_out = tf.zeros(q_vals.shape)
sp_cond = tf.constant(False, shape=q_vals.shape)
for sp_indx in range(1,6):

    #print(sp_indx)
    # This will put True where sp index is sp_indx, False otherwise:
    cond_sp_indx = tf.equal(extended_sp_map,sp_indx)

    # This is tensor T, where the dominant label for sp_indx superpixel is:
    T = tf.logical_and(cond_max_label,cond_sp_indx)

    #pdb.set_trace()
    sp_cond = tf.logical_or(sp_cond,T)
    #The potental added to all pixels in sp_indx:
    #sp_out += w_low * tf.multiply(tf.to_float(T),q_vals) + w_high * tf.multiply(tf.to_float(tf.logical_not(T)),q_vals)
    #sp_out += w_low * tf.to_float(T) + w_high * tf.to_float(tf.logical_not(T))

    # show
    #print(ses.run(T))

print(ses.run(sp_cond))
first_term = tf.multiply(tf.to_float(sp_cond),q_vals)
first_term_resp = tf.matmul(w_low_m,tf.reshape(first_term, (nb_classes,-1)))
first_term_resp_back = tf.reshape(first_term_resp, (nb_classes, 4, 5))

sp_out =  first_term_resp_back + w_high * tf.multiply(tf.to_float(tf.logical_not(sp_cond)),q_vals)
# sp_out = tf.reduce_sum(sp_out)
print(ses.run(sp_out))
