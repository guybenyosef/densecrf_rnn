# sp tem
import tensorflow as tf

s = tf.InteractiveSession()

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
# This will put True where where sp index is sp_indx, False otherwise:
sp_indx = 4
cond_sp_indx = tf.equal(extended_sp_map,sp_indx)

# This is tensor T, where the dominant label for sp_indx superpixel is:
T = tf.logical_and(cond_max_label,cond_sp_indx)

s.run(q_vals)
s.run(T)

# These would be the learned parameters:
w_low = tf.constant(0.25)
w_high = tf.constant(0.75)
#The potental added to all pixels in sp_indx:
sp_out = tf.reduce_sum(w_low * tf.multiply(tf.to_float(T),q_vals) + w_high * tf.multiply(tf.to_float(tf.logical_not(T)),q_vals))

s.run(sp_out)






ar_max = tf.argmax(aa,dimension=0)
bb = tf.zeros([3, 5, 3])



a = tf.constant([0.3, 0.5, 0.79, 0.79, 0.11])

out = tf.sparse_to_dense(tf.argmax(a),tf.cast(tf.shape(a), dtype=tf.int64), tf.reduce_max(a))

cond = tf.equal(a, tf.reduce_max(a))